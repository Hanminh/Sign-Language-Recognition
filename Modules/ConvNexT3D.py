import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model  
import torch.utils.model_zoo as  model_zoo
from torch.utils.checkpoint import checkpoint
import torchvision.models as models

class AttentionPool2D(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim= None, cluster= 1):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim)
        self.num_heads = num_heads
        self.cluster = cluster
        self.query = nn.Parameter(torch.rand(self.cluster, 1, embed_dim), requires_grad= True)

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = x.flatten(start_dim= 3).permute(3, 0, 2, 1).reshape(-1, N*T, C).contiguous()
        x, _ = F.multi_head_attention_forward(
            query= self.query.repeat(1, N*T, 1),
            key= x,
            value= x,
            num_heads= self.num_heads,
            q_proj_weight= self.q_proj.weight,
            k_proj_weight= self.k_proj.weight,
            v_proj_weight= self.v_proj.weight,
            in_proj_weight= None,
            in_proj_bias= torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k= None,
            bias_v= None,
            add_zero_attn= False,
            dropout_p= 0.0,
            out_proj_weight= self.c_proj.weight,
            out_proj_bias= self.c_proj.bias,
            use_separate_proj_weight= True,
            training= self.training,
            need_weights= False,
            embed_dim_to_check= x.shape[-1]
        )
        return x.view(self.cluster, N, T, C).contiguous().permute(1, 3, 2, 0) # NCTE

class TemporalWeighting(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = input_size // 16
        self.conv_transform =nn.Conv1d(input_size, hidden_size, kernel_size= 1, stride= 1, padding= 0)
        self.conv_back = nn.Conv1d(hidden_size, input_size, kernel_size= 1, stride= 1, padding= 0)
        
        self.num = 3
        self.conv_enhance = nn.ModuleList(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride= 1, padding= int(i+1), groups= hidden_size, dilation= int(i+1)) \
                for i in range(self.num)
        )
        self.weights = nn.Parameter(torch.ones(self.num) / self.num, requires_grad= True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad= True)
        self.relu = nn.ReLU(inplace= True)
        
    def forward(self, x):
        out = self.conv_transform(x.mean(-1).mean(-1))
        aggregated_out = 0
        for i in range(self.num):
            aggregated_out += self.conv_enhance[i](out) * self.weights[i]
        out = self.conv_back(aggregated_out)
        return x * (F.sigmoid(out.unsqueeze(-1).unsqueeze(-1)) - 0.5) * self.alpha
    
class UnfoldNeighbour(nn.Module):
    def __init__(self, window_size= 9, window_stride= 1, window_dilation= 1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation
        
        self.padding = (window_size + (window_size - 1) * (window_dilation - 1) - 1) // 2
        self.unfold = nn.Unfold(
            kernel_size= (window_size, 1),
            dilation= (window_dilation, 1),
            padding= (self.padding, 0),
            stride= (window_stride, 1)
        )
        
    def forward(self, x):
        N, C, T, H, W = x.shape
        x = x.view(N, C, T, H*W)
        x = self.unfold(x) # N, C*window_size, T, H*W
        x = x.view(N, C, self.window_size, T, H, W).permute(0, 1, 3, 2, 4, 5).reshape(N, C, T, self.window_size, H, W)
        return x

class Get_Correlation(nn.Module):
    def __init__(self, channels, neighbors= 3):
        super().__init__()
        
        # Correlation layer
        self.neighbors = neighbors
        reduction_channel = channels // 16
        
        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size= 1, bias= False)
        self.clusters = 1
        self.weights2 = nn.Parameter(torch.ones(self.neighbors * 2) / (self.neighbors * 2), requires_grad= True)
        self.unfold = UnfoldNeighbour(window_size= 2 * self.neighbors + 1)
        self.weights3 = nn.Parameter(torch.ones(3) / 3, requires_grad= True)
        self.weights4 = nn.Parameter(torch.ones(3) / 3, requires_grad= True)
        self.attpool = AttentionPool2D(embed_dim= channels, num_heads= 1, output_dim= channels, cluster= self.clusters)
        
        # Identification layer
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size= 1, bias= False)
        self.spatial_aggregation1= nn.Conv3d(reduction_channel, reduction_channel, kernel_size= (9, 3, 3), padding= (4, 1, 1), groups= reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size= (9, 3, 3), padding= (4, 2, 2), dilation= (1, 2, 2), groups= reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size= (9, 3, 3), padding= (4, 3, 3), dilation= (1, 3, 3), groups= reduction_channel)
        self.weights= nn.Parameter(torch.ones(3) / 3, requires_grad= True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size= 1, bias= False)
        
    def forward(self, x):
        N, C, T, H, W = x.shape
        def clustering(query, key):
            affinities = torch.einsum('bctp, bctl-> btpl', query, key)
            return torch.einsum('bctl, btpl->bctp', key, F.sigmoid(affinities) - 0.5)
        
        x_mean = x.mean(3, keepdim= True).mean(4, keepdim= False)
        x_max = x.max(-1, keepdim= False)[0].max(-1, keepdim= True)[0]
        x_att = self.attpool(x)
        
        x2 = self.down_conv2(x)
        upfold = self.unfold(x2)
        upfold = (torch.concat([upfold[:, :, :, :self.neighbors], upfold[:, :, :, self.neighbors+1:]], dim= 3) * self.weights2.view(1, 1, 1, -1, 1, 1)).view(N, C, T, -1)
        x_mean = x_mean * self.weights4[0] + x_max * self.weights4[1] + x_att * self.weights4[2]
        x_mean = clustering(x_mean, upfold)
        features = x_mean.view(N, C, T, self.clusters, 1)
        
        x_down = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x_down) * self.weights[0] + \
            self.spatial_aggregation2(x_down) * self.weights[1] + \
            self.spatial_aggregation3(x_down) * self.weights[2]
        
        aggregated_x = self.conv_back(aggregated_x)
        
        features = features * (F.sigmoid(aggregated_x) - 0.5)
        return features
    
class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.data_format = data_format
    def forward(self, x):
        # print(x.shape)
        if self.data_format == "channels_last":
            return self.layer_norm(x)
        elif self.data_format == "channels_first":
            x = x.permute(0, 2, 3, 4, 1).contiguous() # B, C, T, H, W -> B, T, H, W, C
            # view x as (B *T, H, W, C)
            batch_size, T, H, W, C = x.shape
            x = x.view(-1, H, W, C).contiguous() # (B * T, H, W, C)
            x = self.layer_norm(x)
            x = x.view(batch_size, T, H, W, C).contiguous()
            x = x.permute(0, 4, 1, 2, 3)
            return x
        
class Block3D(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value= 1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size= (1, 7, 7), padding= (0, 3, 3), groups= dim)
        self.norm = LayerNorm3D(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # B, C, T, H, W -> B, T, H, W, C
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 4, 1, 2, 3) # B, T, H, W, C -> B, C, T, H, W
        x = input + self.drop_path(x)
        return x

class ConvNeXt3D(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 num_neighbors= [1, 3, 5],
                 ):
        super().__init__()

        self.correlation = nn.ModuleList()
        for i in range(3):
            self.correlation.append(Get_Correlation(channels= dims[i+1], neighbors= num_neighbors[i]))
        
        self.temporal_weight = nn.ModuleList()
        for i in range(3):
            self.temporal_weight.append(TemporalWeighting(input_size= dims[i+1]))
        
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad= True)
        self.avgpool = nn.AvgPool2d(kernel_size= 7, stride= 1)
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size= (1, 4, 4), stride= (1, 4, 4,)),
            LayerNorm3D(dims[0], eps= 1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm3D(dims[i], eps= 1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size= (1, 2, 2), stride= (1, 2, 2)),
            )
            self.downsample_layers.append(downsample_layer)
            
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                    *[Block3D(dim=dims[i], drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]       
            )
            self.stages.append(stage)
            cur += depths[i]
            
        self.norm = LayerNorm3D(dims[-1], eps= 1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.zero_()
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            # nn.init.constant_(m.bias, 0)
            
    def forward_features(self, x):
        for i in range(4):
            # print(f"Before stage {i}: ", x.shape)
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i > 0:
                x = x + self.correlation[i-1](x) * self.alpha[i-1]
                x = x + self.temporal_weight[i-1](x)
            # print(f"After stage {i}: ", x.shape)
        return self.norm(x.mean([-2, -1]).permute(0, 2, 1)) 
    def forward(self , x):
        x = self.forward_features(x)
        x = x.view(-1, x.shape[-1]).contiguous()  # Flatten the features
        x = self.head(x)
        return x

model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

def inflate_weights(model, pretrained_state):
    model_state = model.state_dict()
    for name, param in pretrained_state.items():
        if name in model_state:
            if 'conv' in name or 'downsample_layers' in name:
                # Inflate 2D conv weights (C, C', H, W) to 3D (C, C', 1, H, W)
                param_3d = param.unsqueeze(2)
                model_state[name].copy_(param_3d)
            else:
                model_state[name].copy_(param)
        else:
            # print(f"Skipping {name} as it is not in the 3D model.")
            pass
    
    model.load_state_dict(model_state)
    # return model
    
@register_model
def convnext3d_tiny(pretrained= True, num_neighbors= [1, 3, 5]):
    model = ConvNeXt3D(
        in_chans=3, num_classes=1000, 
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.1, 
        layer_scale_init_value=1e-6, head_init_scale=1.,
        num_neighbors= num_neighbors
    )
    if pretrained:
        pretrained_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        inflate_weights(model, pretrained_model.state_dict())
    return model

# test
def test():
    model = convnext3d_tiny(pretrained= True)
    x = torch.randn(2, 8, 3, 224, 224)
    x = x.permute(0, 2, 1, 3, 4) # [2, 3, 8, 224, 224]

    y = model(x)
    print(y.shape) # expect [2, 1000]
    
# test()