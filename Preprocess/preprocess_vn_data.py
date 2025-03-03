import cv2 as cv
import os
import numpy as np
import pandas as pd
import glob
import pdb
import re
import pickle
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.getenv("VN_DATA_PATH")
SAVE_PATH = os.getenv("VN_SAVE_PATH")
def format_len(num):
    if num < 10:
        return f'00000{num}'
    elif num < 100:
        return f'0000{num}'
    elif num < 1000:
        return f'000{num}'
    elif num < 10000:
        return f'00{num}'

def read_video(video_path, word, id):
    cap = cv.VideoCapture(video_path)
    dir = f'{SAVE_PATH}\\{id}'
    # check if the directory exists
    if not os.path.exists(dir):
        os.makedirs(dir)
    # print(dir)
    num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, 220:1060]
        frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_LANCZOS4)
        if num % 5 == 0:
            cv.imwrite(f'{dir}\\{format_len(num // 5)}.jpg', frame)
        num += 1
    # print(num)
    cap.release()
    cv.destroyAllWindows()
    
list_paths = os.listdir(ROOT_PATH)

for path in tqdm(list_paths):
    video_path = f'{ROOT_PATH}\\{path}'
    # print(video_path)
    id = path.split('_')[0]
    word = path.split('_')[1].split('.')[0]
    read_video(video_path, word, id)
    
    

    