import cv2 as cv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from dotenv import load_dotenv

SKIP_FRAME = 1
# Load environment variables
load_dotenv()
ROOT_PATH = os.getenv('VN_SINGLE_WORD_DATA_PATH')
SAVE_PATH = os.getenv('VN_SINGLE_WORD_SAVE_PATH')

def format_len(num):
    """Format number to a fixed-length string."""
    return f'{num:06d}'

def read_video(row, root_path, save_path):
    """Process a single video and extract frames."""
    try:
        file, label, video_lb_id = row
        name = file.split('.')[0]
        video_path = f'{root_path}\\Data\\{file}'
        save_dir = f'{save_path}\\{name}'

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Read and process video
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_LANCZOS4)
            if num %  SKIP_FRAME== 0:
                cv.imwrite(f'{save_dir}\\{format_len(num // SKIP_FRAME)}.jpg', frame)
            num += 1

        cap.release()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def extract_frame_from_vid(data_type='1_200', num_workers=12):
    """Extract frames from videos using multiprocessing."""
    csv_path = f'{ROOT_PATH}\\label_{data_type}\\full_data_{data_type}_center_ord1.csv'
    df = pd.read_csv(csv_path)

    # Prepare arguments for multiprocessing
    rows = [df.iloc[i] for i in range(len(df))]

    # Use partial to pass additional arguments to read_video
    process_video = partial(read_video, root_path=ROOT_PATH, save_path=SAVE_PATH)

    # Process videos in parallel with a progress bar
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap(process_video, rows), total=len(rows), desc="Processing videos"))

if __name__ == '__main__':
    extract_frame_from_vid(data_type='1_200', num_workers=8)