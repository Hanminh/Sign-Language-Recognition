import cv2 as cv
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ROOT_PATH = os.getenv("VN_DATA_PATH")
SAVE_PATH = os.getenv("VN_SAVE_PATH")


# skip frame 
SKIP_FRAME = 3

# Padding function
def format_len(num):
    return f'{num:06d}'

# Video processing function
def read_video(filename):
    video_path = os.path.join(ROOT_PATH, filename)
    id = filename.split('_')[0]
    word = filename.split('_')[1].split('.')[0]

    cap = cv.VideoCapture(video_path)
    save_dir = os.path.join(SAVE_PATH, id)

    os.makedirs(save_dir, exist_ok=True)

    num = 0
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:, 220:1060]
        frame = cv.resize(frame, (256, 256), interpolation=cv.INTER_LANCZOS4)
        if num % 3 == 0:
            cv.imwrite(os.path.join(save_dir, f'{format_len(num // 3)}.jpg'), frame)
        num += 1

    cap.release()
    cv.destroyAllWindows()
    return f"Processed {filename}"

if __name__ == '__main__':
    list_paths = os.listdir(ROOT_PATH)

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(read_video, list_paths), total=len(list_paths)):
            pass
