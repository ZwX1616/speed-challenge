
# coding: utf-8

import numpy as np
import pandas as pd
import os
import csv
import skvideo.io
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH =  os.path.join(CURRENT_DIR, 'data')
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
PREPARED_DATA_PATH = os.path.join(DATA_PATH, 'prepared-data')
PREPARED_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'train_imgs')
PREPARED_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'test_imgs')

TRAIN_FRAMES = 20400
TEST_FRAMES = 10798


from multiprocessing import Lock
tqdm.set_lock(Lock())  # manually set internal lock

train_y = list(pd.read_csv(os.path.join(DATA_PATH, 'train.txt'), header=None, squeeze=True))

assert(len(train_y)==TRAIN_FRAMES)

def prepare_dataset(video_loc, img_folder, dataset_type):
    meta_dict = {}
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    tqdm.write('reading in video file...')
    cap = skvideo.io.vread(video_loc)
    tqdm.write('constructing dataset...')
    for idx, frame in enumerate(tqdm(cap)):    
        img_path = os.path.join(img_folder, str(idx)+'.jpg')
        frame_speed = float('NaN') if dataset_type == 'test' else train_y[idx]
        meta_dict[idx] = [img_path, idx, frame_speed]
        skvideo.io.vwrite(img_path, frame)
    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    meta_df.columns = ['image_path', 'image_index', 'speed']
    
    tqdm.write('writing meta to csv')
    meta_df.to_csv(os.path.join(PREPARED_DATA_PATH, dataset_type+'_meta.csv'), index=False)
    
    return "done dataset_constructor"


if __name__ == "__main__":
    prepare_dataset(TRAIN_VIDEO, PREPARED_IMGS_TRAIN, 'train')
    prepare_dataset(TEST_VIDEO, PREPARED_IMGS_TEST, 'test')


