
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import os
import csv
import skvideo.io
from processing import process
from tqdm import tqdm
from multiprocessing import Lock

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH =  os.path.join(CURRENT_DIR, 'data')
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
STOP_VIDEO = os.path.join(DATA_PATH, 'stop.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
PREPARED_DATA_PATH = os.path.join(DATA_PATH, 'prepared-data')
PREPARED_IMGS_STOP = os.path.join(PREPARED_DATA_PATH, 'stop_imgs')
FLOW_IMGS_STOP = os.path.join(PREPARED_DATA_PATH, 'flow_stop_imgs')
PREPARED_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'train_imgs')
FLOW_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'flow_train_imgs')
PREPARED_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'test_imgs')
FLOW_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'flow_test_imgs')

SIZE = (100, 100)


def prepare_dataset(video_path, frame_folder, flow_folder, name, speeds=None):
    tqdm.set_lock(Lock())  # manually set internal lock
    #Step 1, Extract frames and speed
    dataframe_dict = {}
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    print("Reading the video file")
    video_sk = skvideo.io.vread(video_path)
    print("Extracting the frames")
    for index, frame in enumerate(tqdm(video_sk)):    
        saving_path = os.path.join(frame_folder, str(index)+'.jpg')
        if speeds is None:
            speed = 0
        else:
            speed = speeds[index]
        dataframe_dict[index] = [saving_path, index, speed]
        skvideo.io.vwrite(saving_path, frame)
    
    processed_dataframe = pd.DataFrame.from_dict(dataframe_dict, orient='index')
    processed_dataframe.columns = ['frame_path', 'frame_index', 'speed']
    print("Saving the dataframe")
    processed_dataframe.to_csv(os.path.join(PREPARED_DATA_PATH, name +'_meta.csv'), index=False)
    #Step 2, compute optical flow between frames and average the speed
    flow_dict = {}
    if not os.path.exists(flow_folder):
        os.makedirs(flow_folder)
    print("Computing the optical flow")
    for index in tqdm(range(len(processed_dataframe ) - 1)):
        idx1 = index
        idx2 = index + 1
        frame1 = processed_dataframe.iloc[[idx1]]
        frame2 = processed_dataframe.iloc[[idx2]]

        assert(frame2['frame_index'].values[0] - frame1['frame_index'].values[0] == 1)
        assert(frame2['frame_index'].values[0] > frame1['frame_index'].values[0])

        frame1_path = frame1['frame_path'].values[0]
        frame1_speed = frame1['speed'].values[0]
        frame2_path = frame2['frame_path'].values[0]
        frame2_speed = frame2['speed'].values[0]

        flow = process(frame1_path, frame2_path, SIZE)

        flow_path = os.path.join(flow_folder, str(index) + '.png') 

        cv2.imwrite(flow_path, flow)

        speed = np.mean([frame1_speed, frame2_speed]) 
        flow_dict[index] = [flow_path, speed]

    flow_dataframe = pd.DataFrame.from_dict(flow_dict, orient='index')
    flow_dataframe.columns = ['flow_path', 'speed']
    print("Saving the flow dataframe")
    flow_dataframe.to_csv(os.path.join(PREPARED_DATA_PATH, name +'_flow_meta.csv'), index=False)


if __name__ == "__main__":
    speeds_train = list(pd.read_csv(os.path.join(DATA_PATH, 'train.txt'), header=None, squeeze=True))
    speeds_stop = [0] * 480
    print("==Processing Stop==")
    prepare_dataset(STOP_VIDEO, PREPARED_IMGS_STOP, FLOW_IMGS_STOP, 'stop', speeds_stop)
    print("==Processing Train==")
    prepare_dataset(TRAIN_VIDEO, PREPARED_IMGS_TRAIN, FLOW_IMGS_TRAIN, 'train', speeds_train)
    print("==Processing Test==")
    prepare_dataset(TEST_VIDEO, PREPARED_IMGS_TEST, FLOW_IMGS_TEST, 'test')


