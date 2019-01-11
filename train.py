
# coding: utf-8
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import processing
import argparse
from model import build_model
import os
import pandas as pd
import pickle
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend.tensorflow_backend as KTF

MINI_BATCH_SIZE = 16
EPOCHS = 40
VERSION = 3

SIZE = (100, 100)
CHANNEL = 2
WIDTH = SIZE[0]
HEIGHT = SIZE[1]


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH =  os.path.join(CURRENT_DIR, 'data')
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
PREPARED_DATA_PATH = os.path.join(DATA_PATH, 'prepared-data')
PREPARED_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'train_imgs')
FLOW_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'flow_train_imgs')
PREPARED_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'test_imgs')
FLOW_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'flow_test_imgs')
SAVES = os.path.join(CURRENT_DIR, 'saves')
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, 'saves', str(VERSION), str(EPOCHS))
WEIGHTS_PATH = os.path.join(MODEL_SAVE_PATH, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
HISTORY_PATH = os.path.join(MODEL_SAVE_PATH, 'history.p')
TENSORBOARD = os.path.join(MODEL_SAVE_PATH, 'tensorboard')




def main(args):
    # Create the save folder
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    train_meta = pd.read_csv(args.train_flow_meta_file)
    print('shape: ', train_meta.shape)
    
    #Load the data
    X = np.empty((len(train_meta), HEIGHT, WIDTH, CHANNEL))
    Y = np.empty((len(train_meta), 1))
    for index, row in tqdm(train_meta.iterrows()):
        frame = cv2.imread(row["flow_path"])
        frame = cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)
        #Normalize
        frame = frame / 40
        #Drop the useless channel
        frame = frame[:,:,[0,2]]
        X[index,:,:,:] = frame
        Y[index] = row["speed"]
    #Shuffle the data
    randomize = np.arange(len(train_meta))
    np.random.shuffle(randomize)
    X = X[randomize]
    Y = Y[randomize]

    modelCheckpoint = ModelCheckpoint(WEIGHTS_PATH,
                                      monitor='val_loss',
                                      save_best_only=False,
                                      mode='min',
                                      verbose=1,
                                      save_weights_only=True)

    tensorboard = TensorBoard(log_dir=TENSORBOARD, histogram_freq=0,
                              write_graph=True, write_images=True)

    callbacks_list = [modelCheckpoint, tensorboard]


    model = build_model(HEIGHT, WIDTH, CHANNEL)


    model.summary()

    history = model.fit(
        X,
        Y,
        batch_size=MINI_BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1,
        validation_split=args.split)


    pickle.dump(history.history, open(HISTORY_PATH, "wb"))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_flow_meta_file",
                        help="Train Flow Images meta file")
    parser.add_argument("--split", type=float, default=0,
                        help="Percentage of data used to validate the model")
    args = parser.parse_args()
    main(args)
