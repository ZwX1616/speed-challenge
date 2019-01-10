
# coding: utf-8
import numpy as np
import tensorflow as tf
import cv2
import processing
import argparse
from model import build_model
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers import ELU
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF

MINI_BATCH_SIZE = 16
EPOCHS = 80
STEPS_PER_EPOCH = 400
VERSION = 1

SIZE = (100, 100)


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
    
    generator = ImageDataGenerator(rescale=1./40.,validation_split=args.split)

    directory = FLOW_IMGS_TRAIN
    train_meta.flow_path = train_meta.flow_path.apply(lambda x: x.replace(directory + '/', ''))
    print(train_meta.head())

    train_generator = generator.flow_from_dataframe(
        dataframe=train_meta,
        directory=directory,
        has_ext=True,
        x_col="flow_path",
        y_col="speed",
        subset="training",
        batch_size=MINI_BATCH_SIZE,
        seed=42,
        shuffle=True,
        target_size=SIZE,
        class_mode="other",
        )
    validation_generator = generator.flow_from_dataframe(
        dataframe=train_meta,
        directory=directory,
        x_col="flow_path",
        y_col="speed",
        subset="validation",
        batch_size=MINI_BATCH_SIZE,
        seed=42,
        shuffle=True,
        target_size=SIZE,
        class_mode="other",
    )
    
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  patience=5,
                                  verbose=1,
                                  min_delta=0.23,
                                  mode='min',)

    modelCheckpoint = ModelCheckpoint(WEIGHTS_PATH,
                                      monitor='val_loss',
                                      save_best_only=False,
                                      mode='min',
                                      verbose=1,
                                      save_weights_only=True)

    tensorboard = TensorBoard(log_dir=TENSORBOARD, histogram_freq=0,
                              write_graph=True, write_images=True)

    callbacks_list = [modelCheckpoint, tensorboard]

    CHANNEL = 3
    WIDTH = SIZE[0]
    HEIGHT = SIZE[1]
    model = build_model(HEIGHT, WIDTH, CHANNEL)


    model.summary()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=200)


    pickle.dump(history.history, open(HISTORY_PATH, "wb"))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_flow_meta_file",
                        help="Train Flow Images meta file")
    parser.add_argument("--split", type=float, default=0,
                        help="Percentage of data used to validate the model")
    args = parser.parse_args()
    main(args)
