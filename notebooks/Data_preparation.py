
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import csv
import skvideo.io
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# paths
DATA_PATH = '/home/ubuntu/repositories/speed-challenge/data'
TRAIN_VIDEO = os.path.join(DATA_PATH, 'train.mp4')
TEST_VIDEO = os.path.join(DATA_PATH, 'test.mp4')
PREPARED_DATA_PATH = '/home/ubuntu/repositories/speed-challenge/prepared-data'
PREPARED_IMGS_TRAIN = os.path.join(PREPARED_DATA_PATH, 'train_imgs')
PREPARED_IMGS_TEST = os.path.join(PREPARED_DATA_PATH, 'test_imgs')

TRAIN_FRAMES = 20400
TEST_FRAMES = 10798


# In[3]:


#Init tqdm
from multiprocessing import Lock
tqdm.set_lock(Lock())  # manually set internal lock


# In[4]:


train_y = list(pd.read_csv(os.path.join(DATA_PATH, 'train.txt'), header=None, squeeze=True))


# In[5]:


assert(len(train_y)==TRAIN_FRAMES)


# In[8]:


def prepare_dataset(video_loc, img_folder, dataset_type):
    meta_dict = {}

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


# In[9]:


# train data
prepare_dataset(TRAIN_VIDEO, PREPARED_IMGS_TRAIN, 'train')


# In[10]:


# test data
prepare_dataset(TEST_VIDEO, PREPARED_IMGS_TEST, 'test')


# In[13]:


train_meta = pd.read_csv(os.path.join(PREPARED_DATA_PATH, 'train_meta.csv'))
assert(train_meta.shape[0] == TRAIN_FRAMES)
assert(train_meta.shape[1] == 3)


# In[14]:


train_meta.head()


# In[15]:


for i in range(5):
    print('speed:',train_meta['speed'][i] )
    img=mpimg.imread(train_meta['image_path'][i])
    print('shape:', img.shape)
    plt.imshow(img)
    plt.show()


# In[16]:


fig, ax = plt.subplots(figsize=(20,10))
plt.plot(train_meta['speed'])
plt.xlabel('image_index (or time since start)')
plt.ylabel('speed')
plt.title('Speed vs time')
plt.show()


# In[18]:


test_meta = pd.read_csv(os.path.join(PREPARED_DATA_PATH, 'test_meta.csv'))
assert(test_meta.shape[0] == TEST_FRAMES)
assert(test_meta.shape[1] == 3)


# In[19]:


test_meta.head()


# In[20]:


for i in range(5):
    print('speed:',test_meta['speed'][i] )
    img=mpimg.imread(test_meta['image_path'][i])
    print('shape:', img.shape)
    plt.imshow(img)
    plt.show()

