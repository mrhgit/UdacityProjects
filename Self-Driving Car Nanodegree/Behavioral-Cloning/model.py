
# coding: utf-8

# In[1]:


# Imports

import keras # For the neural network
import csv # For the input data
import cv2 # For loading images

import random
#from skimage import transform, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.max_open_warning': 0})

linux_sim_dir = '/home/mach1uvnc/carnd/CarND-Behavioral-Cloning-P3-master/linux_sim/'
recordings_dir = linux_sim_dir+'recordings/'
all_recordings = ['track_1','track_1b','track_2','track_2b']


# In[2]:


# CSV Format:
# Center Image, Left Image, Right Image, Steering (-1,1), Throttle (0,1), Brake (0), Speed (0,30)

def load_csv(recording):
    lines = []
    with open(recordings_dir+recording+"/driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines

def load_image(recording,filename):
    source_file = filename.split("/")[-1] # get just the filename
    source_path = recordings_dir+recording+"/IMG/"+source_file # append it to new directory
    imageBGR = cv2.imread(source_path) # read in the image
    imageRGB = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB) # Convert to match drive.py - RGB
    return imageRGB

def load_data(recording,csvlines,addLeft=True,addRight=True,corr_factor=0.4,addLRFlip=True):
    images = []
    steerings = []
    
    for line in csvlines:
        steering = float(line[3])
        # Center Camera
        steerings.append(steering)
        images.append(load_image(recording,line[0]))
        
        if addLeft:
            # Left Camera
            steerings.append(steering + corr_factor)
            images.append(load_image(recording,line[1]))
            
        if addRight:
            # Right Camera
            steerings.append(steering - corr_factor)
            images.append(load_image(recording,line[2]))
    
    if addLRFlip:
        pass
    
    return images, steerings


# In[3]:


# Load and Preview Data
rec0 = all_recordings[0]
rec0_csv = load_csv(rec0)
X_data, y_data = load_data(rec0,rec0_csv)

rec1 = all_recordings[1]
rec1_csv = load_csv(rec1)
rec1_images, rec1_meas = load_data(rec1, rec1_csv)
X_data += rec1_images
y_data += rec1_meas
del rec1_images, rec1_meas

print (len(rec0_csv),len(rec1_csv),len(X_data),len(y_data))


# In[4]:


for img_idx in random.sample(list(range(len(X_data))),10):
    plt.figure()
    plt.imshow(X_data[img_idx])
    plt.axis('off')
    plt.title('Index: %d  Steering: %.3f' % (img_idx,y_data[img_idx]))


# In[5]:


X_data = np.asarray(X_data)
y_data = np.asarray(y_data)


# In[6]:


# Add mirror-image version to better generalize

orig_len = len(X_data)
X_data_mirrored = [X_data[i][:,::-1,:] for i in range(len(X_data))]
y_data_mirrored = np.copy(y_data)*-1
print(X_data.shape,len(X_data_mirrored))
X_data = np.asarray(list(X_data)+X_data_mirrored)
y_data = np.asarray(list(y_data)+list(y_data_mirrored))
print(X_data.shape)
plt.figure()
plt.imshow(X_data[0])
plt.figure()
plt.imshow(X_data[orig_len])
plt.figure()
plt.plot(y_data)


# In[7]:


# Build the Neural Network using Keras
from keras.models import Sequential, Model
from keras.layers import Input,Flatten, Dense, Activation, Lambda, MaxPooling2D, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

model = Sequential() # Start a Keras model
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=X_data[0].shape)) # Clip to above hood but below horizon
model.add(Lambda(lambda x: (x / 255.0) - 0.5)) # Normalize
#model.add(BatchNormalization(input_shape=X_data[0].shape)) # Alternative way of normalizing

# Triple Layer of convolution
model.add(Convolution2D(8,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(8,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(8,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())

# Dropout
model.add(Dropout(0.5))

# Triple Layer of Fully Connected (aka Dense)
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))

model.add(Dense(1)) # This is our regression output - Steering Angle


# In[8]:


# Build Training Code
model.compile(loss='mse', optimizer='adam')
model.summary()


# In[9]:


# Train and Validate
cb_save_best = keras.callbacks.ModelCheckpoint("model.h5",verbose=1,save_best_only=True)
hist_object = model.fit(X_data,y_data, validation_split=0.2, shuffle=True, nb_epoch=3, callbacks=[cb_save_best])

