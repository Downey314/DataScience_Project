#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 23:26:25 2021

@author: downey
"""

from comet_ml import Experiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
#Iterating through every audio file and extract features
from tqdm import tqdm
#import Ipython.display as ipd
import librosa
import librosa.display


import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime



# Load dataset
my_data = pd.read_csv('/Users/downey/Desktop/data/Voice_Problem.csv')
# Create a list of the class labels
labels = list(my_data['Label'].unique())
print(labels)


print(my_data[my_data['Label'] == labels[0]][:1].reset_index())
print(my_data[my_data['Label'] == labels[1]][:1].reset_index())

# Let's grab a single audio file from each class
files = dict()

for i in range(len(labels)):
    #Select first file that match the label and set the index to 0 
    tmp = my_data[my_data['Label'] == labels[i]][:1].reset_index()
    
    path = "/Users/downey/Desktop/data/{}/{}".format(tmp['Label'][0], tmp['File_Name'][0])

    #path = 'UrbanSound8K/audio/fold{}/{}'.format(tmp['Label'][0], tmp['File_Name'][0]))
    files[labels[i]] = path
    print(files)

#Look at waveforms foe each sample
fig = plt.figure(figsize = (15,15))# Log graphic of waveforms to Comet

fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

for i, label in enumerate(labels):
    fn = files[label]
    fig.add_subplot(5, 2, i+1)
    plt.title(label)
    data, sample_rate = librosa.load(fn)
    librosa.display.waveplot(data, sr= sample_rate)
    
#plt.savefig('class_examples.png')
# Log graphic of waveforms to Comet
#experiment.log_image('class_examples.png')


dataset_path = "/Users/downey/Desktop/data"
filename = "/Users/downey/Dropbox/Voice Acoustic Data/Sylvia/Control Teachers/S4_D1_post ahh 1.wav"

plt.figure(figsize = (14, 5))

data, sample_rate = librosa.load(filename)
#len(data) - how many sample points in a signal
print((len(data), sample_rate)) #Sample rate is 22050 - How many times per seconds a sound is sampled 

#Duration of 1 sample
sample_duration = 1 / sample_rate
#Duration of the audio signal in seconds
total_time = sample_duration * data.size
print(f"the total time of the audio is about {total_time: .3f} seconds long")

librosa.display.waveplot(data, sr = sample_rate)

#The MFCCs summaries the frequency distribution across the window size. 
mfccs = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
print(f"The shape for MFCCs is {mfccs.shape}")
print(mfccs.shape)

#Extracting MFCCs for every audio files 
audio_path = "/Users/downey/Dropbox/Voice Acoustic Data/"
metadata = pd.read_csv("/Users/downey/Dropbox/Voice Acoustic Data/Sylvia/Voice_Problem.csv")
print(metadata.head(10))
#Check whether the data is imbalanced 
print(metadata["Label"].value_counts())


def feature_extractor_mfccs(file):
    data, sample_rate = librosa.load(filename)
    mfccs = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
    scaled_mfccs = np.mean(mfccs.T, axis = 0)
    return scaled_mfccs

extracted_features = []



data, sample_rate = librosa.load("/Users/downey/Dropbox/Voice Acoustic Data/Sylvia/Control Teachers/S4_D1_post ahh 1.wav")
mfccs = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
print(mfccs.shape)
print(mfccs)
scaled_mfccs = np.mean(mfccs.T, axis = 0)
print(scaled_mfccs.shape)
print(scaled_mfccs)

for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
    #print("the diepath is ", dirpath)
    #print(dirnames)
    #print(filenames)
    if dirpath is not dataset_path:
        print(len(filenames))
        for f in filenames:
            if(not(f == ".DS_Store")):
                #Load audio files
                file_path = os.path.join(dirpath, f)
                print(file_path)
                signal, sr = librosa.load(file_path, sr = sample_rate)
                semantic_label = dirpath.split("/")[-1]
                
                mfccs = librosa.feature.mfcc(y = signal, sr = sample_rate, n_mfcc = 40)
                scaled_mfccs = np.mean(mfccs.T, axis = 0)
                
                extracted_features.append([scaled_mfccs, semantic_label])
                
print(len(extracted_features))


extracted_features_df = pd.DataFrame(extracted_features, columns = ["Feature", "Labels"])
print(extracted_features_df)

#extracted_features_df['Labels'] = extracted_features_df['Labels'].replace(['Voice Problem Teachers copy'], 1)
#extracted_features_df['Labels'] = extracted_features_df['Labels'].replace(['Control Teachers copy'], 0)
print(extracted_features_df)
print(extracted_features_df.dtypes) #Data Types

#Split the dataset into independent and dependent dataset
X = np.array(extracted_features_df["Feature"].tolist())
Y = np.array(extracted_features_df["Labels"].tolist())
#Label Encoding 
#y = np.array(pd.get_dummies(Y))

#One-Hot Encoding - Converting columns to binary categorical datatype
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(Y))
print(f"The X feature shape is {X.shape}")
print(y)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print( y_train.shape)
print(y_test.shape)

#Model Creation
print(tf.__version__)

#Number of Labels
num_labels = y.shape[1]
print("Number of Labels: ", num_labels) #2

model = Sequential()

#First Layer
model.add(Dense(100, input_shape = (40, )))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#Second Layer
model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.5))
#Third Layer
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.5))

#Final Layer [Outpout layer]
model.add(Dense(num_labels))
model.add(Activation("sigmoid")) #Multi-class classification problem

model.summary()

model.compile(loss = 'binary_crossentropy', metrics=['accuracy'], optimizer = 'adam')

#Training the model
num_epochs = 100
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath = "weights.best.hdf5", verbose = 1, save_best_only = True)

start = datetime.now()

model.fit(X_train, y_train, batch_size = num_batch_size, 
                            epochs = num_epochs, 
                            validation_data=(X_test, y_test), 
                            callbacks=[checkpointer], 
                            verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)



# Evaluating the model on the training and testing set
score = model.evaluate(X_train, y_train, verbose = 0)
print("Training Accuracy: {0:.2%}".format(score[1]))

test_accuracy = model.evaluate(X_test, y_test, verbose = 0)
print("Testing Accuracy: {0:.2%}".format(score[1]))


#Testing some Audio data
filename_C = "/Users/downey/Desktop/data/Control_Teachers/S25_D1_POST Ahh1 .wav"
filename_P = "/Users/downey/Desktop/data/Voice_Problem_Teachers/S3_D1_ah1.wav"
audio, sample_rate = librosa.load(filename_P, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis = 0)

print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)

mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)

predict_x=model.predict(mfccs_scaled_features) 
classes_x=np.argmax(predict_x,axis=1)

prediction_class = labelencoder.inverse_transform(classes_x) 
print(prediction_class)







                






