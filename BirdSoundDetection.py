
print("Let's start!\n")
print("Loading...")

import glob 
import random
print("...")


import tensorflow as tf
import pandas as pd
print("...")

import librosa
import librosa.display
print("...")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import time
print("...")
import numpy as np
from sklearn import metrics 
print("...")
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import LabelEncoder

print("Libraries are imported successfully")

# Getting the dataset
datafolder = "/content/drive/My Drive/Datasets/ff1010bird/"
loc = datafolder+"wav/"
metadata = pd.read_csv("/content/drive/My Drive/Datasets/ff1010bird/ff1010bird_metadata_2018.csv")
m,n = metadata.shape
P = 0.05
idx = np.random.permutation(m)
metadataTrain = metadata.iloc[idx[:round(P*m)]] 
metadataValidation = metadata.iloc[idx[round(P*m):round(m/10)]]
trainFiles = [loc+str(s)+".wav" for s in metadataTrain["itemid"]]
#print(trainFiles)
trainLabels = [str(s) for s in metadataTrain["hasbird"]]
#print(trainLabels)
validationFiles = [loc+str(s)+".wav" for s in metadataValidation["itemid"]]
#print(validationFiles)
validationLabels = [str(s) for s in metadataValidation["hasbird"]]
#print(validationLabels)
trainFiles = np.vstack((trainFiles, trainLabels)).T
#print(trainFiles)
validationFiles = np.vstack((validationFiles,validationLabels)).T
#print(validationFiles),

# Function for extracting features 
def extract_features(dir,
                     bands=40,frames=41):
  def _windows(data, window_size):
    start = 0
    while start < len(data):
      yield int(start), int(start + window_size)
      start += (window_size // 2)

          
  window_size = 512 * (frames - 1)
  features, labels = [], []
  for fn in dir:
    segment_log_specgrams, segment_labels = [], []
    sound_clip,sr = librosa.load(fn[0])
    label = fn[1]
    for (start,end) in _windows(sound_clip,window_size):
      if(len(sound_clip[start:end]) == window_size):
        signal = sound_clip[start:end]
        melspec = librosa.feature.melspectrogram(signal,n_mels=bands)
        logspec = librosa.amplitude_to_db(melspec)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        segment_log_specgrams.append(logspec)
        segment_labels.append(label)
            
    segment_log_specgrams = np.asarray(segment_log_specgrams).reshape(
                len(segment_log_specgrams),bands,frames,1)
    segment_features = np.concatenate((segment_log_specgrams, np.zeros(
                np.shape(segment_log_specgrams))), axis=3)
    for i in range(len(segment_features)): 
      segment_features[i, :, :, 1] = librosa.feature.delta(
                    segment_features[i, :, :, 0]) 
    if len(segment_features) > 0: # check for empty segments 
      features.append(segment_features)
      labels.append(segment_labels)
  return features, labels
print("What the heck")
# Loading and extracting features from sound files
sc, sr = librosa.load("/content/drive/My Drive/Datasets/ff1010bird/wav/191742.wav")
#ti = TicToc() # create TicToc instance
#ti.tic() # Start timer
#print(trainFiles[0][1])
trainfeatureVectors, trainVectorLabels=extract_features(trainFiles)
ValidationfeatureVectors, ValidationVectorLabels=extract_features(validationFiles)
#ti.toc() # Print elapsed time

# Reshaping (flattening) the extracted features to make them suitable for inserting the model
trainVectorLabels = np.array(trainVectorLabels).flatten()
trainVectorLabels = trainVectorLabels.reshape(len(trainVectorLabels),1)
ValidationVectorLabels = np.array(ValidationVectorLabels).flatten()
ValidationVectorLabels = ValidationVectorLabels.reshape(len(ValidationVectorLabels),1)

trainfeatureVectors_flat = np.array(trainfeatureVectors).flatten()
trainfeatureVectors_flat = trainfeatureVectors_flat.reshape(int(trainfeatureVectors_flat.shape[0]/len(trainVectorLabels)),len(trainVectorLabels))

# Normalizing the vector values
norm_train = np.linalg.norm(trainfeatureVectors_flat)
trainfeatureVectors_flat = trainfeatureVectors_flat/norm_train #########

print("Dim of feature vectors of training set: " + trainfeatureVectors_flat.shape)
#trainVectorLabels_flat = trainfeatureVectors.flatten()
ValidationfeatureVectors_flat = np.array(ValidationfeatureVectors).flatten()
ValidationfeatureVectors_flat = ValidationfeatureVectors_flat.reshape(int(ValidationfeatureVectors_flat.shape[0]/len(ValidationVectorLabels)),len(ValidationVectorLabels))

# Normalizing the vector values
norm_validation = np.linalg.norm(ValidationfeatureVectors_flat)
ValidationfeatureVectors_flat = ValidationfeatureVectors_flat/norm_validation #########

print("Dim of feature vectors of validation set: " + ValidationfeatureVectors_flat.shape)

# Constructing the model 
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(trainfeatureVectors_flat.shape[0])),
        # tf.keras.layers.experimental.RandomFourierFeatures(
        #     output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        # ),
        tf.keras.layers.Dense(512, activation= "relu"),
        tf.keras.layers.Dense(256, activation = "linear"),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(256, activation = "relu"),
        tf.keras.layers.Dense(64, activation = "linear"),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(64, activation = "relu"),
        tf.keras.layers.Dense(32, activation = "relu"),
        tf.keras.layers.Dense(16, activation = "relu"),
        tf.keras.layers.Dense(1, activation = "sigmoid"),
    ]
)

# Compiling the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

# Fitting the training data into the model
history = model.fit(trainfeatureVectors_flat.T.astype(np.float),trainVectorLabels.astype(np.float),batch_size = 512,epochs=50,verbose=0,shuffle=True)

model.summary()

#Evaluating and making predictoin with validation set
valid_loss, valid_acc = model.evaluate(ValidationfeatureVectors_flat.T.astype(np.float), ValidationVectorLabels.astype(np.float))
predictions = model.predict(ValidationfeatureVectors_flat.T.astype(np.float))

# predictions = model.predict(ValidationfeatureVectors_flat.T.astype(np.float))

print("Validation loss: " + valid_loss)
print("Validation accuracy: " + valid_acc)

print(predictions[0])

# Binarization of prediction data
predictions = (predictions>=0.5)

print(predictions[0])