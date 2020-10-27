
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

import json
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
metadataFiles = [loc+str(s)+".wav" for s in metadata["itemid"]]
#print(metadataFiles)
metadataLabels = [str(s) for s in metadata["hasbird"]]
#print(metadataLabels)
metadataFiles = np.vstack((metadataFiles, metadataLabels)).T
#print(metadataFiles)

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
# Extracting features from sound files and save them to file
"""for i in range (int (len(metadataFiles)/100)+1):
  if (len(metadataFiles)%100)==0 and i==(len(metadataFiles)/100):
    break
  if i==(len(metadataFiles)/100):
    featureVectors, VectorLabels=extract_features(metadataFiles[:][i*100:len(metadataFiles)])
  else:
    featureVectors, VectorLabels=extract_features(metadataFiles[:][i*100:(i+1)*100])
  featureVectors= np.array(featureVectors).tolist()
  VectorLabels= np.array(VectorLabels).tolist()
  with open('/content/drive/My Drive/Datasets/ff1010bird/mfcc_mfccdelta/'+'features_'+str(i)+'.json', 'w') as f:
    json.dump(featureVectors, f)
  with open('/content/drive/My Drive/Datasets/ff1010bird/mfcc_mfccdelta/'+'labels_'+str(i)+'.json', 'w') as f:
    json.dump(VectorLabels, f)
  print(str(i))"""

#Load features and labels from trainFiles
for i in range (int (len(metadataFiles)/100)+1):
  if (len(metadataFiles)%100)==0 and i==(len(metadataFiles)/100):
    break
  with open('/content/drive/My Drive/Datasets/ff1010bird/mfcc_mfccdelta/'+'features_'+str(i)+'.json','r') as fs:
    featureVectorstmp = json.loads(fs.read())
  with open('/content/drive/My Drive/Datasets/ff1010bird/mfcc_mfccdelta/'+'labels_'+str(i)+'.json','r') as f:
    VectorLabelstmp = json.loads(f.read())
  if i==0:
    featureVectors=featureVectorstmp
    VectorLabels=VectorLabelstmp
  else:
    featureVectors=np.concatenate((featureVectors,featureVectorstmp))
    VectorLabels=np.concatenate((VectorLabels,VectorLabelstmp))
  fs.close()
  f.close()
print("Features and labels loaded")

#Determine train and dev sets from the dataset
print(len(featureVectors))
m = len(VectorLabels)
P = 0.9
idx = np.random.permutation(m)
trainfeatureVectors=featureVectors[idx[:round(P*m)]]
trainVectorLabels=VectorLabels[idx[:round(P*m)]]
ValidationfeatureVectors=featureVectors[idx[round(P*m):round(m)]]
ValidationVectorLabels=VectorLabels[idx[round(P*m):round(m)]]
del featureVectors, VectorLabels

# Reshaping (flattening) the extracted features to make them suitable for inserting the model
trainVectorLabels = np.array(trainVectorLabels).flatten()
trainVectorLabels = trainVectorLabels.reshape(len(trainVectorLabels),1)
ValidationVectorLabels = np.array(ValidationVectorLabels).flatten()
ValidationVectorLabels = ValidationVectorLabels.reshape(len(ValidationVectorLabels),1)

trainfeatureVectors = np.array(trainfeatureVectors).flatten()
trainfeatureVectors = trainfeatureVectors.reshape(int(trainfeatureVectors.shape[0]/len(trainVectorLabels)),len(trainVectorLabels))

# Normalizing the vector values
trainfeatureVectors = trainfeatureVectors/np.linalg.norm(trainfeatureVectors) #########

print("Dim of feature vectors of training set: " + trainfeatureVectors.shape)

ValidationfeatureVectors = np.array(ValidationfeatureVectors).flatten()
ValidationfeatureVectors = ValidationfeatureVectors.reshape(int(ValidationfeatureVectors.shape[0]/len(ValidationVectorLabels)),len(ValidationVectorLabels))

# Normalizing the vector values
ValidationfeatureVectors = ValidationfeatureVectors/np.linalg.norm(ValidationfeatureVectors) #########

print("Dim of feature vectors of validation set: " + ValidationfeatureVectors.shape)

# Constructing the model
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(trainfeatureVectors.shape[0])),
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
history = model.fit(trainfeatureVectors.T.astype(np.float),trainVectorLabels.astype(np.float),batch_size = 512,epochs=50,verbose=0,shuffle=True)

model.summary()

#Evaluating and making predictoin with validation set
valid_loss, valid_acc = model.evaluate(ValidationfeatureVectors.T.astype(np.float), ValidationVectorLabels.astype(np.float))
predictions = model.predict(ValidationfeatureVectors.T.astype(np.float))

# predictions = model.predict(ValidationfeatureVectors_flat.T.astype(np.float))

print("Validation loss: " + valid_loss)
print("Validation accuracy: " + valid_acc)

print(predictions[0])

# Binarization of prediction data
predictions = (predictions>=0.5)

print(predictions[0])
