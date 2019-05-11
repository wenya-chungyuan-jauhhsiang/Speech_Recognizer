import librosa
import numpy as np
import cv2
import glob

#%% 把spoken_numbers_pcm內的.wav檔做CQT-spectrogram，並把矩陣存入cqt_npy (直接存矩陣不存圖，就沒有resolution的問題)
def store_cqt_npy(fname):
    y, sr = librosa.load('spoken_numbers_pcm/'+fname)
    n_bins=84
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins)), ref=np.max)
    CQT = cv2.resize(CQT.astype('float'), (60, n_bins), interpolation = cv2.INTER_CUBIC)
    np.save('cqt_npy/'+fname, CQT)

filename = glob.glob(r'spoken_numbers_pcm/*.wav')
for i in range(len(filename)):
    # mac
    filename[i] = filename[i].split('/')[1]
    # window
    # filename[i] = filename[i].split('\\')[1]
    store_cqt_npy(filename[i])


#%% 讀取npy，總共2400筆，2100做train、300做test
import random
from keras.utils import np_utils
x_list = []
y_list = []
filename_npy = glob.glob(r'cqt_npy/*.npy')
# 洗牌一下，到時只要切最後一塊出來當test就好，比較方便不用用抽樣的
random.shuffle(filename_npy)

for i in range(len(filename_npy)):
    x_list.append(np.load(filename_npy[i]))
    # mac
    y_list.append(filename_npy[i].split('/')[1][0])
    # window
    # y_list.append(filename_npy[i].split('\\')[1][0])

x_train = np.asarray(x_list[:2100])
y_train = np.asarray(y_list[:2100])

x_test = np.asarray(x_list[2100:])
y_test = np.asarray(y_list[2100:])

x_train = x_train.reshape(-1,84,60,1)
x_test = x_test.reshape(-1,84,60,1)

# normalized和1-hot
x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
x_test = (x_test - x_test.min()) / (x_test.max() - x_test.min())
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#%% 測試一下GPU是不是available
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.test.is_gpu_available()


#%% import package
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import optimizers

#%% model
model = Sequential()

input_shape = (84, 60, 1)

model.add(Conv2D(16, (7, 7), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer=optimizers.Adam(lr=0.0005),
              loss='kullback_leibler_divergence',
              metrics=['accuracy'])
model.summary()

#%% fit model
model_history = model.fit(x_train, y_train, batch_size=128, epochs=30,
                          validation_data = (x_test, y_test),
                          shuffle = True)

#%% 作圖
fig = plt.figure(figsize=(16,5))
ax1 = fig.add_subplot(1,2,1)
plt.plot(model_history.history["acc"])

plt.title("model training accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["model_final"], loc = "best")

ax2 = fig.add_subplot(1,2,2)
plt.plot(model_history.history["val_acc"])

plt.title("model validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["model_final"], loc = "best")

plt.show()

#%% score
score = model.evaluate(x_test, y_test, batch_size=10000)
print("Loss: %f" %score[0])
print("testing accuracy: %f" %(score[1]*100))

#%% 儲存model
model.save('CQT_CNN.h5')

#%% load_model
# Keras版本不同可能不能load，版本相近才有辦法
from keras.models import load_model
