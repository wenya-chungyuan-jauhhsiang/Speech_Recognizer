import librosa
import numpy as np
import cv2
import glob

def store_cqt_npy(fname):
    y, sr = librosa.load('spoken_numbers_pcm/'+fname)
    n_bins=84
    CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins)), ref=np.max)
    CQT = cv2.resize(CQT.astype('float'), (60, n_bins), interpolation = cv2.INTER_CUBIC)
    np.save('cqt_npy/'+fname, CQT)


filename = glob.glob(r'spoken_numbers_pcm/*ki_300.wav')
for i in range(len(filename)):
    filename[i] = filename[i].split('/')[1]
    store_cqt_npy(filename[i])

#%%
import random
x_list = []
y_list = []
filename_npy = glob.glob(r'cqt_npy/*.npy')
# 洗牌一下，到時只要切最後一塊出來當test就好，比較方便不用用抽樣的
random.shuffle(filename_npy)
for i in range(len(filename_npy)):
    x_list.append(np.load(filename_npy[i]))
    y_list.append(filename_npy[i].split('/')[1][0])

x_train = np.asarray(x_list[:6])
y_train = np.asarray(y_list[:6])

x_test = np.asarray(x_list[6:])
y_test = np.asarray(y_list[6:])

x_train = x_train.reshape(-1,84,60,1)
x_test = x_test.reshape(-1,84,60,1)