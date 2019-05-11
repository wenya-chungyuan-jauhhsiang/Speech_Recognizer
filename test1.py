import librosa
import numpy as np
import cv2

# def get_spectrogram(fname):
#     y, sr = librosa.load(fname)
#     return librosa.feature.melspectrogram(y=y, sr=sr)



y, sr = librosa.load('spoken_numbers_pcm/0_Agnes_100.wav')

# TODO 不同音檔產生出的spectrogram size不同（即圖片大小不同，要用reshape、interpolation還是其他種處理方式？）
# TODO 但如果後續要做log、CQT等轉換的話，也可以那之後再resize也無妨

# size = np.array([100,10])

# 除了用melspectrogram本身的arg以外，也可以吃librosa.filters.mel的kwarg
# n_mels提高應該可以提高細膩程度？？？還是用預設128就好？
n_mels = 128
# !!!melspectrogram裡面就有調用過stft了
spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
# 先利用opencv的resize來處理，都先resample到n_mels*60之類的，resize內row, col和一般認知反過來放 不知為啥
spectrogram = cv2.resize(spectrogram.astype('float'), (60, n_mels), interpolation = cv2.INTER_CUBIC)


#%%
import matplotlib.pyplot as plt
import librosa.display
# 可以用librosa.display.specshow搭配plt作圖
librosa.display.specshow(spectrogram)
plt.show()

# 原本的圖出來很雜，透過一些amplitude_to_db轉換，圖會更有特色
# https://librosa.github.io/librosa/generated/librosa.display.specshow.html
#%%
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
librosa.display.specshow(D, y_axis='linear')
# plt.imshow(D)
plt.show()

#%%
# CQT用得和mel不一樣，不是stft，有人說在一些情況下，CQT表現比STFT好？這個case的話就不能直接用melfrequency了
# 康乃爾的某篇論文
# https://arxiv.org/abs/1902.00631
# 另一篇IEEE
# https://ieeexplore.ieee.org/document/6701843
CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
librosa.display.specshow(CQT, y_axis='cqt_note')
# plt.imshow(CQT)
plt.show()