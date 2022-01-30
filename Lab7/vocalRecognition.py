import os
from os.path import dirname, join as pjoin
import scipy.io.wavfile as wav
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def chunking(windows, freqs, fftValue):
    to_return = []
    for n in range(1, windows + 1):
        to_value = []
        for i, el in enumerate(freqs):
            if(el > (200 + ((n-1)*50)) and el < (200+n*50)):
                to_value.append(abs(fftValue[i]))

            if(el > 200 + n*50):
                break

        to_mean = np.array(to_value).mean()
        to_std = np.array(to_value).std()
        to_return.append(to_mean)
        to_return.append(to_std)

    return to_return


data_dir = pjoin('free-spoken-digit', 'dev')
traces = np.array(os.listdir(data_dir))

lenght = []
value = []
data1 = []
index = []
for i in range(len(traces)):
    wav_fname = pjoin(data_dir, traces[i])
    samplerate, data = wav.read(wav_fname)
    lenght.append(data.shape[0])
    rough_ids = traces[i].split(".")[0]
    label = rough_ids.split("_")
    data1.append(data)
    index.append(label[0])
    value.append(label[1])


data_dir = pjoin('free-spoken-digit', 'eval')
traces = np.array(os.listdir(data_dir))

data_eval = []
index_eval = []


for i in range(len(traces)):
    wav_fname = pjoin(data_dir, traces[i])
    samplerate, data = wav.read(wav_fname)
    lenght.append(data.shape[0])
    rough_ids = traces[i].split(".")[0]
    data_eval.append(data)
    index_eval.append(rough_ids)


audios = pd.DataFrame()
audios['data'] = data1
audios['id'] = index
audios['class'] = value

audios_eval = pd.DataFrame()
audios_eval['data'] = data_eval
audios_eval['id'] = index_eval


print(type(index))
audios.index = audios['id']
audios.index = audios.index.astype('int32')
audios = audios.sort_index()
print(type(audios['data'].loc[2]))
print(audios['data'].loc[2])

audios_eval.index = audios_eval['id']
audios_eval.index = audios_eval.index.astype('int32')
audios_eval = audios_eval.sort_index()


print(audios.sort_index())

'''
first_audio = audios['data'].iloc[0]
N = len(audios['data'].iloc[0])
T = 1.0/8000
secs = N/float(8000)
t = np.arange(0, secs, T)
print(t)
freqs = fftfreq(len(first_audio), t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(int(N/2))]
fft_freqs_side = np.array(freqs_side)

w = signal.windows.blackman(N)
FFT = fft(first_audio)
FFT_window = fft(first_audio*w)
FFT_side = FFT[range(int(N/2))]
FFT_side_window = FFT_window[range(int(N/2))]
'''


window = 50
trasformata = pd.DataFrame(np.zeros((len(audios['id']), window)))

features = []
for i, el in enumerate(audios['data']):
    N = len(el)
    T = 1.0/8000
    secs = N/float(8000)
    t = np.arange(0, secs, T)
    freqs = fftfreq(len(el), t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(int(N/2))]

    fft_freqs_side = np.array(freqs_side) #frequenze
    w = signal.windows.blackman(N)
    FFT_window = fft(el*w)
    FFT_side_window = FFT_window[range(int(N/2))] #valori fft
    features.append(chunking(60, fft_freqs_side, FFT_side_window))

features_eval = []
for i, el in enumerate(audios_eval['data']):
    N = len(el)
    T = 1.0/8000
    secs = N/float(8000)
    t = np.arange(0, secs, T)
    freqs = fftfreq(len(el), t[1]-t[0])
    fft_freqs = np.array(freqs)
    freqs_side = freqs[range(int(N/2))]

    fft_freqs_side = np.array(freqs_side) #frequenze
    w = signal.windows.blackman(N)
    FFT_window = fft(el*w)
    FFT_side_window = FFT_window[range(int(N/2))] #valori fft
    features_eval.append(chunking(60, fft_freqs_side, FFT_side_window))



y = np.array(audios['class'])
x = np.array(features)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

forest = RandomForestClassifier(n_estimators=120,  max_features='sqrt')
forest.fit(x_train, y_train)
y_predict_forest = forest.predict(x_test)

x_eval = np.array(features_eval)
y_eval = forest.predict(x_eval)

score = f1_score(y_test, y_predict_forest, average='macro')

print("accuracy ", score)



df = pd.DataFrame({"Id": audios_eval['id'], "Predicted": y_eval})
df.index = df["Id"]
df.index = df.index.astype("int32")
df = df.sort_index()

df.to_csv("mypredictions.csv", sep=",", index=False)






