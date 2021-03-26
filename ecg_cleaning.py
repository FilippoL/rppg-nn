import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.signal import butter, iirnotch, lfilter


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def notch_filter(cutoff, q):
    nyq = 0.5 * fs
    freq = cutoff / nyq
    b, a = iirnotch(freq, q)
    return b, a


def highpass(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    return x


def lowpass(data, fs, order=5):
    b, a = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch(data, powerline, q):
    b, a = notch_filter(powerline, q)
    z = lfilter(b, a, data)
    return z


def final_filter(data, fs, order=5):
    b, a = butter_highpass(cutoff_high, fs, order=order)
    x = lfilter(b, a, data)
    d, c = butter_lowpass(cutoff_low, fs, order=order)
    y = lfilter(d, c, x)
    f, e = notch_filter(powerline, 30)
    z = lfilter(f, e, y)
    return z

f = h5py.File('./data_in/data.hdf5', 'r')
ecg_signal = list(f['pulse'])
fs = 1000

cutoff_high = 0.5
cutoff_low = 2
powerline = 60
order = 5

plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(ecg_signal)
ax1.set_title("Raw ECG signal")

filter_signal = final_filter(ecg_signal, fs, order)
ax2 = plt.subplot(212)
plt.plot(filter_signal)
ax2.set_title("Clean ECG signal")
plt.show()
