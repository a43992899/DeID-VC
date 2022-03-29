import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
from hparams import DATA_PATH, AUTOVC_ROOT, PROJECT_ROOT
from tqdm import tqdm

def str2num(string):
    if len(string) == 3: # for wsj
        ascii = np.fromstring(string, dtype=np.uint8)
        return np.sum(ascii * np.array([36**2, 36, 1]))
    elif len(string) == 4 or string=="s5": # for vctk
        return int(string[1:])
    elif len(string) == 7 and string.startswith("id"): # for voxceleb
        return int(string[2:])
    else:
        raise ValueError("Invalid speaker id format: {}".format(string))

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
    
mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = DATA_PATH+'./wavs'
# spectrogram directory
targetDir = DATA_PATH+'./spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in tqdm(sorted(subdirList), position=1):
    # print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    prng = RandomState(str2num(subdir)) # support wsj
    for fileName in tqdm(sorted(fileList), position=0):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)    
        # save spect    
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                S.astype(np.float32), allow_pickle=False)    
        
