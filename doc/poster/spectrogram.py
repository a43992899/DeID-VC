#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import wave, sys
  
# shows the sound waves
def visualize(path: str, title: str):
	# read audio file
	raw = wave.open(path)
	signal = raw.readframes(-1)
	signal = np.frombuffer(signal, dtype ="int16")
	f_rate = raw.getframerate()

	plt.title(title)
	plt.xlabel("Time (seconds)")
	plt.ylabel("Frequency (Hz)")
	plt.specgram(signal, Fs=f_rate, NFFT=1024, noverlap=512)

	plt.savefig(path + ".pdf")

if __name__ == "__main__":
	mpl.rcParams['figure.figsize'] = 12, 4
	path = sys.argv[1]
	title = sys.argv[2]
	visualize(path, title)
