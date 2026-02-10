from scipy.signal import convolve
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from const import DATA_FOLDER, OUTPUT_FOLDER

soundbyte, soundbyte_samplerate = sf.read(DATA_FOLDER / "laugh2.wav")

left = soundbyte[:, 0]
right = soundbyte[:, 1]

# Plot the soundbyte on the same plot
alpha = .70
lim = .35
plt.plot(left, label="Left Channel", alpha=alpha)
plt.plot(right, label="Right Channel", alpha=alpha)
plt.title("Soundbyte")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.ylim(-lim, lim)
plt.legend()
plt.show()
