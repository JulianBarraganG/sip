import numpy as np 
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.fft import fft 
from const import OUTPUT_FOLDER, SAMPLES_FOLDER 
import soundfile as sf

def spectogram(signal: NDArray, sample_rate: int, window_size: int, readable: bool = True) -> plt:
    stride = window_size // 2 
    spectrogram = []
    for i in range(0, len(signal) - window_size, stride):
        windowed_signal = signal[i:i+window_size] 
        spectrum = np.abs(fft(windowed_signal))
        spectrogram.append(spectrum[:window_size//2])  

    bin_width = sample_rate / window_size
    duration = len(signal) / sample_rate
    max_freq = sample_rate / 2  

    plt.figure(figsize=(10, 6))
    if readable:
        plt.imshow(
            10 * np.log10(np.array(spectrogram)).T,
            aspect='auto',
            origin='lower',
            extent=[0, duration, -bin_width/2, max_freq - bin_width/2] , vmin= -5, vmax = 5 # <-- maps axes to seconds and Hz
        )
    else: 
        plt.imshow(
        10 * np.log10(np.array(spectrogram)).T,
        aspect='auto',
        origin='lower',
        extent=[0, duration, -bin_width/2, max_freq - bin_width/2]  # <-- maps axes to seconds and Hz
    )
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.yticks(np.arange(0, 1000, 50))  # tick every 50 Hz
    plt.ylim(0, 1000)  # now this correctly means 0 to 1000 Hz
    return plt


if __name__ == "__main__":
    output = OUTPUT_FOLDER / "task1"
    output.mkdir(exist_ok=True, parents=True)
    progression_sound = sf.read(SAMPLES_FOLDER / "progression.wav")
    sample_rate = progression_sound[1]
    spec = spectogram(progression_sound[0], sample_rate= sample_rate, window_size=1024, readable=True)
    plt.savefig(output / "spectrogram_readable_W_1024.png")
    spec = spectogram(progression_sound[0], sample_rate= sample_rate, window_size=512, readable=True)
    plt.savefig(output / "spectrogram_readable_W_512.png")
    spec = spectogram(progression_sound[0], sample_rate= sample_rate, window_size=2048, readable=True)
    plt.savefig(output / "spectrogram_readable_W_2048.png")

    spec = spectogram(progression_sound[0], sample_rate= sample_rate, window_size=1024, readable=False)
    plt.savefig(output / "spectrogram_unreadable_W_1024.png")