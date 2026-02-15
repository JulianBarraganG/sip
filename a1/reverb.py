from scipy.signal import convolve
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from const import SAMPLES_FOLDER, CLAPS_FOLDER, SPLASHES_FOLDER, OUTPUT_FOLDER
import os

save_folder = os.path.join(OUTPUT_FOLDER, "reverb")


if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Wrote a plotting function for simplicity 
def plot_soundbyte(soundbytes: list[np.ndarray], labels: list[str], title: str, xlim: tuple[int, int] | None = None, alpha: float = .60, ylim: float | None = None) -> None:
    plt.figure(figsize=(12, 6))
    for soundbyte, label in zip(soundbytes, labels):
        plt.plot(soundbyte, label=label, alpha=alpha)
    
    plt.title(title)
    #Set size of plot 
    if xlim is not None:
        plt.xlim(xlim)    
    if ylim is not None:
        plt.ylim(-ylim, ylim)

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(save_folder, title.replace(" ", "_") + ".png"))

#TASK 1

#Just load the soundfile, split the channels, and plot it
laugh2_filename = os.path.join(SAMPLES_FOLDER, "laugh2.wav")
laugh_sound, laugh_samplerate = sf.read(laugh2_filename)
print("laugh samplerate: ", laugh_samplerate)

left = laugh_sound[:, 0]
right = laugh_sound[:, 1]

plot_soundbyte([left, right], ["Left Channel", "Right Channel"], "Left and Right Channels of Laugh Sound", xlim=(75000, 78000))





# TASK 2 

#load impulses
clap_filename = os.path.join(CLAPS_FOLDER, "Narrow-Passage-Between-Two-Houses--Hand-Clap-Sample--(DPA4060-omni).wav")
splash_filename = os.path.join(SPLASHES_FOLDER, "Splash 10.wav")

clap, clap_samplerate = sf.read(clap_filename)
splash, splash_samplerate = sf.read(splash_filename)

#Convolve each channel seperately, otherwise we convolve across channels and it becomes a mess
convolved_clap_left = convolve(laugh_sound[:, 0], clap[:,0])
convolved_clap_right = convolve(laugh_sound[:, 1], clap[:,1])
convolved_splash_left = convolve(laugh_sound[:, 0], splash[:,0])
convolved_splash_right = convolve(laugh_sound[:, 1], splash[:,1])
                                
#Just plot one channel for readability
plot_soundbyte([laugh_sound[:, 0], convolved_clap_left, convolved_splash_left], ["Original Laugh Left", "Convolved with Clap Left", "Convolved with Splash Left"], "Laugh Sound Convolved with Impulses (Left Channel)", xlim=(75000, 78000))

#stitch the channels back together and write to file
convolved_clap = np.stack((convolved_clap_left, convolved_clap_right), axis=1)
convolved_splash = np.stack((convolved_splash_left, convolved_splash_right), axis=1)

sf.write(os.path.join(save_folder, "laugh_convolved_clap.wav"), convolved_clap, laugh_samplerate)
sf.write(os.path.join(save_folder, "laugh_convolved_splash.wav"), convolved_splash, laugh_samplerate)


#Task 3 - sum normalization 

#I choose to normalize by setting the sum of the impulse to 1
def normalize_sum(impulse: np.ndarray) -> np.ndarray:
    sum = np.sum(np.abs(impulse))
    if sum > 0:
        normalized_impulse = impulse / sum
    else:
        normalized_impulse = impulse

    return normalized_impulse



sum_normalized_clap = normalize_sum(clap)
sum_normalized_splash = normalize_sum(splash)

norm_conv_clap_left = convolve(laugh_sound[:, 0], sum_normalized_clap[:,0])
norm_conv_splash_left = convolve(laugh_sound[:, 0], sum_normalized_splash[:,0])

plot_soundbyte([laugh_sound[:, 0], norm_conv_clap_left, norm_conv_splash_left], ["Original Laugh Left", "Convolved with Sum Normalized Clap Left", "Convolved with Sum Normalized Splash Left"], "Laugh Sound Convolved with Sum Normalized Impulses (Left Channel)", xlim=(75000, 78000))

