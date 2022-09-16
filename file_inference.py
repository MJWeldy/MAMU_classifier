# Requires python 10.0.0 use pyenv to install 
# pyenv install 10.0.0
# pyenv local 10.0.0
# usage using poetry to call python
# poetry install will install dependencies
# poetry run python main.py v_0 P:\PROJECTS\HJA_BirdNET_transfer\data\test_audio\
# This will process a folder of audio files and write images to the figures folder and csvs to data/clean

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import lite as tflite

from BirdNET import config as cfg
from BirdNET import analyze
from BirdNET import model
from BirdNET import embeddings

from classifier import utilities

#utility functions
def process_wav(wav, classifier_model):
    chunks = analyze.getRawAudioFromFile(wav)
    samples = []
    for c in range(len(chunks)):
        samples.append(chunks[c])
    data = np.array(samples, dtype='float32')
    e = model.embeddings(data) 
    scores = classifier_model(e)
    scores = scores.numpy()
    return scores

def display_results(wav, scores, class_map, out_name):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display

    y, sr = librosa.load(wav)
    file_length = len(y)/sr
    fig, ax = plt.subplots(nrows=3, ncols=1, constrained_layout=True)
    fig.suptitle(wav, fontsize=16)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr = sr, x_axis='time')
    plt.xlim([0, file_length])
    plt.subplot(3, 1, 2)
    D = np.abs(librosa.core.stft(y))
    D = librosa.amplitude_to_db(D,ref=np.max)
    librosa.display.specshow(D, y_axis='linear', x_axis='time', sr = sr)
    plt.ylim([0, 12000])
    plt.xlim([0, file_length])
    plt.subplot(3, 1, 3)
    top_N = len(class_map.keys())
    top_class_indices = range(0, top_N , 1)
    #y_tick_labels = [get_keys_from_value(class_map, x) for x in yticks]
    xvals = [0,len(y)/sr]
    yvals = [1,5]
    #plt.imshow(scores[:, top_class_indices].T,extent=[xvals[0],xvals[1],yvals[0],yvals[1]], aspect='auto',  cmap='gray_r')
    plt.imshow(scores[:, top_class_indices].T, aspect='auto',  cmap='gray_r', vmin=0, vmax=1)
    patch_padding = (3 / 2) / 3
    #plt.xlim([0, scores.shape[0] + patch_padding])
    plt.xlim([-0.5, scores.shape[0] - 0.5])
    def get_keys_from_value(d, val):
        return [k for k, v in d.items() if v == val]
    yticks = range(0, top_N, 1)
    plt.yticks(yticks, [get_keys_from_value(class_map, x) for x in yticks]);
    plt.savefig(f"figures\\{out_name}.png")
    plt.close(fig)
    return fig

def main():
    class_map = utilities.generate_class_map('data/foreground')

    wavs = glob.glob(f"{input_path}*.wav")
    classifier_model = utilities.load_model(f'classifier/checkpoints/{VERSION_NUMBER}.h5')
    for wav in wavs:
        out_name = wav.split('\\')[-1].split('.')[0]
        scores = process_wav(wav, classifier_model)
        display_results(wav, scores, class_map, out_name)
        df = pd.DataFrame(scores, columns=class_map.keys())
        df.to_csv(f'data/clean/{out_name}.csv')
    print("Main execution done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('version_number')
    parser.add_argument('input_path')
    args = parser.parse_args()
    VERSION_NUMBER = args.version_number
    input_path = args.input_path
    main()
