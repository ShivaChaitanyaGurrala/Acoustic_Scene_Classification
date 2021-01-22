import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pathlib


def generate_spectrogram(fpath, folder):
    path = pathlib.Path.cwd().parent / folder
    if not path.is_dir():
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")
    y, sr = librosa.load(fpath)
    # S = librosa.feature.melspectrogram(y)
    plt.figure(figsize=(12, 12))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # plt.subplot(4, 2, 1)
    librosa.display.specshow(D)
    # my_path = os.path.abspath(__file__)
    fpath = fpath.split('/')
    fpath = fpath[len(fpath) - 1].split('.')[0] + "_12x12.png"
    s_file_name = path / fpath
    plt.savefig(s_file_name)
    plt.close()


sample_file = "D:/Research/SoundDataSets/SoundNN/TAU-urban-zips/TAU-urban-acoustic-scenes-2019-development.audio.1/TAU-urban-acoustic-scenes-2019-development/audio/airport-barcelona-0-0-a.wav"
generate_spectrogram(sample_file, "temp_data")