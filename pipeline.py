# This should be used to get the audio recording and convert into spectrogram image and return
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from csv import reader

TRAIN_FILE = "fold1_train"
TEST_FILE = "fold1_test"


def generate_spectrogram(fpath, label, folder):
    path = pathlib.Path.cwd().parent / folder / label
    if not path.is_dir():
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        else:
            print("Folder was created")
    y, sr = librosa.load(fpath)
    # S = librosa.feature.melspectrogram(y)
    plt.figure(figsize=(12, 8))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # plt.subplot(4, 2, 1)
    librosa.display.specshow(D)
    # my_path = os.path.abspath(__file__)
    fpath = fpath.split('/')
    fpath = fpath[len(fpath) - 1].split('.')[0] + ".png"
    s_file_name = path / fpath
    plt.savefig(s_file_name)
    plt.close()


path = pathlib.Path.cwd()
readpath = path.parent / 'TAU-urban-acoustic-meta' / 'evaluation_setup'

for file in readpath.iterdir():
    if TRAIN_FILE in file.as_posix():
        folder = "train"
    elif TEST_FILE in file.as_posix():
        folder = "test"
    if folder is not None:
        if file.is_file():
            with open(file, 'r') as read_obj:
                csv_reader = reader(read_obj)
                header = next(csv_reader)
                # Check file as empty
                if header is not None:
                    # Iterate over each row after the header in the csv
                    for row in csv_reader:
                        #
                        search_file = "**/" + row[0]
                        # use pathlib glob() to fetch the audio file
                        audio_file_path = path.parent.glob(search_file)
                        for x in list(audio_file_path):
                            generate_spectrogram(x.as_posix(), row[1], folder)
