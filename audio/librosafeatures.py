import librosa
import numpy as np
import glob
import csv

def feature_extract(file_name):
    data, sr = librosa.load(file_name, sr = 44100)
    print(file_name, " Features :",len(data))
    return np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data), sr=sr).T,axis=0)

def librosaFeatures(dir):
    wav_files = glob.glob(dir+"*.wav")
    csv_files = glob.glob(dir+"*_lt.csv")
    for (w, c) in zip(wav_files, csv_files):
        feature = feature_extract(w)
        with open(c, 'a') as f:
            csv_writer = csv.writer(f)
            print(feature, "\n__________")
            for i in range(len(feature)):
                csv_writer.writerow([feature[i]])