import librosa
import numpy as np
import glob
import os
import csv
import soundfile as sf

# dir = "Audio_SVM_Classification/"
# dir = "testSongs/"
# dir = "testGuess/"
# dir = "mp3/"

def load_sound_files(parent_dir, file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(parent_dir + fp)
        raw_sounds.append(X)
    return raw_sounds

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name, sr = 44100)
    # X, sample_rate = sf.read(file_name)
    print("Features :",len(X), "sampled at ", sample_rate, "hz")
    # stft = np.abs(librosa.stft(X))
    # mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    # return mfccs,chroma,mel,contrast,tonnetz
    return tonnetz

# sound_file_paths = ["aircon.wav", "carhorn.wav", "play.wav", "dogbark.wav", "drill.wav",
#                    "engine.wav","gunshots.wav","jackhammer.wav","siren.wav","music.wav"]
# sound_names = ["air conditioner","car horn","children playing","dog bark","drilling","engine idling",
#                "gun shot","jackhammer","siren","street music"]

# parent_dir = 'samples/us8k/'

# raw_sounds = load_sound_files(parent_dir, sound_file_paths)
def storeLibrosa(dir):
    wavs = glob.glob(dir+"*.wav")
    csvs = glob.glob(dir+"*_lt.csv")
    for i, (w, c) in enumerate(zip(wavs, csvs)):
        print(w, "\n------------")
        print(c, "\n++++++++++++")
        tonnetz = extract_feature(w)
        with open(c, 'a') as f:
            writer = csv.writer(f)
            print(tonnetz, "\n-----------------------------------")
            for i in range(len(tonnetz)):
                row = [tonnetz[i]]
                writer.writerow(row)
# mfccs, chroma, mel, contrast, tonnetz = extract_feature(dir)
# tonnetz = extract_feature(dir)
# print "MFCC:\n", mfccs, "\n-------------"
# print "CHROMA:\n", chroma, "\n------------"
# print "MEL:\n", mel, "\n------------"
# print "CONTRAST:\n", contrast, "\n------------"

# print "TONNETZ:\n", tonnetz, "\n-----------"
# with open("testSongs/003Q1.wav_lt.csv", 'ab') as f:
# 	writer = csv.writer(f)
# 	for i in range(len(tonnetz)):
# 		row = [tonnetz[i]]
# 		print row
# 		writer.writerow(row)