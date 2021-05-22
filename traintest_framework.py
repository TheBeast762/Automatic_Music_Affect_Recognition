import os
import glob
import shutil
import random
import pandas as pd
from lyrics.moodclassifier import MoodClassifier
from audio.librosafeatures import librosaFeatures
from audio.pyAudioAnalysis import audioFeatureExtraction
from audio import sound_values
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import visualizeQuadrants
import _pickle as cPickle

def extractFeatures(dir):
	audioFeatureExtraction.mtFeatureExtractionToFileDir(dir, 1, 1, 0.025, 0.025, True, True, False)
	librosaFeatures(dir)

def cleanDir(path):
	files = glob.glob(path)
	for f in files:
	    os.remove(f)

def copyFiles(src, dst, files):
	cleanDir(dst + "*")
	srcdir = glob.glob(src + "*")
	for file in srcdir:
		filename = file.split("/")[-1][:4]
		if filename in files:
			shutil.copy(file, dst)

def getData(root,randomness, loadSeed):#0.0 seed if you want new seed
	data = []
	truthV = []
	fnameArr = []

	lyrics_dataset = pd.read_csv("lyrics/truth_lyrics.csv", index_col=0, names=["Avg_Valence","Avg_Arousal","SD_Valence","SD_Arousal"])
	for dir_name, subdir_list, file_list in os.walk(root):
		for file_name in file_list:
			file_dir = '%s/%s' % (root,file_name)
			file = open(file_dir, "r")
			data.append(file.read())
			trueV = 0
			valence = float(lyrics_dataset.loc[os.path.splitext(file_name)[0]]["Avg_Valence"])
			if(valence >= 0):
				trueV = 1
			elif(valence < 0):
				trueV = 0
			truthV.append(trueV)
			fnameArr.append(file_name)

	#random shuffle
	if randomness:
		random_float = 0.0
		if loadSeed: 
			with open("lyrics/seed.txt", "r") as seed_file:
				random_float = float(seed_file.read().replace('\n', ''))
		else:
			random_float = random.random()
		with open("lyrics/seed.txt", "w") as seed_file:
			seed_file.write(str(random_float))
		random.Random(random_float).shuffle(data)
		random.Random(random_float).shuffle(fnameArr)
		random.Random(random_float).shuffle(truthV)
	return data, truthV, fnameArr

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

trainModels = True
if len(sys.argv) > 2:
	if sys.argv[1] == '-reload':
		print("   [Training phase skipped! Loading already trained model]")
		trainModels = False
	else:
		print("   [run with option -reload to test the already trained model ensemble!]")
quadrantMap = {(0,0): 3, (0,1): 2, (1,0): 4, (1,1): 1}
rootDir = 'lyrics/processed_lyrics'
classifierNB = MoodClassifier()
resultsVal = []#f1, accuracy
resultsArousal = []
resultsOverall = []
#Naive Bayes Valence Classifier
data, truthV, filenames = getData(rootDir,True,not trainModels)
total_size = len(data)
training_size = int(total_size * 0.9)
testing_size = total_size - training_size
if trainModels:
	classifierNB.fit(data[:(training_size)],truthV[:(training_size)])
	classifierNB.writeClf("lyrics/modelNB.json")
classifierNB.readClf("lyrics/modelNB.json")
predictionVal = classifierNB.predict(data[(testing_size*-1):])#only predict for test lyric files
answerVal = truthV[(testing_size*-1):]#only find truth for test lyric files
print("Valence F1:",f1_score(answerVal, predictionVal, average="macro"), " accuracy:", accuracy_score(answerVal, predictionVal))
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#VALENCE^               SPLIT                AROUSALv          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
audio_train_dir = "audio/training/"
audio_test_dir = "audio/testing/"
arousalFileName = "arousalSVMclf.file"
bimodal = pd.read_csv("bimodal.csv", sep='\t')#162
audio_data = pd.read_csv("audio_dataset.csv")#162
test_lyrics = [file.split(".")[0] for file in filenames[(testing_size*-1):]]
test_audio = bimodal.loc[bimodal['Lyrics Code'].isin(test_lyrics)]["Song Code"].tolist()
if trainModels:
	training_lyrics = [file.split(".")[0] for file in filenames[:(training_size)]]
	training_audio = audio_data.loc[~audio_data['Song Code'].isin(test_audio)]["Song Code"].tolist()#audio dataset without bimodal test set
	copyFiles("audio/wav/", audio_train_dir, training_audio)
	extractFeatures(audio_train_dir)
	[_, aVal] = sound_values.getSongFeatures(audio_train_dir)
	[_, aLabel] = sound_values.getSongLabels(training_audio, audio_train_dir, audio_data)#get labels from audio files
	arousalModel = SVC(C = 150, kernel = "linear", probability = True)
	arousalModel.fit(aVal, aLabel)	
	cPickle.dump(arousalModel, open(arousalFileName, "wb"))
	with open('audio/test_audio.txt', 'w') as f:
	    for line in test_audio:
	        f.write("%s\n" % line)

#start Testing!
copyFiles("audio/wav/", audio_test_dir, test_audio)
testWavFiles = glob.glob(audio_test_dir+'*.wav')
extractFeatures(audio_test_dir)
arousalModel = cPickle.load(open(arousalFileName, 'rb'))
[_, aVal] = sound_values.getSongFeatures(audio_test_dir)
[aName, aLabel] = sound_values.getSongLabels(test_audio, audio_test_dir, audio_data)
predictArousal = arousalModel.predict(aVal)
resultsArousal.append((f1_score(answerVal, predictionVal, average="macro"), accuracy_score(answerVal, predictionVal)))
bimodalMapping = pd.Series(bimodal["Lyrics Code"].values,index=bimodal["Song Code"]).to_dict()

#Assess quality of testing
correctVals = []
predictVals = []
songCodes = []
test_songs = []
test_artists = []
for i, name in enumerate(aName):#align song lists
	index = test_lyrics.index(bimodalMapping[name])
	test_songs.append(bimodal.loc[bimodal["Song Code"] == name]["Song"].to_string(index=False))
	test_artists.append(bimodal.loc[bimodal["Song Code"] == name]["Artist"].to_string(index=False))
	songCodes.append([file for file in testWavFiles if name in file][0])
	correctVals.append(quadrantMap[(answerVal[index],aLabel[i])])
	predictVals.append(quadrantMap[(predictionVal[index],predictArousal[i])])

input('Push any button to start the Visual Tool!')
visualizeQuadrants.visualize(correctVals, predictVals, songCodes, test_songs, test_artists)
print("Arousal F1:", f1_score(correctVals, predictVals, average="macro"), " accuracy:", accuracy_score(correctVals, predictVals))