import os
import glob
import shutil
import random
import pandas as pd
from lyrics.moodclassifier import MoodClassifier
from audio.storelibrosa import storeLibrosa
from audio.pyAudioAnalysis import audioFeatureExtraction
from audio import trainSVM, values
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from openpyxl import load_workbook
import visualizeQuadrants
import _pickle as cPickle

def extractFeatures(dir):
	audioFeatureExtraction.mtFeatureExtractionToFileDir(dir, 1, 1, 0.025, 0.025, True, True, False)
	storeLibrosa(dir)

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
				random_file = seed_file.read().replace('\n', '')
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
quadrantMap = {(0,0): 3, (0,1): 2, (1,0): 4, (1,1): 1}
rootDir = 'lyrics/processed_lyrics'
classifierNB = MoodClassifier()
resultsVal = []#f1, accuracy
resultsArousal = []
resultsOverall = []
#Naive Bayes Valence Classifier
for _ in range(23):
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
	resultsVal.append((f1_score(answerVal, predictionVal, average="macro"), accuracy_score(answerVal, predictionVal)))
	with open("results_valence.txt", "w") as text_file:
		text_file.write(str(resultsVal))
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
		[_, aVal] = values.getFeatures(audio_train_dir)
		[_, aLabel] = values.getLabels(training_audio, audio_train_dir, audio_data)#get labels from audio files
		arousalModel = SVC(C = 150, kernel = "linear", probability = True)
		arousalModel.fit(aVal, aLabel)	
		cPickle.dump(arousalModel, open(arousalFileName, "wb"))
		with open('audio/test_audio.txt', 'w') as f:
		    for line in test_audio:
		        f.write("%s\n" % line)

	#start Testing!
	copyFiles("audio/wav/", audio_test_dir, test_audio)
	extractFeatures(audio_test_dir)
	arousalModel = cPickle.load(open(arousalFileName, 'rb'))
	[_, aVal] = values.getFeatures(audio_test_dir)
	[aName, aLabel] = values.getLabels(test_audio, audio_test_dir, audio_data)
	predictArousal = arousalModel.predict(aVal)
	test_songs = bimodal.loc[bimodal["Song Code"].isin(aName)]["Song"].tolist()
	test_artists = bimodal.loc[bimodal["Song Code"].isin(aName)]["Artist"].tolist()
	resultsArousal.append((f1_score(answerVal, predictionVal, average="macro"), accuracy_score(answerVal, predictionVal)))
	with open("results_arousal.txt", "w") as text_file:
		text_file.write(str(resultsArousal))
	bimodalMapping = pd.Series(bimodal["Lyrics Code"].values,index=bimodal["Song Code"]).to_dict()
	correctVals = []
	predictVals = []
	for i, name in enumerate(aName):
		index = test_lyrics.index(bimodalMapping[name])
		correctVals.append(quadrantMap[(answerVal[index],aLabel[i])])
		predictVals.append(quadrantMap[(predictionVal[index],predictArousal[i])])

	#visualizeQuadrants.visualize(correctVals, predictVals, aName, test_songs, test_artists)
	resultsOverall.append((f1_score(correctVals, predictVals, average="macro"), accuracy_score(correctVals, predictVals)))
	with open("results_overall.txt", "w") as text_file:
		text_file.write(str(resultsOverall))