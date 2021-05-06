import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

#Gets the labels & (audio) features from the wav_lt.csv files

def getSongLabels(filenames, dir, df):
	wav_files = glob.glob(dir+"*.wav_lt.csv")
	alabel = []
	lname = []
	for file in wav_files:
		name = file.split("/")[-1][:4]
		if name in filenames:
			avgA = list(df.loc[df["Song Code"] == name]["Avg_Arousal"])[0]
			lname.append(name)
			if(avgA >= 0):
				alabel.append(1)
			else:
				alabel.append(0)
	return [lname, alabel]

def getSongFeatures(dir):#indices used for features matching with pyAudioAnalysis.stFeatureExtraction()
	ltFiles = glob.glob(dir+"*_lt.csv")
	val = []
	aro = []
	v = []
	a = []
	minB = np.inf
	maxB = -np.inf

	for i, lt in enumerate(ltFiles):
		file = pd.read_csv(lt, header=None)
		beat = file[0][68]
		if maxB < beat:
			maxB = beat
		if minB > beat:
			minB = beat

	#Get Valence Audio Features npy
	for i, lt in enumerate(ltFiles):
		file = pd.read_csv(lt, header=None)
		if minB != maxB:
			normBeat = (float(file[0][68])-float(minB))/(float(maxB)-float(minB))
		else:
			normBeat = (float(file[0][68])-float(minB))/(float(maxB)+0.001-float(minB))
		#VALENCE
		for i in range(13):			
			v.append(file[0][i+8])		#MFCC       
			v.append(file[0][i+21])		#CV         
		for i in range(6):				#Tonnetz
			v.append(file[0][i+70])
		for i in range(3):
			v.append(file[0][34+i])		#SD
			v.append(file[0][39+i])		#SD_SpX
			v.append(file[0][5+i])		#SpX
			v.append(file[0][0+i])		#ZCR/Energy/EE
		for i in range(13):				#SD_MFCC      
			v.append(file[0][i+42])
		for i in range(13):				#SD_CV        
			v.append(file[0][i+55])

		#AROUSAL
		for i in range(2):
			a.append(file[0][1+i])		#Energy/EE
			a.append(file[0][35+i])		#SD_Energy/EE
		for i in range(3):
			a.append(file[0][5+i])		#SpX   
			a.append(file[0][39+i]) 	#SD_SpX
		a.append(normBeat)				#[0,1] Beat
		a.append(file[0][69])			#SD_Beat
		
		ve = v
		v = np.array(v)
		val.append(v)
		aro.append(a)

		v = []
		a = []
	return val, aro