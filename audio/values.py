import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

#Retrieves the audio features from the csv files

def getFeatures(dir):

	#features = "FeatureExtracts/"
	features = dir
	#features = "trainingData/"
	#dir = "FeatureExtracts/"

	#songs = glob.glob(training+"*.wav")
	ltF = glob.glob(features+"*_lt.csv")

	Valence = []
	Arousal = []
	v = []
	a = []
	maxBeat = 0.001
	minBeat = 100000

	for i, lt in enumerate(ltF):
		file = pd.read_csv(lt, header=None)
		beat = file[0][68]
		if maxBeat < beat:
			maxBeat = beat
		if minBeat > beat:
			minBeat = beat

	#Get Valence Audio Features npy
	for i, lt in enumerate(ltF):
		file = pd.read_csv(lt, header=None)
		# print maxBeat, minBeat
		if maxBeat != minBeat:
			normBeat = (float(file[0][68])-float(minBeat))/(float(maxBeat)-float(minBeat))
		else:
			normBeat = (float(file[0][68])-float(minBeat))/(float(maxBeat)+0.001-float(minBeat))
		# print "normbeat here:", normBeat
		# print normBeat
		#print i
		#print lt
	#--------------------------------------
		v.append(file[0][0])		#ZCR        #ORIGIN
		v.append(file[0][1])		#Energy     #ORIGIN
		v.append(file[0][2])		#EE         #ORIGIN
		# v.append(file[0][3])		#SpC        #Brightness
		# v.append(file[0][4])		#SpS
		v.append(file[0][5])		#SpE
		v.append(file[0][6])		#SpF
		v.append(file[0][7])		#SpR
		
		for i in range(13):			#MFCC       #ORIGIN
			v.append(file[0][i+8])

		for i in range(12):			#CV         #ORIGIN
			v.append(file[0][i+21])

		v.append(file[0][33])		#CVD

		# v.append(float(file[0][68])/float(100))		#Beat
		# v.append(normBeat)								#Normalized Beat
		# print float(file[0][68])/float(100)
		# v.append(file[0][68])						#Beat #Untouched

		v.append(file[0][70])					#tonnetz	#Tones
		v.append(file[0][71])
		v.append(file[0][72])
		v.append(file[0][73])
		v.append(file[0][74])
		v.append(file[0][75])

		# tones = (file[0][70] + file[0][71] + file[0][72] + file[0][73] + file[0][74] + file[0][75])/6
		# v.append(tones)

		#-------------------
		
		#SD

		v.append(file[0][34])		#SD_ZCR      #ORIGIN
		v.append(file[0][35])		#SD_EnergWy   #ORIGIN
		v.append(file[0][36])		#SD_EE       #ORIGIN
		# v.append(file[0][37])		#SD_SpC
		# v.append(file[0][38])		#SD_SpS
		v.append(file[0][39])		#SD_SpE
		v.append(file[0][40])		#SD_SpF
		v.append(file[0][41])		#SD_SpR

		for i in range(13):			#SD_MFCC      #ORIGIN
			v.append(file[0][i+42])

		for i in range(12):			#SD_CV        #ORIGIN
			v.append(file[0][i+55])

		v.append(file[0][67])		#SD_CVD

		# v.append(file[0][69])		#SD_Beat
		
		

		# print v
#________________________________________

		#AROUSAL


	#----------------------------------
		# a.append(file[0][0])		#ZCR
		a.append(file[0][1])		#Energy #ORIGIN
		a.append(file[0][2])		#EE     #ORIGIN
		# a.append(file[0][3])		#SpC
		# a.append(file[0][4])		#SpS
		a.append(file[0][5])		#SpE    #ORIGIN
		a.append(file[0][6])		#SpF    #ORIGIN
		a.append(file[0][7])		#SpR    #ORIGIN

		# for i in range(13):			#MFCC
		# 	#print labels[i+8]
			# a.append(file[0][i+8])

		# for i in range(12):			#CV
		# 	#print labels[i+21]
		# 	a.append(file[0][i+21])

		# a.append(float(file[0][68])/float(100))		#Beat     #ORIGIN
		a.append(normBeat)								#Normalized Beat
		# a.append(float(file[0][68])/float(1000))		#Beat     #ORIGIN
		# a.append(file[0][68])		#Beat #ORIGIN	#Untouched

		# a.append(file[0][70])					#tonnetz	#Tones
		# a.append(file[0][71])
		# a.append(file[0][72])
		# a.append(file[0][73])
		# a.append(file[0][74])
		# a.append(file[0][75])

		#SD
		# a.append(file[0][34])		#SD_ZCR
		a.append(file[0][35])		#SD_Energy  #ORIGIN
		a.append(file[0][36])		#SD_EE      #ORIGIN
		# a.append(file[0][37])		#SD_SpC
		# a.append(file[0][38])		#SD_SpS
		a.append(file[0][39])		#SD_SpE     #ORIGIN
		a.append(file[0][40])		#SD_SpF     #ORIGIN
		a.append(file[0][41])		#SD_SpR     #ORIGIN

		# for i in range(13):			#SD_MFCC
			# print labels[i+42]
			# a.append(file[0][i+42])

		# for i in range(12):			#SD_CV
		# 	#print labels[i+55]
		# 	a.append(file[0][i+55])

		#a.append(file[0][67])		#SD_CVD

		a.append(file[0][69])		#SD_Beat	#ORIGIN
		
		ve = v
		v = np.array(v)
		# a = np.array(a)
		# norm1 = v / np.linalg.norm(v)
		norm = normalize(v[:,np.newaxis],axis=0).ravel()
		# norma = normalize(a[:,np.newaxis],axis=0).ravel()
		# print np.all(norm1 == norm)
		Valence.append(v)
		Arousal.append(a)

		v = []
		a = []

		#print file.shape
	#	print file[0][len(file[0])-1]
		#print file.head()
		#for row, label in zip(file[0], labels):
			#print row, label
			#if label == 
	#print ltF
	#print "Valence"
	#print Valence
	#print "Arousal"
	#print Arousal
	return Valence, Arousal

#-----------------------------------

def getLabels(filenames, dir, df):#A140
	#compare .wav files in dir to filenames, if it is there, return label based on audio_dataset.csv avg_valence
	c = glob.glob(dir+"*.wav_lt.csv")#"*.wav_lt.csv"
	lname = []
	alabel = []
	for file in c:
		name = file.split("/")[-1][:4]
		if name in filenames:
			avgA = df.loc[df["Song Code"] == name]["Avg_Arousal"].tolist()#at name index
			lname.append(name)
			if(avgA[0] >= 0):
				alabel.append(1)
			else:
				alabel.append(0)
	return [lname, alabel]

#------------------------------

#f = getFeatures()
#s = getLabels()
#print "Axel World"
#print s
#print "something"
#print f
#print "\n\n\n\n"
#print len(s[0]), len(s[1]), len(s[2]), len(f[0]), len(f[1])
#print len(f[0][0]), len(f[1][0])