import glob
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import datasets
import _pickle as cPickle
import matplotlib.pyplot as plt
import csv

def saveToCsv(fileName, songName, valence, arousal):
	print(fileName)
	Q1 = 0
	Q2 = 0
	Q3 = 0
	Q4 = 0
	data = []
	data.append(("Name", "Valence", "Arousal"))
	for (n, v, a) in zip(songName["Song"], valence, arousal):
		data.append((n, v, a))
		if n[6] == "4":
			Q4 += 1
		elif n[6] == "3":
			Q3 += 1
		elif n[6] == "2":
			Q2 += 1
		elif n[6] == "1":
			Q1 += 1
	data.append(("Quadrants", Q1, Q2, Q3, Q4))
	file = open(fileName, 'w')
	wr = csv.writer(file, quoting=csv.QUOTE_ALL)
	for a in data:
		wr.writerow(a)
	file.close()

	print("Save Complete for", fileName)

def trainSVM(cValue, trainValues, trainLabels, testValues, testLabels):
	trainModel = SVC(C = cValue, kernel = "linear", probability = True)
	trainModel.fit(trainValues, trainLabels)

	result = trainModel.score(testValues, testLabels)
	print("Test Score and Train Score")
	print(result, trainModel.score(trainValues, trainLabels))
	testing = np.array(testValues[0]).reshape(1, -1)
	print(trainModel.predict(testing))
	return(trainModel, result)

def crossValidationSVM(cValue, cv, trainValues, trainLabels):
	print("Training Cross-Validation with", cv, "Times")
	model = SVC(C = cValue, kernel = "linear")
	scores = cross_val_score(model, trainValues, trainLabels, cv=cv, scoring = "f1_macro")
	print("CrossVal")
	print(scores)
	print("Accuracy:", scores.mean(), (scores.std() * 2))
	print("Cross-Validation Complete")
	trainValues = np.array(trainValues)
	trainLabels = np.array(trainLabels)
	kfold = KFold(n_splits=cv)
	for i, (train, test) in enumerate (kfold.split(trainValues, trainLabels)):
		model = SVC(C = cValue, kernel = "linear")
		model.fit(trainValues[train], trainLabels[train])
		y_Test = model.predict(trainValues[test])
		from sklearn.metrics import classification_report, confusion_matrix
		print("KFold number", i)
		print(confusion_matrix(y_Test, trainLabels[test]))
		print(classification_report(y_Test, trainLabels[test]))

	return scores


