import glob
import pandas as pd
import numpy as np
import _pickle as cPickle
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import string
import csv
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.model_selection import cross_val_score
import values as value

test_dirs = ["1_16_Test/","2_26_Test/"]

#Save st, mt, and lt in csv
#Fe.featuresExtract(folder)

#FileNames for Trained Data
ValenceFileName = "ValenceClassifier.file"
ArousalFileName = "ArousalClassifier.file"

ValenceFileName = "01_1VClassifier.file"
ArousalFileName = "01_1AClassifier.file"

#Models
ValenceModel = cPickle.load(open(ValenceFileName, 'rb'))
ArousalModel = cPickle.load(open(ArousalFileName, 'rb'))
valence_list, arousal_list, lname_list, vlabel_list, alabel_list = [], [], [], [], []
for dir in test_dirs:
	[Valence, Arousal] = value.getFeatures(dir)
	[LName, VLabel, ALabel] = value.getLabels(dir)


#Store in pandas dataframe
VX = pd.DataFrame(Valence)
Vy = pd.Series(VLabel)

AX = pd.DataFrame(Arousal)
Ay = pd.Series(ALabel)
# print "Valence Features"
# print VX
# print "Arousal Features"
# print AX

#Get train and test data
#VX_train, VX_test, Vy_train, Vy_test = train_test_split(VX, Vy, test_size = 0.5)
#AX_train, AX_test, Ay_train, Ay_test = train_test_split(AX, Ay, test_size = 0.5)

#Results
Vresult = ValenceModel.score(VX, Vy)
Aresult = ArousalModel.score(AX, Ay)


Vpred = ValenceModel.predict(VX)
Apred = ArousalModel.predict(AX)

# Vscore = cross_val_score(ValenceModel, VX, Vy, cv=10)
# Ascore = cross_val_score(ArousalModel, AX, Ay, cv=10)

Vscore = 1
Ascore = 1

#How far is the value from the hyperplane.
VP = np.array(ValenceModel.decision_function(VX))
AP = np.array(ArousalModel.decision_function(AX))
# print VP
# print AP

VB = ValenceModel.predict_proba(VX)
AB = ArousalModel.predict_proba(AX)

# ast = ValenceModel.get_params(True)
# print ast

dV = pd.DataFrame(Vpred)
dA = pd.DataFrame(Apred)

array = dV.index.values
# print "Valence"
# print dV
# print "Arousal"
# print dA, "\n\n"

correct = 0
total = 0
cValence = 0
cArousal = 0
q1g, q2g, q3g, q4g = [0, 0, 0, 0]
q1, q2, q3, q4 = [0, 0, 0, 0]
q1t, q2t, q3t, q4t = [0, 0, 0, 0]

correctResults = []
wrongResults = []
correctResults.append(("Song", "Quadrant", "Correct_Valence", "Correct_Arousal", "Guessed_Valence", "Guessed_Arousal"))
wrongResults.append(("Song", "Quadrant", "Correct_Valence", "Correct_Arousal", "Guessed_Valence", "Guessed_Arousal"))
maxV = -10
minV = 10
maxA = -10
minA = 10


for i, (Name, Vvalue, Avalue, val, ar, vp, ap, vb, ab) in enumerate(zip(LName, VLabel, ALabel, Vpred, Apred, VP, AP, VB, AB)):

	
	print("-------------------------")
	print("Name:", Name)
	print("Correct Values:", Vvalue, Avalue)
	print("Prediction Guessed Values:", val, ar)

	if maxV < vp:
		maxV = vp
	if minV > vp:
		minV = vp
	if maxA < ap:
		maxA = ap
	if minA > ap:
		minA = ap

	# if vb[0] > vb[1] and vb[0] - 0.01 <= vb[1] + 0.01:
	# # 	val = 0
	# if vb[0] + 0.01 >= vb[1] - 0.01 and vb[1] > vb[0]:			#Only this condition is being met. 587 valence. 52.89%
	# 	val = 1													#Now it is 584 valence. 52.67%
		# temp = vb[0]
		# vb[0] = vb[1]
		# vb[1] = temp
	# if ab[0] > ab[1] and ab[0] - 0.01 <= ab[1] + 0.01:
		# ar = 0
	# elif ab[0] < ab[1] and ab[0] + 0.01 >= ab[1]:
		# ar = 1

	# if vb[0] > vb[1]:
		# val = 0


#This is for checking valence inconsistency

	# if vb[0] <= vb[1]:											#Only valence faces problems. 580 Valence. 52.67%
		# val = 1													#Now this is 587 Valence. 52.89%
		


		# temp = VB[i][0]
		# VB[i][0] = VB[i][1]
		# VB[i][1] = temp

	# if ab[0] > ab[1]:
	# 	ar = 0
	# elif ab[0] <= ab[1]:
	# 	ar = 1 

	print("Prediction Probability Guessed Values:", val, ar)
	print("Valence Distance from the Hyperplane:", vp)
	print("Arousal Distance from the Hyperplane:", ap)
	print("Probabilities:", vb, ab)
	print("\n")
	
	quadrant = "Q1"
	# if Vvalue == val:
	if val > 0 and Vvalue == 1:
		cValence += 1
	elif val <= 0 and Vvalue == 0:
		cValence += 1

	# if Avalue == ar:
	if ar > 0 and Avalue == 1:
		cArousal += 1
	elif ar <= 0 and Avalue == 0:
		cArousal += 1

	isCorrect = False
	# if Vvalue == val and Avalue == ar:
	if val > 0 and ar > 0 and Vvalue == 1 and Avalue == 1:	#Q1
		correct += 1
		isCorrect = True
	elif val <= 0 and ar > 0 and Vvalue == 0 and Avalue == 1:	#Q2
		correct += 1
		isCorrect = True
	elif val <= 0 and ar <= 0 and Vvalue == 0 and Avalue == 0:	#Q3
		correct += 1
		isCorrect = True
	elif val > 0 and ar <= 0 and Vvalue == 1 and Avalue == 0:	#Q4
		correct += 1
		isCorrect = True

	if isCorrect == True:
		print(Name," is located at Valence:", Vvalue,"and Arousal:",Avalue, "\n")
		# if val == 1 and ar == 1:
		if val > 0 and ar > 0:
			quadrant = "Q1"
			q1g += 1
		# elif val == 0 and ar == 1:
		elif val <= 0 and ar > 0:
			quadrant = "Q2"
			q2g += 1
		# elif val == 0 and ar == 0:
		elif val <= 0 and ar <= 0:
			quadrant = "Q3"
			q3g += 1
		# elif val == 1 and ar == 0:
		elif val > 0 and ar <= 0:
			quadrant = "Q4"
			q4g += 1
		correctResults.append((Name, quadrant, Vvalue, Avalue, val, ar))
	else:
		if Vvalue == 1 and Avalue == 1:
			quadrant = "Q1"
		elif Vvalue == 0 and Avalue == 1:
			quadrant = "Q2"
		elif Vvalue == 0 and Avalue == 0:
			quadrant = "Q3"
		elif Vvalue == 1 and Avalue == 0:
			quadrant = "Q4"
		wrongResults.append((Name, quadrant, Vvalue, Avalue, val, ar))
	# if val == 1 and ar == 1:
	if val > 0 and ar > 0:
		q1 += 1
	# elif val == 0 and ar == 1:
	elif val <= 0 and ar > 0:
		q2 += 1
	# elif val == 0 and ar == 0:
	elif val <= 0 and ar <= 0:
		q3 += 1
	# elif val == 1 and ar == 0:
	elif val > 0 and ar <= 0:
		q4 += 1


	if Vvalue == 1 and Avalue == 1:
		q1t += 1
	elif Vvalue == 0 and Avalue == 1:
		q2t += 1
	elif Vvalue == 0 and Avalue == 0:
		q3t += 1
	elif Vvalue == 1 and Avalue == 0:
		q4t += 1
	total += 1

print("----------------------")
print("Total Q1:", q1t)
print("Guessed Q1:", q1)
print("Correct Q1:", q1g)
print("\n")

print("----------------------")
print("Total Q2:", q2t)
print("Guessed Q2:", q2)
print("Correct Q2:", q2g)
print("\n")

print("----------------------")
print("Total Q3:", q3t)
print("Guessed Q3:", q3)
print("Correct Q3:", q3g)
print("\n")

print("----------------------")
print("Total Q4:", q4t)
print("Guessed Q4:", q4)
print("Correct Q4:", q4g)
print("\n")

print("Correct Valence:", cValence)
print("Correct Arousal:", cArousal)

print("Correct:", correct)
print("Total:", total)
print(round(float(correct)/float(total), 4)*100, "%")


print("Valence Result:")
print(Vresult)

print("Arousal Result:")
print(Aresult)

print("Cross Validations Scores:")
print("Valence Scores")
print(Vscore)
print("Arousal Scores")
print(Ascore)

#Precision = True Positive / (True Positive + False Positive) = True Positive / Total Predicted Positive
precision = float(float(correct) / (float(correct) + (float(total) - float(correct))))
valprecision = float(float(cValence) / (float(cValence) + (float(total) - float(cValence))))
arprecision = float(float(cArousal) / (float(cArousal) + (float(total) - float(cArousal))))
print("Total Precision:", precision)
print("Valence Precision:", valprecision)
print("Arousal Precision:", arprecision)

#Recall = True Positive / (True Positive + False Negative) = True Positive / Total Actual Positive
#True Positive / Total Actual Positive

#F1 = 2((Precision * Recall)/(Precision + Recall))

VyA = Vy.tolist()
AyA = Ay.tolist()

VyList = [int(a) for a in VyA]
AyList = [int(a) for a in AyA]

Vpred = Vpred.tolist()
Apred = Apred.tolist()

Vpred = [int(a) for a in Vpred]
Apred = [int(a) for a in Apred]

#On extracting True Positives, False Positives, ...
#True Negatives, False Negatives

#True Negative, False Positive, False Negative, True Positive
vtn, vfp, vfn, vtp = confusion_matrix(VyList, Vpred).ravel()
atn, afp, afn, atp = confusion_matrix(AyList, Apred).ravel()

print(vtn, vfp, vfn, vtp)
print(atn, afp, afn, atp)

print("Valence Confusion Matrix")
print(confusion_matrix(VyList, Vpred))
print("Valence Classification Report")
print(classification_report(VyList, Vpred))
print("Actual Data:",VyList)
print("Predicted:  ",Vpred)

print("Arousal Confusion Matrix")
print(confusion_matrix(AyList, Apred))
print("Arousal Classification Report")
print(classification_report(AyList, Apred))
print("Actual Data:",AyList)
print("Predicted:  ",Apred)

print("Max Decision for Valence and Arousal:")
print(maxV, maxA)
print("Min Decision for Valence and Arousal:")
print(minV, minA)

file = open("01Correct_A_V_Values.csv", 'w')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)
for q in correctResults:
	wr.writerow(q)
file.close()


file = open("01Wrong_A_V_Values.csv", 'w')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)
for q in wrongResults:
	wr.writerow(q)
file.close()

plt.figure("Valence vs Arousal")
plt.xlim(minV+(minV/3), maxV+(maxV/3))
plt.ylim(minA+(minA/3), maxA+(maxA/3))
# plt.plot( VP, AP,"ro")
# color = np.cos(VP)	#Makes it look like nips
# color = np.cos(AP)	#Same
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# print colors
color = []
names = []
for l in LName:
	#ano = l.split("_")
	#c = ano[1].split(".")
	if l.find("Q1") > -1:
		color.append(colors["green"])
		names.append(l[l.find("Q1")]+l[l.find("Q1")+1])
	elif l.find("Q2") > -1:
		color.append(colors["red"])
		names.append(l[l.find("Q2")]+l[l.find("Q2")+1])
	elif l.find("Q3") > -1:
		color.append(colors["blue"])
		names.append(l[l.find("Q3")]+l[l.find("Q3")+1])
	elif l.find("Q4") > -1:
		color.append(colors["purple"])
		names.append(l[l.find("Q4")]+l[l.find("Q4")+1])
plt.scatter(VP, AP, c=color)
# for (n, v, a, c) in zip(names, VP, AP, color):
# 	plt.scatter(VP, AP, c=c, label=n)
# for l, v, a in zip(LName, VP, AP):
	#ano = l.split("_")
	#ano[1] = ano[1].split(".")
	# plt.annotate(ano[0]+ano[1][0], xy=(v, a))
	# plt.annotate(ano[1][0], xy=(v,a))
	#plt.annotate(ano[0], xy=(v, a))
plt.xlabel("Valence")
plt.ylabel("Arousal")

# plt.figure("Arousal")
# plt.plot(VP, AP, "bo")
# plt.ylabel("Arousal")
# plt.xlabel("Valence")

plt.show()