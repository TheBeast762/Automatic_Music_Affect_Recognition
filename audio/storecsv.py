import os
import pandas as pd
import pickle
import csv
import glob

#Stores all the modified ground truth values from csvs into one csv file.
dirs = ["training/","testing/"]

# for bat in batch:
i = 0
for dir in dirs:
	Q = []
	Q.append(("Song", "Quadrant", "Valence", "Arousal", "ValClass", "ArClass"))
	for filename in glob.glob(dir+"*.wav_lt.csv"):
		name = filename.split("/")[1].split("_lt.csv")[0]
		quadrant = name.split(".")[0][-1]
		if quadrant == "1":
			Q.append((name, "Q"+quadrant, 1, 1, 1, 1))
		elif quadrant == "2":
			Q.append((name, "Q"+quadrant, 0, 1, 0, 1))
		elif quadrant == "3":
			Q.append((name, "Q"+quadrant, 0, 0, 0, 0))
		elif quadrant == "4":
			Q.append((name, "Q"+quadrant, 1, 0, 1, 0))
		i += 1
	with open(os.getcwd() + os.sep + dir+"valence_arousal_Values.csv", 'w') as f:
		wr = csv.writer(f, quoting=csv.QUOTE_ALL).writerows(Q)