import math
import json

class MoodClassifier(object):
	def getVocab(self, Data):
		wordVocab = {}
		lines = Data.splitlines()
		for line in lines:
			tokens = line.split(" ")
			for word in tokens:
				if (str(word).isspace()):
					continue
				elif (word not in wordVocab):
					wordVocab[word] = 1
				else:
					wordVocab[word] += 1
		return wordVocab

	def fit(self, Data, TruthV):
		self.log_class_priors = {}
		self.word_counts = {}
		self.vocab = set()

		n = len(Data)
		self.log_class_priors['posV'] = math.log(sum(1 for label in TruthV if label == 1) / n)
		self.log_class_priors['negV'] = math.log(sum(1 for label in TruthV if label == 0) / n)

		self.word_counts['posV'] = {}
		self.word_counts['negV'] = {}

		for lyric, truthv in zip(Data, TruthV):
			if truthv == 1:
				valClass = 'posV'
			elif truthv == 0:
				valClass = 'negV'
			counts = self.getVocab(lyric)
			for word, count in counts.items():
				if len(word) < 1:
					continue
				if word not in self.vocab:
					self.vocab.add(word)
				if word not in self.word_counts[valClass]:
					self.word_counts[valClass][word] = 0.0

				self.word_counts[valClass][word] += count

	def predict(self, Data):
		resultVal = []
		resultAro = []
		for lyric in Data:
			counts = self.getVocab(lyric)
			valP_score = 0
			valN_score = 0
			for word, count in counts.items():
				if word not in self.vocab:
					continue
				#laplacing
				log_w_given_posVal = math.log( (self.word_counts['posV'].get(word, 0.0) + 1) / (sum(self.word_counts['posV'].values()) + len(self.vocab)) )
				log_w_given_negVal = math.log( (self.word_counts['negV'].get(word, 0.0) + 1) / (sum(self.word_counts['negV'].values()) + len(self.vocab)) )

				valP_score += log_w_given_posVal
				valN_score += log_w_given_negVal

			valP_score += self.log_class_priors['posV']
			valN_score += self.log_class_priors['negV']

			biggest = 0
			if (valP_score >= valN_score):
				resultVal.append(1)
			elif (valP_score < valN_score):
				resultVal.append(0)
		return resultVal

	def writeClf(self, path):#to Json file
		dict_list = [self.log_class_priors, self.word_counts, dict.fromkeys(self.vocab, 0)]
		with open(path, 'w') as outfile:
			json.dump(dict_list, outfile)

	def readClf(self, path):
		self.log_class_priors = {}
		self.word_counts = {}
		self.vocab = set()
		with open(path) as json_file: 
			data = json.load(json_file)
			self.log_class_priors = dict(data[0])
			self.word_counts = dict(data[1])
			self.vocab = set(data[2].keys())
