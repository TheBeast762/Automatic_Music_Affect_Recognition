import math
import json

class MoodClassifier(object):
	def get_vocab(self, X):
		vocab = {}
		data_lines = X.splitlines()
		for row in data_lines:
			tokens = row.split(' ')
			for word in tokens:
				if str(word).isspace():
					continue
				elif word in vocab:
					vocab[word] += 1
				else:
					vocab[word] = 1
		return vocab

	def fit(self, X, truth_labels):
		data_len = len(X)
		self.word_count = {'valencePos': {}, 'valenceNeg': {}}
		self.mood_class = {	'valencePos': math.log(sum(1 for label in truth_labels if label == 1) / data_len), 
							'valenceNeg': math.log(sum(1 for label in truth_labels if label == 0) / data_len)}
		self.vocab = set()

		for lyric, truthv in zip(X, truth_labels):
			if truthv == 1:
				valClass = 'valencePos'
			elif truthv == 0:
				valClass = 'valenceNeg'
			counts = self.get_vocab(lyric)
			for word, count in counts.items():
				if len(word) < 1:
					continue
				if word not in self.vocab:
					self.vocab.add(word)
				if word not in self.word_count[valClass]:
					self.word_count[valClass][word] = 0.0

				self.word_count[valClass][word] += count

	def predict(self, X):
		resultVal = []
		resultAro = []
		for lyric in X:
			counts = self.get_vocab(lyric)
			valP_score = 0
			valN_score = 0
			for word, count in counts.items():
				if word not in self.vocab:
					continue
				#laplacing
				log_w_given_posVal = math.log( (self.word_count['valencePos'].get(word, 0.0) + 1) / (sum(self.word_count['valencePos'].values()) + len(self.vocab)) )
				log_w_given_negVal = math.log( (self.word_count['valenceNeg'].get(word, 0.0) + 1) / (sum(self.word_count['valenceNeg'].values()) + len(self.vocab)) )

				valP_score += log_w_given_posVal
				valN_score += log_w_given_negVal

			valP_score += self.mood_class['valencePos']
			valN_score += self.mood_class['valenceNeg']

			biggest = 0
			if (valP_score >= valN_score):
				resultVal.append(1)
			elif (valP_score < valN_score):
				resultVal.append(0)
		return resultVal

	def writeClf(self, path):#to Json file
		dict_list = [self.mood_class, self.word_count, dict.fromkeys(self.vocab, 0)]
		with open(path, 'w') as outfile:
			json.dump(dict_list, outfile)

	def readClf(self, path):
		self.mood_class = {}
		self.word_count = {}
		self.vocab = set()
		with open(path) as json_file: 
			data = json.load(json_file)
			self.mood_class = dict(data[0])
			self.word_count = dict(data[1])
			self.vocab = set(data[2].keys())
