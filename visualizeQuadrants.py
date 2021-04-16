import matplotlib.pyplot as plt
import pyaudio  
import wave  
import time

def play_audio(songname): 
	songDuration = 10
	f = wave.open(songname,"rb")   
	p = pyaudio.PyAudio()  
	stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
	                channels = f.getnchannels(),  
	                rate = f.getframerate(),  
	                output = True)  
	data = f.readframes(1024)  
	t_end = time.time() + songDuration#seconds to play
	while time.time() < t_end: 
	    stream.write(data)  
	    data = f.readframes(1024)  
	stream.stop_stream()  
	stream.close()  
	p.terminate()   

def visualize(correctVals, predictVals, songCodes, songTitles, songArtists):#
	for i, val in enumerate(correctVals):
		fig, ax = plt.subplots()
		ax.set_aspect('equal')
		ax.set_yticks([])
		ax.set_xticks([])
		plt.ylabel("Arousal", loc='top')
		plt.xlabel("Valence", loc='right')
		ax.set_ylim(-1,1)
		ax.set_xlim(-1,1)
		rectangle = plt.Rectangle((0,0), 1, 1, fc='lime')
		plt.gca().add_patch(rectangle)
		rectangle = plt.Rectangle((0,0), 1, -1, fc='lightblue')
		plt.gca().add_patch(rectangle)
		rectangle = plt.Rectangle((0,0), -1, 1, fc='salmon')
		plt.gca().add_patch(rectangle)
		rectangle = plt.Rectangle((0,0), -1, -1, fc='purple')
		plt.gca().add_patch(rectangle)
		plt.text(0.15, 0.65, "Excited", color='gray')
		plt.text(0.4, 0.4, "Delighted", color='gray')
		plt.text(0.65, 0.15, "Happy", color='gray')
		plt.text(0.65, -0.25, "Content", color='gray')
		plt.text(0.4, -0.5, "Relaxed", color='gray')
		plt.text(0.15, -0.75, "Calm", color='gray')
		plt.text(-0.35, -0.75, "Tired", color='gray')
		plt.text(-0.6, -0.5, "Bored", color='gray')
		plt.text(-0.85, -0.25, "Depressed", color='gray')
		plt.text(-0.85, 0.15, "Frustrated", color='gray')
		plt.text(-0.6, 0.4, "Angry", color='gray')
		plt.text(-0.35, 0.65, "Tense", color='gray')
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_position('zero')
		ax.spines['left'].set_position('zero')
		ax.plot(val[0]-0.5,val[1]-0.5,'s', markersize=20)
		ax.plot(predictVals[i][0]-0.5,predictVals[i][1]-0.5,'X', markersize=20)
		plt.gca().legend(('Predict','Correct'), bbox_to_anchor=(1.05, 1),loc='upper left', labelspacing=1.0, frameon=False)
		plt.text(-1, -1.1, songTitles[i] + " - " + songArtists[i])#Print Song title + artist
		plt.show()
		play_audio("audio/wav/"+songCodes[i]+".wav")