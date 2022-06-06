import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import soundfile as sf
from base64 import encodebytes
from dotenv import load_dotenv
from PIL import Image



app = Flask(__name__)
load_dotenv()
dic = {0 : 'CHEST', 1 : 'HEAD', 2 : 'MIX'}

model = load_model('mel_model.h5')
model2 = load_model('mfcc_model.h5')

model.make_predict_function()

# 1st prediction
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model.predict(i)
	print("[mel] ",p)
	return [p]

# 2nd prediction
def predict_label2(img_path2):
	i = image.load_img(img_path2, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model2.predict(i)
	print("[MFCC]   ",p)

	return [p]



def getTentativeWeight(p):
	
	tentative_max = float("-inf")
	pos_idx = -1

	for idx, val in enumerate(p[0]):
		if val > tentative_max:
			tentative_max = val
			pos_idx = idx	
	
	return pos_idx


@app.route("/whoisthis")
def whoisthis():
	return jsonify("jade wensyl fariscal")


# main computation
@app.route("/getWavFileToProcess", methods=['POST'])
def maine():
	# open wav file from flutter
	with open('wavfiles/temp.wav', mode='wb') as f:
		f.write(request.data)
	# get the file directly
	direct_path_file = 'temp'
	scale_file = 'wavfiles/' + direct_path_file + '.wav'
	filename = direct_path_file
    	
	# load file
	plt.switch_backend('agg')
    # call log mel spectrogram
	y, sr = librosa.load(scale_file)
	ps = librosa.feature.melspectrogram(y=y, sr=sr)
	ps_db= librosa.power_to_db(ps, ref=1.0)
	librosa.display.specshow(ps_db, x_axis='s', y_axis='log')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig("static/" + filename + "_mel.png") 
	plt.clf()

    # call log mfcc
	x, fs = librosa.load(scale_file)
	mfccs = librosa.feature.mfcc(x, sr=fs)
	mfccs = sklearn.preprocessing.scale(mfccs)
	librosa.display.specshow(mfccs, sr=fs, x_axis='time')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig("static/" + filename + "mfcc.png") 
	# CHEST/f1_arpeggios_belt_c_e.wav_mel.png
	plt.clf()

	#print spectrogram
	y, sr = librosa.load(scale_file)
	ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
	librosa.display.specshow(ps, y_axis='log', x_axis='time')
	plt.tight_layout()
	plt.savefig("static/" + filename + "_spect.png")

	img_path = "static/" + direct_path_file + "_mel.png"
	img_path2 = "static/" + direct_path_file + "mfcc.png"


	#mel PREDICTION
	p = predict_label(img_path)[0]
	#MFCC PREDICTION
	p3 = predict_label2(img_path2)[0]

	# PRINT RESULT /
	print("___________________________________________")
	print(p)
	print(getTentativeWeight(p), dic[getTentativeWeight(p)])
	print(dic[getTentativeWeight(p)])
	print("___________________________________")
	print(p3)
	print(getTentativeWeight(p3), dic[getTentativeWeight(p3)])
	print(dic[getTentativeWeight(p3)])
	print("\n")
	

	# p1_result = tentative_max * 100 
	# p3_result = tentative_max2 * 100
	pos_idx = getTentativeWeight(p)
	pos_idx2 = getTentativeWeight(p3)
	
	# file1 = direct_path_file + "_mel.png"
	# file2 = direct_path_file + "_mfcc.png"
	final_output = 'N/A'
	output_details = ''
	# CHEST AND CHEST RESULT
	if pos_idx == 0 and pos_idx2 == 0:
		final_output = "CHEST"
	# #HEAD AND HEAD RESULT
	if pos_idx == 1 and pos_idx2 == 1:
		final_output = "HEAD"
	# #MIX AND MIX RESULT
	if pos_idx == 2 and pos_idx2 == 2:
		final_output = "MIX"
	# #CHEST AND HEAD RESULT
	if pos_idx == 0 and pos_idx2 == 1:
		final_output = "CHEST HEAD"
		output_details ="Note: The application noticed that you changed your voice placement at the recording, meaning you are using two voice placements at a time (from chest to head voice). To have an accurate result try to record again."
	# #HEAD AND CHEST RESULT
	if pos_idx == 1 and pos_idx2 == 0:
		final_output ="HEAD AND CHEST"
		output_details = "Note: The application noticed that you changed your voice placement at the recording, meaning you are using two voice placements at a time (from head to chest voice). To have an accurate result try to record again."
	# #CHEST AND MIX RESULT
	if pos_idx == 0 and pos_idx2 == 2:
		final_output = "CHEST AND MIX"
		output_details = "Note: The application noticed that you used a chest and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
	# #MIX AND CHEST RESULT
	if pos_idx == 2 and pos_idx2 == 0:
		final_output = "MIX AND CHEST"
		output_details = "Note: The application noticed that you used a mixed and chest voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
	# #HEAD AND MIX RESULT
	if pos_idx == 1 and pos_idx2 == 2:
		final_output = "HEAD AND MIX"
		output_details = "Note: The application noticed that you used a head and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
	# #MIX AND HEAD RESULT
	if pos_idx == 2 and pos_idx2 == 1:
		final_output = "MIX AND HEAD"
		output_details = "Note: The application noticed that you used a mixed and head voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."


	pil_img = Image.open("static/temp_spect.png", mode='r') # reads the PIL image
	byte_arr = io.BytesIO()
	pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
	encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
	print(encoded_img)

	return jsonify({"final_output": final_output, "img": encoded_img, "output_details": output_details})


if __name__ =='__main__':
	app.run()
		