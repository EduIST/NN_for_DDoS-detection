import os
import time
from sklearn.model_selection import train_test_split
import preprocess_data
import preprocess_streamdata
import collect_data
import NN
from numpy import genfromtxt
import numpy as np
import pandas as pd
import time

''' ************* Helper *************

	Model: "NN"
	_________________________________________________________________
	 Layer (type)                Output Shape              Param #   
	=================================================================
	 h1 (Dense)                  (None, 12)                276       
	                                                                 
	 drop1 (Dropout)             (None, 12)                0         
	                                                                 
	 h2 (Dense)                  (None, 6)                 78        
	                                                                 
	 drop2 (Dropout)             (None, 6)                 0         
	                                                                 
	 output (Dense)              (None, 1)                 7         
	                                                                 
	=================================================================
	Total params: 361
	Trainable params: 361
	Non-trainable params: 0
	_________________________________________________________________
	516276/516276 - 390s - loss: 0.6858 - accuracy: 0.6736 - F1: 0.8017 - 390s/epoch - 756us/step
	Untrained model, accuracy: [0.6857917308807373, 0.6736334562301636, 0.8017211556434631]
	516276/516276 - 399s - loss: 0.0055 - accuracy: 1.0000 - F1: 1.0000 - 399s/epoch - 774us/step
	Restored model, accuracy: [0.005545021500438452, 0.9999770522117615, 0.9999828338623047]
	516276/516276 - 385s - loss: 0.0055 - accuracy: 1.0000 - F1: 1.0000 - 385s/epoch - 745us/step
	[0.005545021500438452, 0.9999770522117615, 0.9999828338623047]

    ************* Helper ************* '''

if __name__ == '__main__':

	print("STARTING: " + str(time.time()))

	''' Preprocess data '''

	df_train = preprocess_data.uploads_training_data()
	labels = df_train.loc[ : , 'is_malicious' ] #<class 'pandas.core.series.Series'>
	df_train = df_train.drop(columns=['is_malicious'])	
	print("DATA UPLOADED: " + str(time.time()))

	normalized_standardized, labels = preprocess_streamdata.data_normalization(df_train, labels)

	#np.savetxt("/Users/edubarros/Desktop/labels.csv", labels, delimiter=",")
	np.savetxt("/Users/edubarros/Desktop/normalized_standardized.csv", normalized_standardized, delimiter=",")

	print("DATA STANDARDIZED: " + str(time.time())) #<class 'numpy.ndarray'>

	''' Use preprocessed data 

	normalized_standardized = genfromtxt('/Users/edubarros/Desktop/datanormalized_standardized_final2.csv', delimiter=',')
	labels = pd.read_csv('/Users/edubarros/Desktop/labels.csv', header=None)
	labels = labels.iloc[:, 1]
	print("DATA NORMALIZED UPLOADED: " + str(time.time()))
	'''
	
	#TODO GUARDAR MODELO E USA-LO NUMA FUNÇÃO PROPRIA

	data_train, data_test, label_train, label_test = train_test_split(normalized_standardized, labels, test_size=0.33, random_state=42)
	n_features = len(normalized_standardized[0])
	print("DATA DIVIDED: " + str(time.time()))

	NN.NeuralNetwork(data_train, data_test, label_train, label_test, n_features)

	print("FINISHED (trained, tested and saved): " + str(time.time()))

	''' 
	## LIVE TRAFFIC ##
	actual_number_of_files = len(os.listdir("../Probe/"))

	while 1:
		
		if len(os.listdir("../Probe/")) > actual_number_of_files:

			flow_path = collect_data.generate_flows()

			df = preprocess_streamdata.generate_array(flow_path)

			normalized_standardized_np = preprocess_streamdata.data_normalization(df)
			#preprocess_streamdata.PCA_extraction(normalized_standardized_np)

			#Model Construction

			#Model Evaluation 

		actual_number_of_files = len(os.listdir("../Probe/"))
		time.sleep(0.5)

exit(1)
	''' 

