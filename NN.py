import tensorflow as tf
from tensorflow.keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
#import shap
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

'''
- https://www.tensorflow.org/install/pip#virtual-environment-install
- https://www.tensorflow.org/tutorials/keras/save_and_load
- https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
- https://www.yourdatateacher.com/2021/05/17/how-to-explain-neural-networks-using-shap/
- https://www.tensorflow.org/tutorials/keras/save_and_load

$ . venv/bin/activate

'''

def NeuralNetwork(data_train, data_test, label_train, label_test, n_features):

	model = create_model(n_features)
	print("MODEL COMPILED: " + str(time.time()))

	# Create a callback that saves the model's weights
	checkpoint_path = "trained_model/cp.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

	# Train model 
	#training = model.fit(x=data_train, y=label_train, batch_size=32, epochs=11, shuffle=True, verbose=0, validation_split=0.3, callbacks=[cp_callback])
	training = model.fit(x=data_train, y=label_train, batch_size=32, epochs=11, shuffle=True, verbose=0, validation_split=0.3)
	print("MODEL FITTED: " + str(time.time()))

	# Create a basic model instance
	#new_model = create_model(n_features)
	
	# Evaluate the untrained model
	#values = new_model.evaluate(data_test, label_test, verbose=2)
	#print("Untrained model, accuracy: " + str(values))
	
	# Loads the weights
	#new_model.load_weights(checkpoint_path)
	
	# Re-evaluate the model
	#values = new_model.evaluate(data_test, label_test, verbose=2)
	#print("Restored model, accuracy: " + str(values))

	# Validate model
	print(model.evaluate(data_test, label_test, verbose=2))
	print("MODEL EVALUATED: " + str(time.time()))


	'''
	#SHAP
	explainer = shap.KernelExplainer(model.predict,X_train)
	shap_values = explainer.shap_values(X_test,nsamples=100)
	shap.summary_plot(shap_values,X_test,feature_names=list(range(0,10)))

	# plot
	metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]	
	fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
		   
	## training	
	ax[0].set(title="Training")	
	ax11 = ax[0].twinx()	
	ax[0].plot(training.history['loss'], color='black')	
	ax[0].set_xlabel('Epochs')	
	ax[0].set_ylabel('Loss', color='black')	
	for metric in metrics:		
		ax11.plot(training.history[metric], label=metric)	
	ax11.set_ylabel("Score", color='steelblue')	
	ax11.legend()

	## validation	
	ax[1].set(title="Validation")	
	ax22 = ax[1].twinx()	
	ax[1].plot(training.history['val_loss'], color='black')	
	ax[1].set_xlabel('Epochs')	
	ax[1].set_ylabel('Loss', color='black')	
	for metric in metrics:		  
		ax22.plot(training.history['val_'+metric], label=metric)	
	ax22.set_ylabel("Score", color="steelblue")	
	#plt.show()

	''' 

def uploadNeuralNetwork_andPredict(n_features, flow):

	# Create a callback that saves the model's weights
	checkpoint_path = "trained_model/cp.ckpt"
	# Create a basic model instance
	new_model = create_model(n_features)
	# Loads the weights
	new_model.load_weights(checkpoint_path)
	# Predict
	print(new_model.predict(flow))

def uploadNeuralNetwork_andEvaluate(n_features, data_test, label_test):

	# Create a callback that saves the model's weights
	checkpoint_path = "trained_model/cp.ckpt"
	# Create a basic model instance
	new_model = create_model(n_features)
	# Loads the weights
	new_model.load_weights(checkpoint_path)
	# Re-evaluate the model
	values = new_model.evaluate(data_test, label_test, verbose=2)
	print("Restored model, accuracy: " + str(values))

def create_model(n_features):
	model = models.Sequential(name="NN", layers=[
		### hidden layer 1
		layers.Dense(name="h1", input_dim=n_features,
					 units=int(round((n_features+1)/2)), 
					 activation='relu'),
		layers.Dropout(name="drop1", rate=0.2),
		
		### hidden layer 2
		layers.Dense(name="h2", units=int(round((n_features+1)/4)), 
					 activation='relu'),
		layers.Dropout(name="drop2", rate=0.2),
		
		### layer output
		layers.Dense(name="output", units=1, activation='sigmoid')
	])
	model.summary()

	#visualize_nn(model, description=True, figsize=(10,8))

	# Compile the neural network
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1, Recall, Precision, tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()])
	
	return model

'''
Define Metrics
'''
def Recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def Precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def F1(y_true, y_pred):
	precision = Precision(y_true, y_pred)
	recall = Recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

'''
Extract info for each layer in a keras model.
'''
def utils_nn_config(model):
	lst_layers = []
	if "Sequential" in str(model): #-> Sequential doesn't show the input layer
		layer = model.layers[0]
		lst_layers.append({"name":"input", "in":int(layer.input.shape[-1]), "neurons":0, 
						   "out":int(layer.input.shape[-1]), "activation":None,
						   "params":0, "bias":0})
	for layer in model.layers:
		try:
			dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
						 "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
						 "params":layer.get_weights()[0], "bias":layer.get_weights()[1]}
		except:
			dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
						 "out":int(layer.output.shape[-1]), "activation":None,
						 "params":0, "bias":0}
		lst_layers.append(dic_layer)
	return lst_layers

'''
Plot the structure of a keras neural network
'''
def visualize_nn(model, description=False, figsize=(10,8)):
	## get layers info
	lst_layers = utils_nn_config(model)
	layer_sizes = [layer["out"] for layer in lst_layers]
	
	## fig setup
	fig = plt.figure(figsize=figsize)
	ax = fig.gca()
	ax.set(title=model.name)
	ax.axis('off')
	left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
	x_space = (right-left) / float(len(layer_sizes)-1)
	y_space = (top-bottom) / float(max(layer_sizes))
	p = 0.025
	
	## nodes
	for i,n in enumerate(layer_sizes):
		top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
		layer = lst_layers[i]
		color = "green" if i in [0, len(layer_sizes)-1] else "blue"
		color = "red" if (layer['neurons'] == 0) and (i > 0) else color
		
		### add description
		if (description is True):
			d = i if i == 0 else i-0.5
			if layer['activation'] is None:
				plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
			else:
				plt.text(x=left+d*x_space, y=top, fontsize=10, color=color, s=layer["name"].upper())
				plt.text(x=left+d*x_space, y=top-p, fontsize=10, color=color, s=layer['activation']+" (")
				plt.text(x=left+d*x_space, y=top-2*p, fontsize=10, color=color, s="Σ"+str(layer['in'])+"[X*w]+b")
				out = " Y"  if i == len(layer_sizes)-1 else " out"
				plt.text(x=left+d*x_space, y=top-3*p, fontsize=10, color=color, s=") = "+str(layer['neurons'])+out)
		
		### circles
		for m in range(n):
			color = "limegreen" if color == "green" else color
			circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
			ax.add_artist(circle)
			
			### add text
			if i == 0:
				plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
			elif i == len(layer_sizes)-1:
				plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
			else:
				plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
	
	## links
	for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
		layer = lst_layers[i+1]
		color = "green" if i == len(layer_sizes)-2 else "blue"
		color = "red" if layer['neurons'] == 0 else color
		layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
		layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
		for m in range(n_a):
			for o in range(n_b):
				line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], 
								  [layer_top_a-m*y_space, layer_top_b-o*y_space], 
								  c=color, alpha=0.5)
				if layer['activation'] is None:
					if o == m:
						ax.add_artist(line)
				else:
					ax.add_artist(line)
	plt.show()
