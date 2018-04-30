
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
import random
import json
import os
import database as db
import matplotlib.pyplot as plt 
from time import time
import sys
from operator import itemgetter, attrgetter

comparison_datasets=[]

def create_comparison_set(d_size,nb_measures):
	for trk_comp in range(3,12):
		comparison_datasets.append(create_dataset(db.request_track(trk_comp),dataset_size=d_size,nb_measures=nb_measures))

#tf.enable_eager_execution()

def create_dataset_tf(track_array_json, gateway_ref, **kwargs):
	'''
	@summary: Creates a random sample dataset from track data
	@param track: the input track array
	@param gateway_ref: Reference gateway array, the output will be according to this order
	@kwargs dataset_size: how many points the dataset will contain per track. 
							The total dataset size is dataset_size * nb_tracks
	@kwargs nb_measures: over how many random points a dataset item is generated
	@result: (dataset in tf format, track id as labels)
	'''
	#Set default values
	dataset_size = 20 # per track!
	nb_measures = 10
	train_test = 0.8

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']
	if 'train_test' in kwargs:
		train_test = kwargs['train_test']

	if train_test > 1 or train_test < 0:
		return "ERROR: Impossible train-test ratio"

	compilation_train = []
	compilation_test = []

	for trk_json in track_array_json:
		#for every track we need to have some points which are only training and some distinct other points for testing
		track = json.loads(trk_json.decode('utf-8'))
		track_train = track[:int(train_test*len(track))]
		track_test = track[int(train_test*len(track)):]

		trk_dict_train = create_dataset(track_train,dataset_size=dataset_size,nb_measures=nb_measures,pre_treated=True)
		trk_dict_test = create_dataset(track_test,dataset_size=dataset_size,nb_measures=nb_measures,pre_treated=True)

		#attribute the gateway features to the correct place in the gateway reference array
		#for every point in the dataset
		#training set

		for p in trk_dict_train:
			tensor = []
			for eui in gateway_ref:
				if eui in p['Gateways']:
					tensor.append(p['Gateways'][eui])
				else:
					tensor.append([0,0,0])
			compilation_train.append({"Data":tensor,"Label":p['Track']})

		for p in trk_dict_test:
			tensor = []
			for eui in gateway_ref:
				if eui in p['Gateways']:
					tensor.append(p['Gateways'][eui])
				else:
					tensor.append([0,0,0])
			compilation_test.append({"Data":tensor,"Label":p['Track']})

	#create random order
	random.shuffle(compilation_test)
	random.shuffle(compilation_train)

	#create output arrays wich have the same order
	data_train = []
	labels_train = []
	data_test = []
	labels_test = []

	for point in compilation_train:
		data_train.append(point['Data'])
		labels_train.append(point['Label'])
	for point in compilation_test:
		data_test.append(point['Data'])
		labels_test.append(point['Label'])

	return ((data_train, labels_train),(data_test,labels_test))


def jaccard_classifier(input_track, **kwargs):
	'''
	@summary: Track classification engine based on jaccard similarity
	@param input_track: the input track to classify
	@kwargs d_size: how many points the comparison dataset will contain per track. 
	@kwrgs nb_iter: on how many examples the mean is calculated
	@kwargs nb_measures: over how many random points the comparison dataset item is generated
	@result: tuple (most likely track, similarity index)
	'''
	d_size = 100
	nb_iter = 1
	nb_measures = 30
	if 'd_size' in kwargs:
		d_size = kwargs['d_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']
	if 'nb_iter' in kwargs:
		nb_iter = kwargs['nb_iter']

	similarity = []

	for i in range(nb_iter):
		for trk_comp in range(3,12):
			comparison_dataset = comparison_datasets[trk_comp-3]
			c = 0
			for p1 in input_track:
				for p2 in comparison_dataset:
					c += jaccard_index(p1,p2)
			mean = c / (d_size**2)
			if i == 0:
				similarity.append(mean)
			else:
				similarity[trk_comp-3] += mean
	similarity_list = []
	for i,sim in enumerate(similarity):
		similarity_list.append((i+3,sim/nb_iter))
	similarity_list.sort(key=itemgetter(1),reverse=True)
	return(similarity_list[0])
				
def jaccard_classifier_best(input_track, **kwargs):
	'''
	@summary: Track classification engine based on jaccard similarity
	@param input_track: the input track to classify
	@kwargs d_size: how many points the comparison dataset will contain per track. 
	@kwrgs nb_iter: on how many examples the mean is calculated
	@kwargs nb_measures: over how many random points the comparison dataset item is generated
	@result: tuple (most likely track, similarity index)
	'''
	d_size = 100
	nb_iter = 1
	nb_measures = 30
	if 'd_size' in kwargs:
		d_size = kwargs['d_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']
	if 'nb_iter' in kwargs:
		nb_iter = kwargs['nb_iter']

	for i in range(nb_iter):
		similarity = []
		print(".",end="")
		sys.stdout.flush() #display point immediately
		for trk_comp in range(3,12):
			comparison_dataset = create_dataset(db.request_track(trk_comp),dataset_size=d_size,nb_measures=nb_measures)
			c = 0
			for p1 in input_track:
				for p2 in comparison_dataset:
					c += jaccard_index(p1,p2)
			mean = c / (d_size**2)
			similarity.append((trk_comp,mean))
		similarity.sort(key=itemgetter(1),reverse=True)
		print(similarity[0])
		print(similarity[1])
	#if best value not reached...
	return(similarity[0])

def create_dataset(track_json, **kwargs):
	'''
	@summary: Creates a random sample dataset from track data
	@param track: the input track
	@kwargs dataset_size: how many points the dataset will contain
	@kwargs nb_measures: over how many random points a dataset item is generated
	@result: dataset
	'''
	if 'pre_treated' not in kwargs:
		track = json.loads(track_json.decode('utf-8'))
	else:
		track = track_json
	#Set default values
	dataset_size = 100
	nb_measures = 10

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']

	if nb_measures > len(track):
		nb_measures = len(track)
		print("WARNING: nb_measures reduced to "+str(nb_measures))

	dataset = []

	for i in range (0,dataset_size):
		random.shuffle(track)
		#calculate ESP, mean, var and make gtw list with all ESP
		t = {}
		freq_count = 0.0
		for k in range (0,nb_measures):				
			for j, eui in enumerate(track[k]['gateway_id']):
				if eui not in t:
					t.update({eui:(track[k]['gateway_esp'][j],1)})
					freq_count += 1.0
				else:
					t.update({eui:(track[k]['gateway_esp'][j]+t[eui][0],t[eui][1]+1)})
					freq_count += 1.0
		
		gtw_info = {} #key: eui, values array: [mean, sigma, frequency]

		#calculate the means and frequency
		for u,v in t.items():
			gtw_info.update({u:[t[u][0]/t[u][1],0,0]})

		#calculate sigma
		q = {}
		for k in range(0,nb_measures):
			for j, eui in enumerate(track[k]['gateway_id']):
				if eui not in q:
					q.update({eui:(np.power(track[k]['gateway_esp'][j]-gtw_info[eui][0],2),1)})
				else:
					q.update({eui:(np.power(track[k]['gateway_esp'][j]-gtw_info[eui][0],2)+q[eui][0],q[eui][1]+1)})

		for u,v in q.items():
			gtw_info.update({u:[gtw_info[u][0],np.sqrt(q[u][0]/q[u][1]),t[u][1]/freq_count]})

		#create the dataset
		dataset.append({'Track':track[k]['track_ID']-3,'Position':(track[k]['gps_lat'],track[k]['gps_lon']),'Gateways':gtw_info})

	return dataset

def jaccard_index(p1_raw,p2_raw):
	'''
	@summary: calculates the jaccard index from two datapoints
	@param p1: datapoint 1 python dict {'gtw':(mean ESP, ESP var, frequency)}
	@param p2: datapoint 2 (order doesn't matter)
	@result: jaccard index between 0 and 1, float
	'''
	p1 = p1_raw['Gateways']
	p2 = p2_raw['Gateways']
	intersection_count = 0
	for gtw, data in p1.items():
		if gtw in p2:
			intersection_count += 1
	return intersection_count / (len(p1)+len(p2)-intersection_count)

def jaccard_index_weighted(p1_raw,p2_raw):
	'''
	@summary: calculates the weighted jaccard index from two datapoints
	@param p1: datapoint 1 python dict {'gtw':(mean ESP, ESP var, frequency)}
	@param p2: datapoint 2 (order doesn't matter)
	@result: jaccard index between 0 and 1, float
	'''
	MEAN = 0
	SIGMA = 1
	FREQ = 2

	p1 = p1_raw['Gateways']
	p2 = p2_raw['Gateways']

	intersection_count = 0
	for gtw, data in p1.items():
		if gtw in p2:
			intersection_count += data[FREQ]
	for gtw, data in p2.items():
		if gtw in p1:
			intersection_count += data[FREQ]

	#norming the result
	intersection_count = intersection_count * (len(p1)+len(p2))/4

	return intersection_count / (len(p1)+len(p2)-intersection_count)

def get_gateways(track_array):
	'''
	@summary: creates an array of all the gateways seen by an array of tracks. useful for discovering new gateways which are not
				in the list or use office gateways for fingerprinting as well
	@param track_array: array of json tracks
	@result: array with gateway EUIs (string)
	'''
	gtws = []

	for trk_json in track_array:
		trk = json.loads(trk_json.decode('utf-8'))
		for i in range(len(trk)):
			for gtw in trk[i]['gateway_id']:
				if gtw not in gtws:
					gtws.append(gtw)
	return gtws

def neuronal_classification(training, testing, nb_tracks, nb_gtw, batch, epochs, neurons1, dropout1):

	#Remark: For a faster processing with GPU, this should be done with tf datasets
	'''
	training_set = tf.data.Dataset.from_tensor_slices(training)
	testing_set = tf.data.Dataset.from_tensor_slices(testing)
	
	#create random order
	training_set = training_set.shuffle(buffer_size=1000)
	testing_set = testing_set.shuffle(buffer_size=1000)
	'''

	training_set = (np.array(training[0]),np.array(training[1]))
	testing_set = (np.array(testing[0]),np.array(testing[1]))

	# Convert labels to categorical one-hot encoding
	one_hot_labels_train = tf.keras.utils.to_categorical(training_set[1], num_classes=nb_tracks)
	one_hot_labels_test = tf.keras.utils.to_categorical(testing_set[1], num_classes=nb_tracks)

	#Creating the NN model with Keras
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(neurons1, activation="relu", input_shape=(nb_gtw, 3,)),
		tf.keras.layers.Dropout(dropout1),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(nb_tracks, activation="softmax") #output layer
		])
	#print the summary of the model
	#model.summary()

	model.compile(
		#for a multi-class classification problem
		optimizer = "rmsprop",
		loss = "categorical_crossentropy",
		metrics = ["accuracy"]
	)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time()))

	results = model.fit(
		training_set[0], one_hot_labels_train,
		epochs=epochs,
		batch_size=batch,
		validation_data=(testing_set[0],one_hot_labels_test),
		callbacks=[tensorboard]
		)	

	return model