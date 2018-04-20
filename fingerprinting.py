
import numpy as np
import tensorflow as tf
import random
import json


def create_dataset(track_json, **kwargs):
	'''
	@summary: calculates the absolute lat-lon coordinates from the reference coordinates
	@param track: the input track
	@kwargs dataset_size: how many points the dataset will contain
	@kwargs nb_measures: over how many random points a dataset item is generated
	@result: dataset
	'''
	track = json.loads(track_json.decode('utf-8'))
	#Set default values
	dataset_size = 100
	nb_measures = 10

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']

	dataset = []

	for i in range (0,dataset_size):
		random.shuffle(track)

		#calculate ESP, mean, var and make gtw list with all ESP
		for k in range (0,nb_measures):

			dataset.append({'Lat':track[k]['gps_lat'],'Lon':track[k]['gps_lon']})
			for entry in track[k]['gateway_id']:
				print(entry)


	'''
	dataset should contain:
	Received gateways with frequency, normed to 1


	'''
	#create random order


