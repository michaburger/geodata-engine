"""
Author: Micha Burger, 24.07.2018
https://micha-burger.ch
LoRaWAN Localization algorithm used for Master Thesis 

This file is containing all the Machine Learning methods
for classification and research of the best class. Also
the old and now unused methods like Neuronal Networks
with Tensorflow are included.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.eager as tfe 
import random
import json
import os
import database as db
import matplotlib.pyplot as plt 
import geopy.distance
from time import time
import sys
from operator import itemgetter, attrgetter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
import warnings

comparison_datasets=[]

def get_gateways(track_array):
	"""Get a list of all the gateway EUIs seen by this track.

    Args:
        track_array (array): 	Array containing arrays of python dict
        						with each track information

    Returns:
        array of string: array of all gateway EUIs
    """
	gtws = []

	for trk in track_array:
		for i in range(len(trk)):
			for gtw in trk[i]['gateway_id']:
				if gtw not in gtws:
					gtws.append(gtw)
	return gtws

def gateway_list_track(track):
	"""Used to feed a single track into a track array used by get_gateways

    Args:
        track (dict): 	python dict containing one single track

    Returns:
        array of string: array of all gateway EUIs
    """
	trk_array = []
	trk_array.append(track)
	return get_gateways(trk_array)

def hexcol(col):
	"""Reformat color by removing '0x' from the string

    Args:
        col (string): 	python dict containing one single track

    Returns:
        string: exactly 2 digits in a string representing RGB
    """
	hexstr = str(hex(col).replace('0x',''))
	if col <16:
		return '0'+ hexstr
	else:
		return hexstr


def random_color():
	"""Generate a random color

    Args:
        void

    Returns:
        string: random RGB color in the format #rrggbb
    """
	return '#'+hexcol(random.randint(0,255))+hexcol(random.randint(0,255))+hexcol(random.randint(0,255))

#apply PCA and return pandas table. When nb_dim > 0, 
#show the first n dimensions on a plot.
def apply_pca(d,nb_clusters,nb_dim):
	"""Apply PCA and return table in pandas format. Plot the first nb_dim
	dimensions on a graph for visualization.

	PCA was used to have a closer look at the nature of the dataset and the
	geoclusters if they could be seperated somehow. It is not used any more
	for the last version of the final algorithm.

    Args:
        d (tuple): 	(dataset,labels) in arrays of float / int


    Returns:
        pandas df: pandas dataframe with the result of the PCA
    """
	dataset = np.array(d[0])
	labels = np.array(d[1])

	#flatten matrix
	nsamples, nx, ny = dataset.shape
	d2_dataset = dataset.reshape((nsamples,nx*ny))

	#normalize data to apply PCA
	data = StandardScaler().fit_transform(d2_dataset)

	nb_components = 30
	pca = PCA(n_components=nb_components)
	principalComponents = pca.fit_transform(data)
	
	labelFrame = pd.DataFrame(data=labels,columns=['label'])
	principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC{}'.format(i) for i in range (1,nb_components+1)])
	finalDf = pd.concat([principalDf, labelFrame], axis = 1)
	
	if nb_dim > 0:
		fig = plt.figure()
		fig.subplots_adjust(left=0.05,bottom=0.08,right=0.97,top=0.95,wspace=0.15,hspace=0.15)
		plt.suptitle('PCA analysis on feature space')

		targets = [i for i in range (1,nb_clusters+1)]
		colors = [random_color() for i in range (1,nb_clusters+1)]

		count=0
		for row in range(1,nb_dim+1):
			for col in range(1,nb_dim+1):
				count += 1
				ax = fig.add_subplot(nb_dim,nb_dim,count)

				#trick to show axis label only at the left and bottom
				if col == 1:
					ax.set_ylabel('PC{}'.format(row))
				if row == nb_dim:
					ax.set_xlabel('PC{}'.format(col))
				for target, color in zip(targets,colors):
					indicesToKeep = finalDf['label'] == target
					ax.scatter(finalDf.loc[indicesToKeep, 'PC{}'.format(col)]
							, finalDf.loc[indicesToKeep, 'PC{}'.format(row)]
							, c = color, s = 50)
				if row == 1 and col == nb_dim:
					ax.legend(targets)
					ax.set_zorder(1)
		plt.show()
	
	#print(pca.explained_variance_ratio_)
	return finalDf

def create_dataset_pandas(track_array, gateway_ref, **kwargs):
	"""Create a dataset from a track array (usually clusters) and bring it 
	into pandas dataframe feature-space format. 

	The function supports a test-train ratio. This creates two completely 
	different datasets by splitting the points. The points of a track must already
	be in a random order for this.

	A track is normally used here for storing one entire geo-cluster. Using the 
	list of reference gateways, we determine how many times the gateway has been
	seen from points of this cluster. This gateway frequency is stored together with 
	the mean and variance of ESP for each gateway. If a gateway has not been seen, 
	we use 0 as a placeholder. This creates a matrix of N x 3 values, N being the
	number of gateways in gateway_ref.

	Multiple datasets can be created for every cluster, normally there are more points
	per cluster than nb_measures. This allows to create different datasets with slight 
	random variations to represent the entire palette of possibilities inside a cluster.

	The function returns a pandas dataframe containing the feature spaces together with 
	its coordinates and cluster ID (Label). rLat,rLon represent the coordinates of a point
	randomly chosen inside the cluster. This is used for generating a map with the dataset 
	without ending up with the cluster data all on the same point. cLat,cLon represent
	the center coordinates of the corresponding cluster


    Args:
        track_array (string):		Array containing the clusters as python dict
        gateway_ref (array):		Array containing all the reference gateways as string
        kwargs dataset_size (int):	Number of feature spaces to generate per cluster (track)
        kwargs nb_measures (int):	Number of points per feature space
        kwargs train_test (float):	Train-Test ratio between 0 and 1
        kwargs offset (int):		If the track IDs have an offset, e.g. starting with 3
        							like they did for the trilateration measures.


    Returns:
        tuple (pandas df, pandas df): two pandas datasets with size according to test-train ratio
    """
	
	#Set default values
	dataset_size = 20 # per track!
	nb_measures = 10
	train_test = 0.5
	offset = 0

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']
	if 'train_test' in kwargs:
		train_test = kwargs['train_test']
	if 'offset' in kwargs:
		offset = kwargs['offset']

	if train_test > 1 or train_test < 0:
		print("ERROR: Impossible train-test ratio")
		return "ERROR: Impossible train-test ratio"

	compilation_train = []
	compilation_test = []

	df_train = pd.DataFrame()
	df_test = pd.DataFrame()

	for track in track_array:
		#for every track we need to have some points which are only training and some distinct other points for testing
		track_train = track[:int(train_test*len(track))]
		track_test = track[int(train_test*len(track)):]

		trk_dict_train = create_dataset(track_train,dataset_size=dataset_size,nb_measures=nb_measures,offset=offset)
		trk_dict_test = create_dataset(track_test,dataset_size=dataset_size,nb_measures=nb_measures,offset=offset)

		#attribute the gateway features to the correct place in the gateway reference array
		#for every point in the dataset
		#training set
		for p in trk_dict_train:
			tensor = []
			for eui in gateway_ref:
				if eui in p['Gateways']:
					tensor.append(-1*p['Gateways'][eui][0])
					tensor.append(40*p['Gateways'][eui][1])
					tensor.append(200*p['Gateways'][eui][2])
				else:
					for i in range(3):
						tensor.append(0)
			tensor_pd = pd.DataFrame(data=[tensor], columns=['C{}'.format(i) for i in range(1,len(tensor)+1)])
			info_pd = pd.DataFrame(data=[[p['Track'],p['Position'][0],p['Position'][1],p['Center'][0],p['Center'][1]]],columns=['Label1','rLat','rLon','cLat','cLon'])
			df_train= pd.concat([df_train,pd.concat([tensor_pd,info_pd],axis=1)])
		#create random order
		df_train = df_train.sample(frac=1).reset_index(drop=True)

		if train_test != 1:
			for p in trk_dict_test:
				tensor = []
				for eui in gateway_ref:
					if eui in p['Gateways']:
						tensor.append(-1*p['Gateways'][eui][0])
						tensor.append(40*p['Gateways'][eui][1])
						tensor.append(200*p['Gateways'][eui][2])
					else:
						for i in range(3):
							tensor.append(0)
				tensor_pd = pd.DataFrame(data=[tensor], columns=['C{}'.format(i) for i in range(1,len(tensor)+1)])
				info_pd = pd.DataFrame(data=[[p['Track'],p['Position'][0],p['Position'][1],p['Center'][0],p['Center'][1]]],columns=['Label1','rLat','rLon','cLat','cLon'])
				df_test= pd.concat([df_test,pd.concat([tensor_pd,info_pd],axis=1)])
			#create random order
			df_test = df_test.sample(frac=1).reset_index(drop=True)

	return df_train,df_test

def create_dataset_tf(track_array, gateway_ref, **kwargs):
	"""Create a dataset from a track array (usually clusters) and bring it 
	into pandas dataframe feature-space format. 

	Difference to create_dataset: This function creates a tuple with data and 
	labels, ready to be used by a Tensorflow NN.

	The function supports a test-train ratio. This creates two completely 
	different datasets by splitting the points. The points of a track must already
	be in a random order for this.

	A track is normally used here for storing one entire geo-cluster. Using the 
	list of reference gateways, we determine how many times the gateway has been
	seen from points of this cluster. This gateway frequency is stored together with 
	the mean and variance of ESP for each gateway. If a gateway has not been seen, 
	we use 0 as a placeholder. This creates a matrix of N x 3 values, N being the
	number of gateways in gateway_ref.

	Multiple datasets can be created for every cluster, normally there are more points
	per cluster than nb_measures. This allows to create different datasets with slight 
	random variations to represent the entire palette of possibilities inside a cluster.

	The function returns a pandas dataframe containing the feature spaces together with 
	its coordinates and cluster ID (Label). rLat,rLon represent the coordinates of a point
	randomly chosen inside the cluster. This is used for generating a map with the dataset 
	without ending up with the cluster data all on the same point. cLat,cLon represent
	the center coordinates of the corresponding cluster


    Args:
        track_array (string):		Array containing the clusters as python dict
        gateway_ref (array):		Array containing all the reference gateways as string
        kwargs dataset_size (int):	Number of feature spaces to generate per cluster (track)
        kwargs nb_measures (int):	Number of points per feature space
        kwargs train_test (float):	Train-Test ratio between 0 and 1
        kwargs offset (int):		If the track IDs have an offset, e.g. starting with 3
        							like they did for the trilateration measures.


    Returns:
        tuple (tuple(data,label),tuple(data,label)): 	Two pandas datasets with size according to test-train ratio
        												Already random order and ready to be used with TensorFlow
    """
    
	#Set default values
	dataset_size = 20 # per track!
	nb_measures = 10
	train_test = 0.5
	offset = 3

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']
	if 'train_test' in kwargs:
		train_test = kwargs['train_test']
	if 'offset' in kwargs:
		offset = kwargs['offset']

	if train_test > 1 or train_test < 0:
		return "ERROR: Impossible train-test ratio"

	compilation_train = []
	compilation_test = []

	for track in track_array:
		#for every track we need to have some points which are only training and some distinct other points for testing
		track_train = track[:int(train_test*len(track))]
		track_test = track[int(train_test*len(track)):]

		trk_dict_train = create_dataset(track_train,dataset_size=dataset_size,nb_measures=nb_measures,offset=offset)
		trk_dict_test = create_dataset(track_test,dataset_size=dataset_size,nb_measures=nb_measures,offset=offset)

		#attribute the gateway features to the correct place in the gateway reference array
		#for every point in the dataset
		#training set
		for p in trk_dict_train:
			tensor = []
			for eui in gateway_ref:
				if eui in p['Gateways']:
					arr = [p['Gateways'][eui][0],p['Gateways'][eui][1],p['Gateways'][eui][2]]
					tensor.append(arr)
				else:
					tensor.append([0,0,0])
			compilation_train.append({"Data":tensor,"Label":p['Track'],"Position":p['Position']})

		for p in trk_dict_test:
			tensor = []
			for eui in gateway_ref:
				if eui in p['Gateways']:
					arr = [p['Gateways'][eui][0],p['Gateways'][eui][1],p['Gateways'][eui][2]]
					tensor.append(arr)
				else:
					tensor.append([0,0,0])
			compilation_test.append({"Data":tensor,"Label":p['Track'],"Position":p['Position']})

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

def cosine_similarity(v1,v2):
	"""Calculates the cosine similarity between two feature space vectors

    Args:
        v1 (array):		feature space vector, len(gtw)*3 entries
        v2 (array):		feature space vector, len(gtw)*3 entries

    Returns:
        float:			Cosine similarity. Higher value for similar points
    """
	if len(v1) != len(v2):
		print("ERROR: not same vector length in cosine similarity function!")
		return 0.0

	dist = (2 - spatial.distance.cosine(v1,v2))/2.0
	return dist

def euclidean_similarity(v1,v2):
	"""Calculates the euclidean similarity between two feature space vectors

    Args:
        v1 (array):		feature space vector, len(gtw)*3 entries
        v2 (array):		feature space vector, len(gtw)*3 entries

    Returns:
        float:			Euclidean similarity. Higher value for similar points
    """
	if len(v1) != len(v2):
		print("ERROR: not same vector length in euclidean similarity function!")
		return 0.0
	dist = (1/spatial.distance.euclidean(v1,v2))*len(v1)*5
	return dist

def correlation_similarity(v1,v2):
	"""Calculates the correlation similarity between two feature space vectors

    Args:
        v1 (array):		feature space vector, len(gtw)*3 entries
        v2 (array):		feature space vector, len(gtw)*3 entries

    Returns:
        float:			Correlation similarity. Higher value for similar points
    """
	if len(v1) != len(v2):
		print("ERROR: not same vector length in correlation similarity function!")
		return 0.0
	dist = (2- spatial.distance.correlation(v1,v2))/2.0
	return dist

def manhattan_similarity(v1,v2):
	"""Calculates the manhattan similarity between two feature space vectors

    Args:
        v1 (array):		feature space vector, len(gtw)*3 entries
        v2 (array):		feature space vector, len(gtw)*3 entries

    Returns:
        float:			Manhattan similarity. Higher value for similar points
    """
	if len(v1) != len(v2):
		print("ERROR: not same vector length in manhattan similarity function!")
		return 0.0
	dist = (1/spatial.distance.cityblock(v1,v2))*len(v1)*len(v2)/2
	return dist


def similarity_classifier_knn(db, test, nncl, **kwargs):
	"""Compares the test point with the database and returns a pre-defined
	number of best matches, according to the chosen similarity metrics.



    Args:
        db (pandas df):				Reference database to use for the search
		test (pandas df):			Test cluster
		nncl (int):					Number of total clusters
		kwargs metrics (string):	'Label1' to use probability or 
									'Label2' to use similarity to find best matches
		kwargs flatten (float):		n-root to take for all the values in the list of
									best matches. Makes the difference between the first
									guess and the last guess smaller
		kwargs first_values (int):	How many of the best guesses to keep in the list to return
		kwargs function (string):	Similarity function. 'cosine', 'euclidean', 'correlation'
									and 'manhattan' are accepted.			

    Returns:
        pandas dataframe:			List of the N most probable locations, with coordinates
    """
	metrics = kwargs['metrics'] if 'metrics' in kwargs else 'Label1'
	flatten = kwargs['flatten'] if 'flatten' in kwargs else 1.0
	first_values = kwargs['first_values'] if 'first_values' in kwargs else 10
	function = kwargs['function'] if 'function' in kwargs else 'cosine'

	db_sparse = db.drop(columns=['cLat','cLon','rLat','rLon'])#,'Label1' if metrics == 'Label2' else 'Label2'])
	test = test.drop(['cLat','cLon','rLat','rLon','Label1'])#,'Label2'])
	v_test = test.tolist()
	#calculate the similarity for every point in the comparison database
	sim = [[] for i in range(0,nncl+1)]

	for i, row in db_sparse.iterrows():
		label = int(row[metrics])
		row = row.drop(metrics)
		#create vector
		v1 = row.tolist()
		if(function == 'cosine'):
			sim[label].append(cosine_similarity(v1,v_test))
		elif(function == 'euclidean'):
			sim[label].append(euclidean_similarity(v1,v_test))
		elif(function == 'correlation'):
			sim[label].append(correlation_similarity(v1,v_test))
		elif(function == 'manhattan'):
			sim[label].append(manhattan_similarity(v1,v_test))
		else:
			print("ERROR: Please specify a correct classification function. Accepted functions: 'cosine', 'euclidean', 'manhattan', correlation'")

	#list the most probable clusters with similarity index
	cluster_sim = []
	for c in sim:
		if len(c) > 0:
			cluster_sim.append((np.mean(c),np.var(c)))
		else:
			cluster_sim.append((0,0))

	cluster_stat = pd.DataFrame(data=cluster_sim,columns=['Mean Similarity','Variance'])
	cluster_stat.sort_values(by='Mean Similarity',ascending=False,inplace=True)

	#ignore warnings in the next part
	warnings.filterwarnings("ignore")
	#rescale and norm for the first 10 clusters --> probabilities
	head = cluster_stat.head(first_values+1)
	head['rescaled'] = ((head.loc[:,'Mean Similarity'] - head.loc[:,'Mean Similarity'].min()) / (head.loc[:,'Mean Similarity'].max()-head.loc[:,'Mean Similarity'].min()))**(1/flatten)
	head['Probability'] = head.loc[:,'rescaled'] / head.loc[:,'rescaled'].sum()
	head.drop(columns='rescaled',inplace=True)
	head.drop(head.index[first_values],inplace=True) #drop last index with probability 0

	#print(head)

	#add coordinates of center
	center_coords = []
	for index in head.index.values:
		c = db.loc[np.isclose(db[metrics],index)].reset_index(drop=True).loc[1,['cLat','cLon']]
		#print("Cluster {} - {} {}".format(index,c.loc['cLat'],c.loc['cLon']))
		center_coords.append((c.loc['cLat'],c.loc['cLon']))

	head = head.reset_index()
	coords_pd = pd.DataFrame(data=center_coords,columns=['Lat','Lon'])
	best_with_coords = pd.concat([head,coords_pd],axis=1)
	best_with_coords.rename(index=str, columns={"index":"Cluster ID"},inplace=True)

	return best_with_coords

#Was used for the first ML / NN comparison
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

def create_dataset(track, **kwargs):
	'''
	@summary: Creates a random sample dataset from track data. Sets the mean position of every cluster as position.
	@param track: the input track
	@kwargs dataset_size: how many points the dataset will contain
	@kwargs nb_measures: over how many random points a dataset item is generated
	@result: dataset
	'''

	#Set default values
	dataset_size = 100
	nb_measures = 10

	if 'dataset_size' in kwargs:
		dataset_size = kwargs['dataset_size']
	if 'nb_measures' in kwargs:
		nb_measures = kwargs['nb_measures']

	#if the tracks have an offset (start with track 3 for exemple, default)
	offset = 3
	if 'offset' in kwargs:
		offset = kwargs['offset']

	if len(track) == 0:
		return []
	if nb_measures > len(track):
		nb_measures = len(track)
		print("WARNING: nb_measures reduced to "+str(nb_measures))

	dataset = []
	latarr = []
	lonarr = []

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
			latarr.append(track[k]['gps_lat'])
			lonarr.append(track[k]['gps_lon'])
		
		mean_lat = sum(latarr) / len(latarr)
		mean_lon = sum(lonarr) / len(lonarr)

		#for display reasons it can be better to display the point at a random position inside the cluster, 
		#representing the set of all points that are represented by this cluster.
		random_lat = track[0]['gps_lat']
		random_lon = track[0]['gps_lon']

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
		

		#create dataset. ok always taking the first point of a track, in the end with n=100 all points should be represented
		#TODO: correct position / size of cluster instead of random point each time.
		dataset.append({'Track':track[0]['track_ID']-offset,'Position':(random_lat,random_lon),'Center':(mean_lat,mean_lon),'Gateways':gtw_info})

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

def neuronal_classification_clusters(clusters_training, clusters_validation, nb_clusters):
	"""Function was used for the NN classification part with tensorflow on the entire
	database. For this purpose, the database has to be split into training and validation
	clusters.

	The function handles everything from creating the model, fitting the model and even 
	evaluating the results. As NN wasn't chosen as a final classifier, I didn't go further
	(like returning the list or so)

    Args:
        clusters_training (pandas df):		clusters for training
        clusters_validation (pandas df):	clusters for validation
        nb_clusters (int):					Total number of clusters

    Returns:
        void
    """

	#prepare data
	training_labels = clusters_training.loc[:,['Label1']]
	validation_labels = clusters_validation.loc[:,['Label1']]
	training_data = clusters_training.drop(columns=['Label1','cLat','cLon','rLat','rLon','Label2'])
	validation_data = clusters_validation.drop(columns=['Label1','cLat','cLon','rLat','rLon','Label2'])

	#Enable GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)

	training_set = (np.array(training_data),np.array(training_labels))
	validation_set = (np.array(validation_data),np.array(validation_labels))

	# Convert labels to categorical one-hot encoding
	one_hot_labels_train = tf.keras.utils.to_categorical(training_set[1], num_classes=nb_clusters)
	one_hot_labels_test = tf.keras.utils.to_categorical(validation_set[1], num_classes=nb_clusters)

	layers = 1
	neurons1 = 16
	neurons_n = 32
	dropout = 0.2

	#Creating the NN model with Keras
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(neurons1, activation="relu", input_shape=(training_set[0].shape[1],)))
	model.add(tf.keras.layers.Dropout(dropout))
	for i in range(layers):
		model.add(tf.keras.layers.Dense(neurons_n, activation="relu"))
		model.add(tf.keras.layers.Dropout(dropout))
	model.add(tf.keras.layers.Dense(neurons_n*2, activation="relu"))
	model.add(tf.keras.layers.Dropout(dropout))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(nb_clusters, activation="softmax"))

	model.compile(
		#for a multi-class classification problem
		optimizer = "rmsprop",
		loss = "categorical_crossentropy",
		#loss = tf.nn.softmax_cross_entropy_with_logits_v2,
		metrics = ["accuracy"]
	)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/lay-{}-n1-{}-nn-{}-drp{}-{}".format(layers+1,neurons1,neurons_n,dropout,time()))

	results = model.fit(
		training_set[0], one_hot_labels_train,
		epochs=128,
		batch_size=8,
		validation_data=(validation_set[0],one_hot_labels_test),
		callbacks=[tensorboard],
		verbose=2
		)

	prediction = model.predict(validation_set[0][0:50])
	#sort and show 10 most probable clusters with probability

	LIST_HEAD_MEAN = 3
	dist_errors = []
	for i, cl_prb in enumerate(prediction):
		c = pd.DataFrame(data=cl_prb,columns=['P'])
		c.sort_values(by=['P'], inplace = True, ascending=False)
		print("Probabilities of the model:")
		print(c.head(20))
		real_cluster_nb = int(validation_set[1][i][0])
		print("Real label: {}".format(real_cluster_nb))
		#mean coords of the cluster
		ref_coords = (float(clusters_validation.loc[real_cluster_nb,['cLat']]), float(clusters_validation.loc[real_cluster_nb,['cLon']]))
		index_list = c.index.tolist()
		lat = []
		lon = []
		dist_m = []
		for index in index_list[:LIST_HEAD_MEAN]:
			lat.append(float(clusters_validation.loc[index,['cLat']]))
			lon.append(float(clusters_validation.loc[index,['cLon']]))
			dist_m.append(geopy.distance.vincenty(ref_coords,(float(clusters_validation.loc[index,['cLat']]),float(clusters_validation.loc[index,['cLon']]))).km*1000)
		print(dist_m)
		print(np.mean(dist_m))
		dist_errors.append(geopy.distance.vincenty(ref_coords,(np.mean(lat),np.mean(lon))).km*1000)
	print(dist_errors)
	print(np.mean(dist_errors))

	#print(results.history["acc"])
	#print(results.history["val_acc"])


def neuronal_classification(training, testing, nb_tracks, nb_gtw, batch, epochs, neurons1, dropout1, n_dataset, n_meas, activation, layers):
	"""Function used to test Neuronal Classification to distinguish between 9 reference points

	The function handles everything from creating the model, fitting the model and even 
	evaluating the results. Returns the mean accuracy over the last 10 epochs. Used for
	automated testing of the best parameters.

    Args:
        training (tuple):		tuple with feature space data and labels
        testing (tuple):		tuple with feature space data and labels
        nb_tracks (int):		Number of reference points (tracks) to 
        						distinguish from
        nb_gtw (int):			How many gateways are in the feature space list
        batch (int):			Keras batch size
        epochs (int):			Keras number of epochs
        neurons1 (int):			Keras number of neurons
        dropout1 (int):			Keras dropout fraction
        n_dataset (int):		Dataset size. Only for display and comparison purpose
        n_meas (int):			How many packets form a feature space. Only for
        						display and comparison purpose
        activation (string):	Keras activation function
        layers (int):			How many layers of the same neuron number

    Returns:
        tuple (float, float):	Mean accuracy and validation accuracy over the last
        						10 epochs of the model					
    """
	#Enable GPU
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = tf.Session(config=config)

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
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(neurons1, activation=activation, input_shape=(nb_gtw, 3,)))
	for i in range(layers):
		model.add(tf.keras.layers.Dropout(dropout1))
		model.add(tf.keras.layers.Dense(neurons1, activation=activation))
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(nb_tracks, activation="softmax"))
			
	
	#print the summary of the model
	#model.summary()

	model.compile(
		#for a multi-class classification problem
		optimizer = "rmsprop",
		loss = "categorical_crossentropy",
		metrics = ["accuracy"]
	)

	tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/nr{}-drp{}-bat{}-dat{}k-pts{}-lay{}-{}".format(neurons1,dropout1,batch,int(n_dataset/1000),n_meas,layers,time()))

	results = model.fit(
		training_set[0], one_hot_labels_train,
		epochs=epochs,
		batch_size=batch,
		validation_data=(testing_set[0],one_hot_labels_test),
		callbacks=[tensorboard],
		verbose=2
		)	

	#return mean over the last 10 epochs
	return np.mean(results.history["acc"][epochs-10:]), np.mean(results.history["val_acc"][epochs-10:])