import matplotlib.pyplot as plt
import numpy as np
import json

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def get_gateways(dataset):
	gtws = []
	for i in range(len(trk)):
		for gtw in trk[i]['gateway_id']:
			if gtw not in gtws:
				gtws.append(gtw)
	return gtws

#return a matrix which contains all gateway data but with a zero
def clustering_matrix(point,gtws):
	print(point)

def add_esp_array(dataset):
	gtws = get_gateways(dataset)

	#arrange data in a vector with ESP of 
	new_set = []
	for point in dataset:
		matrix = clustering_matrix(point,gtws)
		#then add matrix to point object

#unsupervised clustering based on physical distance between the points
def distance_clustering(dataset, **kwargs):
	#default values
	nb_clusters = 10

	if 'nb_clusters' in kwargs:
		nb_clusters = kwargs['nb_clusters']

	#create array containing only coordinates and the points in the same order as dataset
	coords = []
	for point in dataset:
		coords.append([point['gps_lat'],point['gps_lon']])

	X = np.array(coords)
	model = AgglomerativeClustering(n_clusters=nb_clusters,linkage="average")
	model.fit(X)

	clusters = model.fit_predict(X)

	#add cluster id to point data
	for i, point in enumerate(dataset):
		point.update({'track_ID':clusters[i]})
	return dataset

#split cluster dataset into array of datasets for each cluster
def cluster_split(dataset, nb_clusters, **kwargs):
	cluster_array = [[] for i in range(nb_clusters)]

	for point in dataset:
		cluster_id = point['track_ID']
		cluster_array[cluster_id-1].append(point)
	return cluster_array