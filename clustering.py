import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.cluster import DBSCAN


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

#unsupervised clustering based on physical distance between the points. DBSCAN
def distance_clustering_dbscan(dataset, **kwargs):
	#the maximum fraction of points without label. the function will automatically optimize the EPS parameter to reach this value.
	max_unlabeled = 0.05

	if 'max_unlabeled' in kwargs:
		max_unlabeled = kwargs['max_unlabeled']

	#create array containing only coordinates and the points in the same order as dataset
	coords = []
	for point in dataset:
		coords.append([point['gps_lat'],point['gps_lon']])

	X = np.array(coords)

	#metrics
	unlabeled = 1.0
	eps  = 0.00005
	step = 0.00001

	#optimise eps to reach correct fraction of unlabeled points
	while unlabeled > max_unlabeled:
		db = DBSCAN(eps=eps,min_samples=2).fit(X)
		labels = db.labels_
		# Number of clusters in labels, ignoring noise if present.
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		#count elements
		unique, counts = np.unique(labels, return_counts=True)
		if -1 in labels:
			n_outliers = dict(zip(unique, counts))[-1]
		else:
			n_outliers = 0
		unlabeled = float(n_outliers)/len(labels)
		print('EPS: {} - unlabeled points: {}'.format(eps,unlabeled))
		eps += step

	print(db)
	print("nb clusters: {}".format(n_clusters))
	print("nb outliers: {}".format(n_outliers))
	print("metrics: {}".format(metrics))
	print(labels)
	
	#add cluster id to point data
	for i, point in enumerate(dataset):
		point.update({'track_ID':labels[i]})
	print("DBSCAN clustering done!")
	return dataset, n_clusters

#unsupervised clustering based on physical distance between the points. Agglomerative method
def distance_clustering_agglomerative(dataset, **kwargs):
	#default values
	nb_clusters = 10
	min_points = 5

	if 'nb_clusters' in kwargs:
		nb_clusters = kwargs['nb_clusters']
	if 'min_points' in kwargs:
		min_points = kwargs['min_points']

	#create array containing only coordinates and the points in the same order as dataset
	coords = []
	for point in dataset:
		coords.append([point['gps_lat'],point['gps_lon']])

	X = np.array(coords)
	model = AgglomerativeClustering(n_clusters=nb_clusters,linkage="average")
	model.fit(X)

	clusters = model.fit_predict(X)

	#count points per cluster
	unique, counts = np.unique(clusters, return_counts=True)
	occurrence = dict(zip(unique, counts))

	#add cluster id to point data
	for i, point in enumerate(dataset):
		if occurrence[clusters[i]] >= min_points:
			point.update({'track_ID':clusters[i]})
		else:
			point.update({'track_ID':-1})
	print("Agglomerative clustering done!")
	return dataset

#split cluster dataset into array of datasets for each cluster, to be used to plot on the map like different tracks.
def cluster_split(dataset, nb_clusters, **kwargs):
	cluster_array = [[] for i in range(nb_clusters)]

	for point in dataset:
		cluster_id = point['track_ID']
		cluster_array[cluster_id-1].append(point)
	return cluster_array