import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


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
	return dataset, n_clusters + 1

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

	#add cluster id to point data.
	for i, point in enumerate(dataset):
		if occurrence[clusters[i]] >= min_points:
			point.update({'track_ID':clusters[i]})
		else:
			point.update({'track_ID':-1})
	print("Agglomerative clustering done!")
	return dataset

#split cluster dataset into array of datasets for each cluster, to be used to plot on the map like different tracks.
def cluster_split(dataset, nb_clusters, **kwargs):
	cluster_array = [[] for i in range(nb_clusters-1)]

	for point in dataset:
		cluster_id = point['track_ID']
		if cluster_id >= 0:
			cluster_array[cluster_id].append(point)
	return cluster_array

#clustering on feature-space, based on pandas dataset as input
def clustering_feature_space_agglomerative(df, **kwargs):
	features = df.drop(columns=['Label1','Lat','Lon'])

	nb_clusters = 10
	normalize = True

	if 'nb_clusters' in kwargs:
		nb_clusters = int(kwargs['nb_clusters'])
	if 'normalize' in kwargs:
		if kwargs['normalize'] == False:
			normalize = False

	#normalize feature space for a more appropriate clustering
	if normalize:
		data = StandardScaler().fit_transform(features)
	else: 
		data = features

	model = AgglomerativeClustering(n_clusters=nb_clusters,linkage="average")
	model.fit(data)

	labels = model.fit_predict(data)

	labels_pd = pd.DataFrame(data=labels,columns=['Label2'])
	dataset = pd.concat([df,labels_pd],axis=1)

	return dataset

def clustering_feature_space_dbscan(df, **kwargs):
	features = df.drop(columns=['Label1','Lat','Lon'])

	max_unlabeled = 0.05
	min_samples = 5
	normalize = True

	#initialization for while loop and standard parameter
	unlabeled = 1.0

	if 'max_unlabeled' in kwargs:
		max_unlabeled = kwargs['max_unlabeled']
	if 'min_samples' in kwargs:
		min_samples = kwargs['min_samples']
	if 'normalize' in kwargs:
		if kwargs['normalize'] == False:
			normalize = False

	if normalize:
		data = StandardScaler().fit_transform(features)
		eps = 2
		step = 0.1
	else:
		data = features
		eps  = 120
		step = 5

	#optimise eps to reach correct fraction of unlabeled points

	while unlabeled > max_unlabeled:
		db = DBSCAN(eps=eps,min_samples=min_samples).fit(data)
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
	#print('Min_samples: {}'.format(min_samples))
	#print("nb clusters: {}".format(n_clusters))
	

	#print("nb clusters: {}".format(n_clusters))
	#print("nb outliers: {}".format(n_outliers))
	#print("metrics: {}".format(metrics))
	
	#add cluster id to point data
	labels_pd = pd.DataFrame(data=labels,columns=['Label2'])
	dataset = pd.concat([df,labels_pd],axis=1)

	return dataset, n_clusters + 1

#compute the fraction of points which are in the same cluster for the first and also second clustering.
def compute_clustering_metrics(df):
	#calculate metrics and put it in the form {2nd_cluster:{1st_cluster_A_count:N, 1st_cluster_B_count:N}}
	labels = df.loc[:,['Label1','Label2']].values.tolist()
	pairs_count = {}
	for pair in labels:
		if str(pair[0]) in pairs_count:
			if str(pair[1]) in pairs_count[str(pair[0])]:
				pairs_count[str(pair[0])].update({str(pair[1]):pairs_count[str(pair[0])][str(pair[1])]+1})
			else:
				pairs_count[str(pair[0])].update({str(pair[1]):1})
		else:
			pairs_count.update({str(pair[0]):{str(pair[1]):1}})

	#calculate metrics - how many points of the first cluster are still in the second cluster?
	#approximate metrics, works only for nb_2nd < nb_1st 
	correct_count = 0
	false_count = 0
	for cl1 in pairs_count.items():
		majority = 0
		minority = 9999999
		for cl2 in cl1[1].values():
			if cl2 > majority:
				majority = cl2
			if cl2 < minority:
				minority = cl2
		correct_count += majority
		for cl2 in cl1[1].values():
			if cl2 < majority:
				false_count += cl2
	return float(correct_count) / float(correct_count + false_count)

#Second clustering step, fully integrated and taking into account the optimization metrics
def agglomerative_clustering_with_metrics(dataset_pd,nb_clusters,**kwargs):
	cl_size = 1.0
	step = 0.05

	goal_metrics = 0.96

	if 'metrics' in kwargs:
		goal_metrics = kwargs['metrics']

	result = []
	while True:
		#calculate feature space like done for classification preparation. Is giving two times the same feature space as output. 
		dataset_2_cl = clustering_feature_space_agglomerative(dataset_pd,nb_clusters=nb_clusters*cl_size,normalize=False)
		metrics = compute_clustering_metrics(dataset_2_cl)

		print("Cluster size: {} - Metrics: {}".format(cl_size,metrics))
		#print(".",end=" ",flush=True)
		result.append({'Cluster size':cl_size,'Correct Points':metrics})

		cl_size -= step
		if cl_size < 0.0+step or metrics > goal_metrics:
			break

	#result_pd=pd.DataFrame(data=result,columns=['Cluster size','Correct Points'])
	#print(result_pd)
	#result_pd.to_csv('results_2nd_clustering_agglomerative.csv')
	return dataset_2_cl