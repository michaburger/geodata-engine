import datetime
import database as db
import mapping
import json
import plot
import random
import numpy as np
import pandas as pd
import geometry as geo
import fingerprinting as fp
import clustering as cl
import sys
import time

import csv

#create gateway array including office gateways
def gateway_list():
	trk_array = []
	for i in range (3,12):
		trk_array.append(db.request_track(i))
	trk_array.append(db.request_track(20))
	gtws = fp.get_gateways(trk_array)
	return gtws

def gateway_list_track(track):
	trk_array = []
	trk_array.append(track)
	return fp.get_gateways(trk_array)

'''
#import multiple gateways from file
LAT = 2
LON = 3
EUI = 1

with open('antenna.csv') as csvfile:
	inputs = csv.reader(csvfile, delimiter=',')
	for idx, row in enumerate(inputs):
		if(idx>0):
			print(db.add_gateway(row[EUI],row[LAT],row[LON]))
'''


#Trilateration - optimize parameters
#geo.dist_to_gtw()
#geo.trilat_opt()


'''
#Trilateration graphic part with intersecting circles
mapping.add_gateway_layer(db.request_gateways(25))

for ref_point in range (3,9):
	#circle_data = geo.trilat_extract_info(db.request_track(ref_point,0,7),db.request_gateways(30),ref_point)
	#mapping.add_circle_layer(circle_data,ref_point)

	intersections = geo.trilateration(db.request_track(ref_point,0,7),db.request_gateways(30),ref_point,(3,3))
	mean = geo.mean_coords(intersections,1,1,1,1,1)
	mapping.add_marker(mean,'Location'+str(ref_point))

	#mapping.add_intersection_markers(intersections,"Intersections point "+str(ref_point))

mapping.output_map('Circles.html')
'''

#compare the effect of multiple devices
#plot.device_comparison(db.request_track(9,0,7,'78AF580300000485'),db.request_track(9,0,7,'78AF580300000506'))

#weather condition plots
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-15_15:00:00","2018-03-21_00:00:00"),"Weather conditions 15.3.2018 to 20.3.2018")
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-09_17:00:00","2018-03-12_00:00:00"),"Weather conditions 9.3.2018")
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-27_17:00:00","2018-03-28_10:00:00"),"Weather conditions 28.3.2018")


#Start measures with SF7
#plot.time_graph_rssi(db.request_track(2,0,7,"2018-04-03_16:00:00","2018-04-05_10:00:00"),"Weather conditions 4.3.2018")
#plot.time_graph_rssi(db.request_track(2,0,7,"2018-04-18_17:00:00","2018-04-19_10:00:00"),"Weather conditions 18.4.2018")
#plot.temperature_rssi(db.request_track(2),"ESP vs Weather conditions SF7")


'''
#plot.distance_plot(db.request_track(20,5,7),db.request_gateways(30),'0B030153')
sf = 7
txpow = 0
plot.distance_plot_all(db.request_track(20,txpow,sf),db.request_track(3,txpow,sf),db.request_track(4,txpow,sf),db.request_track(5,txpow,sf),db.request_track(6,txpow,sf),db.request_track(7,txpow,sf),db.request_track(8,txpow,sf),db.request_gateways(30),txpow,sf)

#ref_point = 4
#geo.distance_list(db.request_track(ref_point,0,9),db.request_gateways(30),ref_point)
'''

'''
sf = 7
txpow = 0
for i in range(3,12):
	print("******************************")
	print("Trilateration point #" +str(i))
	plot.trilat_quick_plot(db.request_gateways(30),db.request_track(i,txpow,sf,'78AF580300000485'),i,txpow,sf)
'''


#20.4.2018 - Start ML algorithms

'''
#jaccard index inside tracks and between tracks
d_size = 100
for track1 in range(3,12):
	for track2 in range(3,12):
		c = 0
		d = 0
		dataset1 = fp.create_dataset(db.request_track(track1),dataset_size=d_size,nb_measures=30)
		dataset2 = fp.create_dataset(db.request_track(track2),dataset_size=d_size,nb_measures=30)
		#calculate mean
		for p1 in dataset1:
			for p2 in dataset2:
				c += fp.jaccard_index(p1,p2)
		mean = c / (d_size**2)

		
		#calculate stdev
		for p1 in dataset1:
			for p2 in dataset2:
				d += (fp.jaccard_index(p1,p2)-mean)**2
		var = d / (d_size**2)
		sigma = np.sqrt(var)
		

		print(str(mean)+"\t",end="")
	print("")
'''

'''
#test accuracy of jaccard classifier
d_size = 1000
nb_iter = 3
nb_measures = 5
nb_tests = 100

timer = 0
correct_classifications = 0
wrongly_classified_tracks = []
fp.create_comparison_set(d_size,nb_measures)

for t in range(nb_tests):
	start = time.time()
	print(".",end="")
	sys.stdout.flush() #display point immediately
	#generate test track
	trk_nb = random.randint(3,11)
	#print("Test #"+str(t+1)+", Track: "+str(trk_nb))
	classification = fp.jaccard_classifier(fp.create_dataset(db.request_track(trk_nb),dataset_size=d_size,nb_measures=nb_measures),d_size=d_size,nb_measures=nb_measures,nb_iter=nb_iter)
	end = time.time()
	timer += (end-start)
	if classification[0] == trk_nb:
		correct_classifications += 1
	else:
		wrongly_classified_tracks.append((trk_nb,)+classification)
timer = timer / nb_tests
accuracy = 100.0*correct_classifications/nb_tests
print("\n************************************")
print("***ACCURACY OF JACCARD CLASSIFIER***")
print("Mean step execution time: "+str(timer))
print("Dataset size: "+str(d_size))
print("Measures per dataset: "+str(nb_measures))
print("Jaccard classifier iterations: "+str(nb_iter))
print("Number of test samples: "+str(nb_tests))
print("ACCURACY: "+str(accuracy)+"%")
print("Wrongly classified tracks: (correct track, classification, jaccard index) \n"+str(wrongly_classified_tracks))
print("************************************")
'''

'''
#OVERVIEW MAP
#get gateways from database and plot them on the map
mapping.add_gateway_layer(db.request_gateways(25))

#Add Track 1 as a point layer
#gtws = ['0B030153','080E0FF3','080E04C4','080E1007','080E05AD','080E0669','080E0D73','080E1006','080E0D61', '004A0DB4']
gtws = gateway_list_track(db.request_track(20,0,7,'ALL',250,"2018-04-27_11:00:00"))

for cnt, g in enumerate(gtws):
	mapping.add_point_layer(db.request_track(20,0,7,'ALL',250,"2018-04-27_11:00:00"),gtws[cnt],gtws[cnt],3,500)
	#geo.distance_list(db.request_gateways(30),db.request_track(6, start="2018-03-20_00:00:00"),gtws[cnt],6)

#mapping.add_point_layer(db.request_track(1),"3 satellites",3,500)

#adding the heatmap
#mapping.add_heatmap(db.request_track(1))

#output map
mapping.output_map('maps/track20.html')
'''


'''
#4.5.2018 - Clustering
clustering_test_track = db.request_track(20,0,7,'ALL',300,"2018-04-27_11:00:00")

#only take into account the gateways seen in the defined time period. Don't accept gateways built afterwards.
gtws = gateway_list_track(db.request_track(20,0,7,'ALL',300,"2018-04-27_11:00:00","2018-05-31_00:00:00"))
nb_gtws = len(gtws)

#have around 10-30 points per cluster. This is a parameter to optimize
nb_clusters = int(len(clustering_test_track)/30)
print("Number of clusters: {}".format(nb_clusters))

#Agglomerative clustering
set_with_clusters = cl.distance_clustering_agglomerative(clustering_test_track,nb_clusters=nb_clusters,min_points=20)

#DBSCAN clustering
#set_with_clusters, nb_clusters = cl.distance_clustering_dbscan(clustering_test_track,max_unlabeled=0.05)

cluster_array = cl.cluster_split(set_with_clusters,nb_clusters)


#draw map
#for cnt, g in enumerate(gtws):
#	mapping.add_point_layer(set_with_clusters,gtws[cnt],gtws[cnt],3,250,coloring='clusters')
#mapping.output_map('maps/clustering-map-agglomerative.html')


#9.5.2018 - Applying PCA
#training_set, testing_set = fp.create_dataset_tf(cluster_array,gtws,dataset_size=100,nb_measures=10,train_test=1,offset=0)
#fp.apply_pca(training_set,nb_clusters,0)

#24.5.2018
#AGGLOMERATIVE 2ND CLUSTERING
training_set, validation_set = fp.create_dataset_pandas(cluster_array, gtws, dataset_size=500, nb_measures=12)
#intermediate storage to avoid recalculating dataset every time
training_set.to_csv("storage-training.csv")
validation_set.to_csv("storage-validation.csv")
cfile = open("clsize.mikka","w")
cfile.write(str(nb_clusters))
cfile.close()
gfile = open("gtwnb.mikka","w")
gfile.write(str(nb_gtws))
gfile.close()
'''


#import pre-computed dataset
training_set = pd.read_csv("storage-training.csv")
validation_set = pd.read_csv("storage-validation.csv")
cfile = open("clsize.mikka","r")
nb_clusters = int(cfile.read())
cfile.close()
gfile = open("gtwnb.mikka","r")
nb_gtws = int(gfile.read())
gfile.close()


#norm both sets the same way
training_set_norm, validation_set_norm = cl.normalize_data(training_set,validation_set)
print("Data normalized")

#test different parameters
cl_size = 1.0
ncl = int(nb_clusters*cl_size)
clusters_training = cl.clustering_feature_space_agglomerative(training_set_norm,nb_clusters=ncl)
print("Training set done")
clusters_validation = cl.clustering_feature_space_agglomerative(validation_set_norm,nb_clusters=ncl)
print("Validation set done")

clusters_training.to_csv("/data/clusters_training-100-12.csv")
clusters_validation.to_csv("/data/clusters_validation-100-12.csv")


clusters_training = pd.read_csv("/data/clusters_training-100-12.csv")
clusters_validation = pd.read_csv("/data/clusters_validation-100-12.csv")

#Todo: correctly attribute same cluster numbers to label2 for training and validation. or check predictive model.


#mapping.print_map_from_pandas(clusters_training,ncl,'maps/clustering-2nd-agglomerative-raw.html')
#mapping.print_map_from_pandas(clusters_validation,ncl,'maps/clustering-2nd-agglomerative-stdnorm.html')

fp.neuronal_classification_clusters(clusters_training,clusters_validation,nb_clusters)

'''
#30.5.2018
#testing different parameters

clustering_test_track = db.request_track(20,0,7,'ALL',300,"2018-04-27_11:00:00","2018-05-30_00:00:00")
gtws = gateway_list_track(db.request_track(20,0,7,'ALL',300,"2018-04-27_11:00:00","2018-05-30_00:00:00"))

nb_measures = 20
dataset_size = 30

tab = []
for nb_measures in range(6,21,3):
	for cluster_points in range(5,50,5):
		nb_clusters = int(len(clustering_test_track)/cluster_points)
		#Agglomerative clustering
		set_with_clusters = cl.distance_clustering_agglomerative(clustering_test_track,nb_clusters=nb_clusters,min_points=10)
		cluster_array = cl.cluster_split(set_with_clusters,nb_clusters)
		dataset_pd, empty = fp.create_dataset_pandas(cluster_array, gtws, dataset_size=dataset_size, nb_measures=nb_measures)
		for c in range(1,11):
			cl_size = c/10.0
			ncl = int(nb_clusters*cl_size)
			print("Evaluating: Cluster reduction {} - NB clusters 1st {}".format(cl_size,nb_clusters))
			mean_dist, max_dist, min_dist = cl.agglomerative_clustering_mean_distance(dataset_pd,nb_clusters,cl_size)
			print("Mean distance {} - Max {} - Min {}".format(mean_dist,max_dist,min_dist))
			tab.append([cl_size,nb_clusters,ncl,nb_measures,dataset_size,mean_dist,max_dist,min_dist])
df = pd.DataFrame(data=tab,columns=['Cluster reduction','NB clusters 1st','NB clusters 2nd','NB measures','Dataset size','Mean distance','Biggest cluster','Smallest cluster'])
df.to_csv('data/cluster-size-eval.csv')
'''



'''
#****************************
#DBSCAN 2ND CLUSTERING 
dataset_pd, empty = fp.create_dataset_pandas(cluster_array, gtws, dataset_size=100, nb_measures=20)
min_samples = 1

goal_reduction = 0.6
goal_metrics = 0.96

result = []
while True:
	#calculate feature space like done for classification preparation. Is giving two times the same feature space as output. 
	dataset_2_cl, nb_cl = cl.clustering_feature_space_dbscan(dataset_pd,min_samples=min_samples,max_unlabeled=0.05,normalize=False)
	#next line for dbscan only
	cl_size = float(nb_cl)/nb_clusters
	print("min_samples: {} - nb clusters: {}".format(min_samples,nb_cl))

	metrics = cl.compute_clustering_metrics(dataset_2_cl)

	print("Cluster size: {} - Metrics: {}".format(cl_size,metrics))
	#uncomment next line for data logging
	#result.append({'Cluster size':cl_size,'Correct Points':metrics})

	min_samples += 2
	if cl_size < goal_reduction and metrics > goal_metrics:
		break

#result_pd=pd.DataFrame(data=result,columns=['Cluster size','Correct Points'])
#result_pd.to_csv('results_2nd_clustering_dbscan.csv')

mapping.print_map_from_pandas(dataset_2_cl,nb_cl,'maps/clustering-2nd-dbscan.html')

#****************************
'''



'''
#24.4.2018 Tensorflow

#reference_gateways = gateway_list() #for reference-track-clustering
reference_gateways = gtws #for track 20 clustering only

#input arguments
if len(sys.argv) == 8:
	NB_DATA = int(sys.argv[1])
	NB_MEAS = int(sys.argv[2])
	BATCH = int(sys.argv[3])
	EPOCHS = int(sys.argv[4])
	TRAIN_TEST = int(sys.argv[5])
	NEURONS1 = int(sys.argv[6])
	DROPOUT1 = int(sys.argv[7])

else:
	print('WARNING: Wrong input arguments. Default values taken')
	NB_DATA = 10000
	NB_MEAS = 10
	BATCH = 10
	EPOCHS = 50
	TRAIN_TEST = 0.5
	NEURONS1 = 32
	DROPOUT1 = 0.3
	LAYERS = 2

#multiple parameter evaluation during the night
param_layers = [1,2,3]
act_functions = ["relu"]#,"tanh","sigmoid"]
param_neurons = [16,32,64,128,256]
param_dropout = [0.0,0.2,0.4]
param_nb_meas = [20]
param_nb_data = [10000]

#write headers
try:
	f = open('/data/multitest-weekend.log','w')
	f.write('Epochs: {}, Train-test: {}, batch: {}, mean over 10 last epochs.\n'.format(EPOCHS,TRAIN_TEST,BATCH))
	f.write("layers\tneurons\tdropout\tnb_measurement\tnb_data\ttraining_accuracy\tvalidation_accuracy\toverfit\texecution_time\n")
	f.close()
except:
	print("WARNING: File write error. Logging disabled! ")


for n_dataset in param_nb_data:
	for n_meas in param_nb_meas:
		for dropout in param_dropout:
			for neurons in param_neurons:
				for activation in act_functions:
					for layers in param_layers:
						acc_arr = []
						val_acc_arr = []
						ex_arr = []
						for m in range(1):

							#track array classification for reference tracks
							#trk_array = []
							#nb_tracks = 9
							#for i in range (3,3+nb_tracks):
							#	track = db.request_track(i)
							#	trk_array.append(track)

							#track 20 classification after clustering
							trk_array = cluster_array
							nb_tracks = nb_clusters

							training_set, testing_set = fp.create_dataset_tf(trk_array,reference_gateways,dataset_size=n_dataset,nb_measures=n_meas,train_test=TRAIN_TEST,offset=0)
							start = time.time()
							acc, val_acc = fp.neuronal_classification(training_set,testing_set,nb_tracks,len(reference_gateways),BATCH,EPOCHS,neurons,dropout,n_dataset,n_meas,activation,layers)
							end = time.time()
							acc_arr.append(acc)
							val_acc_arr.append(val_acc)
							ex_arr.append(end-start)

						try:
							f = open('/data/multitest-weekend.log','a')
							f.write(str(layers)+"\t"+str(neurons)+"\t"+str(dropout)+"\t"+str(n_meas)+"\t"+str(n_dataset)+"\t"+str(np.mean(acc_arr))+"\t"+str(np.mean(val_acc_arr))+"\t"+str((np.mean(acc_arr)-np.mean(val_acc)/np.mean(acc)))+"\t"+str(np.mean(ex_arr))+"\n")
							f.close()
						except:
							print("WARNING: File write error. Logging disabled! ")

						print("***PARAMETERS***")
						print("Layser: {}".format(layers))
						print("Activation function: "+activation)
						print("Batch size: "+str(BATCH))
						print("Dataset size: "+str(n_dataset))
						print("Measures per dataset: "+str(n_meas))
						print("Dropout: "+str(dropout))
						print("Neurons: "+str(neurons))
						print("***RESULTS***")
						print("Acc: "+str(acc_arr))
						print("Val_acc: "+str(val_acc_arr))
						print("Execution time (s): "+str(ex_arr))

'''


'''
#25.5.2018 - Device calibration
#dev485 = db.request_track(40,0,7,'78AF580300000485')
#dev506 = db.request_track(40,0,7,'78AF580300000506')
#plot.distance_plot_compare(dev485,'78AF580300000485',dev506,'78AF580300000506',db.request_gateways(30))
#gtws = gateway_list_track(db.request_track(40,0,7,'78AF580300000506'))
mapping.add_point_layer(db.request_track(40,0,7,'78AF580300000485'),'78AF580300000485','0B030153',3,500)
mapping.add_point_layer(db.request_track(40,0,7,'78AF580300000506'),'78AF580300000506','0B030153',3,500)
mapping.output_map('maps/device-comparison-all.html')
'''
