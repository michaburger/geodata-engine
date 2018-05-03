import datetime
import database as db
import mapping
import json
import plot
import random
import numpy as np
import geometry as geo
import fingerprinting as fp
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
	trk_array.append(db.request_track(track))
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

'''
#get gateways from database and plot them on the map
mapping.add_gateway_layer(db.request_gateways(25))

#Add Track 1 as a point layer
#gtws = ['0B030153','080E0FF3','080E04C4','080E1007','080E05AD','080E0669','080E0D73','080E1006','080E0D61', '004A0DB4']
gtws = gateway_list_track(20)

for cnt, g in enumerate(gtws):
	mapping.add_point_layer(db.request_track(20),gtws[cnt],gtws[cnt],3,500)
	#geo.distance_list(db.request_gateways(30),db.request_track(6, start="2018-03-20_00:00:00"),gtws[cnt],6)

#mapping.add_point_layer(db.request_track(1),"3 satellites",3,500)

#adding the heatmap
#mapping.add_heatmap(db.request_track(1))

#output map
mapping.output_map('maps/clustering-map.html')
'''

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
#plot.temperature_rssi(db.request_track(2),"RSSI vs Weather conditions SF7")


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


#24.4.2018 Tensorflow

reference_gateways = gateway_list()

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
	BATCH = 50
	EPOCHS = 100
	TRAIN_TEST = 0.5
	NEURONS1 = 32
	DROPOUT1 = 0.3

#multiple parameter evaluation during the night
param_neurons = [32,64,128]
param_dropout = [0.0,0.1,0.2,0.3,0.4]
param_nb_meas = [10,5]
param_nb_data = [10000]

#write headers
try:
	f = open('/data/sigmoid.log','w')
	f.write("Testing parameters for 1-layer NN. Accuracies: Mean over last 4 epochs. Total: 64 Epochs, Batch size 16.\n")
	f.write("neurons\tdropout\tnb_measurement\tnb_data\ttraining_accuracy\tvalidation_accuracy\toverfit\texecution_time\n")
	f.close()
except:
	print("WARNING: File write error. Logging disabled! ")


for n_dataset in param_nb_data:
	for n_meas in param_nb_meas:
		for dropout in param_dropout:
			for neurons in param_neurons:
				acc_arr = []
				val_acc_arr = []
				ex_arr = []
				for m in range(1):
					trk_array = []
					nb_tracks = 9
					for i in range (3,3+nb_tracks):
						track = db.request_track(i)
						#print("Track "+str(i)+ " length: "+str(len(json.loads(track.decode('utf-8')))))
						trk_array.append(track)

					training_set, testing_set = fp.create_dataset_tf(trk_array,reference_gateways,dataset_size=n_dataset,nb_measures=n_meas,train_test=TRAIN_TEST)
					start = time.time()
					acc, val_acc = fp.neuronal_classification(training_set,testing_set,nb_tracks,len(reference_gateways),BATCH,EPOCHS,neurons,dropout,n_dataset,n_meas)
					end = time.time()
					acc_arr.append(acc)
					val_acc_arr.append(val_acc)
					ex_arr.append(end-start)

				try:
					f = open('/data/sigmoid.log','a')
					f.write(str(neurons)+"\t"+str(dropout)+"\t"+str(n_meas)+"\t"+str(n_dataset)+"\t"+str(np.mean(acc_arr))+"\t"+str(np.mean(val_acc_arr))+"\t"+str((np.mean(acc_arr)-np.mean(val_acc)/np.mean(acc)))+"\t"+str(np.mean(ex_arr))+"\n")
					f.close()
				except:
					print("WARNING: File write error. Logging disabled! ")

				print("***PARAMETERS***")
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
#output results in table format
print("Measures\tTest accuracy\tValidation accuracy")
for i in results:
	print(str(i['Measures'])+"\t"+str(i['Test accuracy'])+"\t"+str(i['Validation accuracy']))
'''