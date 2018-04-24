import datetime
import database as db
import mapping
import plot
import numpy as np
import geometry as geo
import fingerprinting as fp

import csv

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
gtws = ['0B030153','080E0FF3','080E04C4','080E1007','080E05AD','080E0669','080E0D73','080E1006','080E0D61', '004A0DB4']

for cnt, g in enumerate(gtws):
	mapping.add_point_layer(db.request_track_no_params(30,12),gtws[cnt],gtws[cnt],3,500)
	#geo.distance_list(db.request_gateways(30),db.request_track(6, start="2018-03-20_00:00:00"),gtws[cnt],6)

#mapping.add_point_layer(db.request_track(1),"3 satellites",3,500)

#adding the heatmap
#mapping.add_heatmap(db.request_track(1))

#output map
mapping.output_map('Map1-ESP-SF12.html)
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
for i in range(9,12):
	print("******************************")
	print("Trilateration point #" +str(i))
	plot.trilat_quick_plot(db.request_gateways(30),db.request_track(i,txpow,sf),i,txpow,sf)
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

#24.4.2018 Tensorflow

#create gateway array including office gateways
def gateway_list():
	trk_array = []
	for i in range (3,12):
		trk_array.append(db.request_track(i))
	trk_array.append(db.request_track(20))
	gtws = fp.get_gateways(trk_array)
	return gtws

reference_gateways = gateway_list()
trk_array = []
nb_tracks = 9
for i in range (3,3+nb_tracks):
	trk_array.append(db.request_track(i))

training_set = fp.create_dataset_tf(trk_array,reference_gateways,dataset_size=20,nb_measures=20)
testing_set = fp.create_dataset_tf(trk_array,reference_gateways,dataset_size=5,nb_measures=20)

fp.neuronal_classification(training_set,testing_set,nb_tracks,len(reference_gateways))