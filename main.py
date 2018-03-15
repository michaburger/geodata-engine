import datetime
import database as db
import mapping
import plot
import triangulation as trg

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

#weather conditions 15.3.2018 to 20.3.2018 plot
plot.time_graph_rssi(db.request_track(2,"2018-03-15_15:00:00","2018-03-21_00:00:00"),"Weather conditions 15.3.2018 to 20.3.2018")

#get gateways from database and plot them on the map
#mapping.add_gateway_layer(db.request_gateways(25))

#Add Track 1 as a point layer
gtws = ['0B030153','080E0FF3','080E04C4','080E1007','080E05AD','080E0669','0B0308D6','080E0D73','080E1006','080E0D61']

#for cnt, g in enumerate(gtws):
	#mapping.add_point_layer(db.request_track(20),gtws[cnt],gtws[cnt],3,250)
	#trg.distance_list(db.request_gateways(30),db.request_track(3),gtws[cnt])

#mapping.add_point_layer(db.request_track(1),"3 satellites",3,500)

#adding the heatmap
#mapping.add_heatmap(db.request_track(1))

#output map
#mapping.output_map('Map1.html')