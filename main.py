import datetime
import database as db
import mapping
import plot
import geometry as geo

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
mapping.output_map('Map1-ESP-SF12.html')
'''


#Trilateration part with intersecting circles
mapping.add_gateway_layer(db.request_gateways(25))

for ref_point in range (3,9):
	circle_data = geo.trilat_point_draw_circles(db.request_track(ref_point,0,7),db.request_gateways(30),ref_point)
	mapping.add_circle_layer(circle_data,ref_point)

mapping.output_map('Circles.html')


#weather condition plots
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-15_15:00:00","2018-03-21_00:00:00"),"Weather conditions 15.3.2018 to 20.3.2018")
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-09_17:00:00","2018-03-12_00:00:00"),"Weather conditions 9.3.2018")
#plot.time_graph_rssi(db.request_track_no_params(2,"2018-03-27_17:00:00","2018-03-28_10:00:00"),"Weather conditions 28.3.2018")

#Start measures with SF7
#plot.time_graph_rssi(db.request_track(2,0,7,"2018-04-03_16:00:00","2018-04-05_10:00:00"),"Weather conditions 4.3.2018")
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
sf = 9
txpow = 5
for i in range(9):
	if i in range(4,7):
		print("******************************")
		print("Trilateration point #" +str(i))
		plot.trilat_quick_plot(db.request_gateways(30),db.request_track(i,txpow,sf),i,txpow,sf)
'''
