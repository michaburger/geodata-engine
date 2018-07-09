import json
import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import geometry as geo
import random

coord_list = [(46.520312, 6.565633),(46.519374, 6.569038),(46.517747, 6.569007),(46.516938, 6.563536),(46.522087, 6.563415),(46.521034, 6.571053),(46.518912, 6.566103),(46.518215, 6.563403),(46.521293, 6.568626)]
color_list = []

#init map
map = folium.Map(location=[46.52, 6.565],zoom_start=16,tiles='stamentoner')
folium.TileLayer(tiles='openstreetmap').add_to(map)

def add_circle_layer(circle_data, ref_point):
	circle_layer = folium.FeatureGroup(name="Reference point " + str(ref_point))
	for i, crl in enumerate(circle_data):
		circle_layer.add_child(folium.Circle(
			location=[crl['Lat'],crl['Lon']],
			fill=False,
			weight=5,
			opacity=0.75,
			radius=geo.distance(crl['ESP-mean']),
			popup=	"Gateway: " + str(crl['EUI']) + "<br/>" +
					"Variance: " + str(geo.deltax(crl['ESP-mean'],crl['ESP-var'])) + "<br/>"+
					"+1 sigma: " + str(geo.distance(crl['ESP-mean'])-crl['ESP-var']) + "m<br/>" +
					"-1 sigma: " + str(geo.distance(crl['ESP-mean'])+crl['ESP-var']) + "m<br/>" 
			))
		
		circle_layer.add_child(folium.Circle(
			location=[crl['Lat'],crl['Lon']],
			fill=False,
			weight=3,
			opacity=0.5,
			radius=geo.distance(crl['ESP-mean']-3*crl['ESP-var']),
			popup= "Gateway: " + str(crl['EUI']) + "<br/>" +
					"+3Sigma"
			))
		circle_layer.add_child(folium.Circle(
			location=[crl['Lat'],crl['Lon']],
			fill=False,
			weight=3,
			opacity=0.5,
			radius=geo.distance(crl['ESP-mean']+3*crl['ESP-var']),
			popup= "Gateway: " + str(crl['EUI']) + "<br/>" +
					"-3Sigma"
			))


	#add trilateration point
	circle_layer.add_child(folium.Marker(location=coord_list[ref_point-3],popup="Ref point: "+str(ref_point),icon=folium.Icon(color='darkred',prefix='fa',icon='angle-double-down')))
	map.add_child(circle_layer)

#has to be called at the end to generate the map file
def output_map(filename):
	map.add_child(folium.LayerControl())
	map.save(filename)

def add_intersection_markers(markers,layer_name):
	mrk = folium.FeatureGroup(name=layer_name)
	for m in markers:
		mrk.add_child(folium.Marker(location=m['Intersection'],icon=folium.Icon(color='green',prefix='fa',icon='crosshairs'),
			popup=	"Gateway: "+str(m['Gateway'])+ "<br/>" +
					"Distance: "+str(m['Closer distance'])+ "<br/>" +
					"Variance: "+str(m['Closer variance'])+ "<br/>" +
					"Distance (mean): "+str(m['Mean distance'])+ "<br/>" +
					"Variance (mean): "+str(m['Mean variance'])+ "<br/>" +
					"Sigma ring: "+str(m['Sigma ring'])
					))
	map.add_child(mrk)

def add_marker(coordinates,layer_name):
	mrk = folium.FeatureGroup(name=layer_name)
	mrk.add_child(folium.Marker(location=[coordinates[0],coordinates[1]],icon=folium.Icon(color='red',prefix='fa',icon='bolt'),popup= layer_name))
	map.add_child(mrk)

def add_gateway_layer(gtw, layerName='Gateways'):
	fgtw = folium.FeatureGroup(name=layerName)

	gtw_id=[]
	gtw_lat=[]
	gtw_lon=[]

	#count: remove index from gtw and put it in count
	for count, gtw in enumerate(gtw):
		gtw_id.append(gtw['gateway_id'])
		gtw_lon.append(gtw['gateway_lon'])
		gtw_lat.append(gtw['gateway_lat'])

	for gtw_id,gtw_lat,gtw_lon in zip(gtw_id,gtw_lat,gtw_lon):
		fgtw.add_child(folium.Marker(location=[gtw_lat,gtw_lon],popup="ID: "+str(gtw_id),icon=folium.Icon(color='darkblue',prefix='fa',icon='rss')))

	map.add_child(fgtw)

def hexcol(col):
	hexstr = str(hex(col).replace('0x',''))
	if col <16:
		return '0'+ hexstr
	else:
		return hexstr

def pick_color_heat(heat):
	low_range = 70 #ESP for green
	mid_range = 90 #ESP for orange
	high_range = 105 #ESP for red
	max_range = 120 #ESP for blue
	green_min = 0
	green_max = 200
	red_min = 0
	red_max = 255
	blue_min = 0
	blue_max = 127
	red_slope = (red_max-red_min)/(mid_range-low_range)
	green_slope = (green_min-green_max)/(high_range-mid_range)
	red_slope_dec = (red_min-red_max)/(max_range-high_range)
	blue_slope = (blue_max-blue_min)/(max_range-high_range)
	red_zero = red_min-red_slope*low_range
	red_zero_dec = red_min-red_slope_dec*max_range
	green_zero = green_min-green_slope*high_range
	blue_zero = blue_min-blue_slope*high_range

	if heat < low_range:
		return '#' + hexcol(red_min) + hexcol(green_max)+ '00'
	elif low_range<=heat<mid_range:
		red = (int) (red_slope*heat + red_zero)
		return'#'+hexcol(red)+hexcol(green_max)+'00'
	elif mid_range<=heat<high_range:
		green = (int) (green_slope*heat+green_zero)
		return '#'+hexcol(red_max)+hexcol(green)+'00'
	elif high_range<=heat<max_range:
		blue = (int) ((blue_slope)*heat+blue_zero)
		red_dec = (int) ((red_slope_dec)*heat+red_zero_dec)
		return '#'+hexcol(red_dec)+'00'+hexcol(blue)
	else:
		return '#0000'+hexcol(blue_max)

def pick_color_clusters(cluster):
	#fill color list array
	nb_clusters = 500
	if len(color_list)==0:
		for i in range(nb_clusters):
			color_list.append('#'+hexcol(random.randint(0,255))+hexcol(random.randint(0,255))+hexcol(random.randint(0,255)))
	if cluster >= nb_clusters:
		return "#ffffff"
	elif cluster == -1:
		return '#ffffff'
	else:
		return color_list[cluster-1]

def random_color():
	return '#'+hexcol(random.randint(0,255))+hexcol(random.randint(0,255))+hexcol(random.randint(0,255))

def pick_color(heat,cluster,color_clusters):
	if color_clusters:
		return pick_color_clusters(cluster)
	else:
		return pick_color_heat(heat)


def pick_opacity(heat):
	if heat == 0:
		return 0
	else:
		return 0.7

#print particles from pandas
def print_particles(particles_pd,layer_name,real_pos,**kwargs):
	particles = kwargs['particles'] if 'particles' in kwargs else True
	heatmap = kwargs['heatmap'] if 'heatmap' in kwargs else True
	particles_layer = folium.FeatureGroup(name=layer_name + ": particles")
	heatmap_layer = folium.FeatureGroup(name=layer_name + ": heatmap")
	real_position = folium.FeatureGroup(name=layer_name + ": real position")

	real_position.add_child(folium.CircleMarker(location=[real_pos[0],real_pos[1]],
			fill=True,radius=10,
			color='',
			fill_color='red',
			fill_opacity=1))

	lat, lon, heat = ([] for i in range(3))
	for idx, particle in particles_pd.iterrows():
		lat.append(particle.loc['lat'])
		lon.append(particle.loc['lon'])
		heat.append(1/(particle.loc['age']+1))
		particles_layer.add_child(folium.CircleMarker(location=[particle.loc['lat'],particle.loc['lon']],
			fill=True,radius=10,
			popup="Age: {}".format(int(particle.loc['age'])),
			color='',
			fill_color='blue',
			fill_opacity=1/(particle.loc['age']+1)**0.7))

	htmp_data=list(zip(lat,lon,heat))
	heatmap_layer.add_child(HeatMap(htmp_data,radius=25,blur=50,min_opacity=0.5,))

	if particles:
		map.add_child(particles_layer)
	if heatmap:
		map.add_child(heatmap_layer)
	map.add_child(real_position)



def add_point_layer(pts, layerName='PointLayer', gateway= '0B030153', minSatellites = 1, minHDOP = 500, **kwargs):

	color_clusters = False
	if 'coloring' in kwargs and kwargs['coloring']=='clusters':
		color_clusters = True

	ftr1 = folium.FeatureGroup(name=layerName)
	lat, lon, time, timest, dev, hum, temp, sp, gps_sat, gps_hdop, gateways, rssi, snr, esp, heat, cluster = ([] for i in range(16))
	for count, trk in enumerate(pts):
		for i, gtw in enumerate(trk['gateway_id']):
			#only consider points with at least 5 satellites and antenna 1 and don't plot the cluster -1
			if(trk['gps_sat']>=minSatellites and trk['gps_hdop']<minHDOP) and trk['track_ID'] >= 0 and gtw == gateway:
				lat.append(trk['gps_lat'])
				lon.append(trk['gps_lon'])
				timest.append(trk['timestamp']['$date']) #todo: format time
				time.append(trk['time'])
				dev.append(trk['devEUI'])
				hum.append(trk['humidity'])
				temp.append(trk['temperature'])
				sp.append(trk['sp_fact'])
				gps_sat.append(trk['gps_sat'])
				gps_hdop.append(trk['gps_hdop'])
				gateways.append(trk['gateway_id'])
				rssi.append(trk['gateway_rssi'])
				snr.append(trk['gateway_snr'])
				esp.append(trk['gateway_esp'])
				cluster.append(str(trk['track_ID']))
				
				#used for the color
				heat.append((int)(trk['gateway_rssi'][i]))

	for lat,lon,time,timest,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat,cluster in zip(lat,lon,time,timest,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat,cluster):
		#print("heat: "+str(heat)+", color: "+pick_color(heat))
		#create delete link
		timeformat = str(time).split(".")

		ftr1.add_child(folium.CircleMarker(location=[lat,lon],
			fill=True,radius=10,
			popup="<b>Time: </b>" + str(time) + "<br/>"
			+ "<b>Device: </b>" + str(dev) + "<br/>"
			+ "<b>Temperature: </b>" + str(temp) + "°C<br/>"
			+ "<b>Humidity: </b>" + str(hum) + "% RH</br>"
			+ "<b>Satellites: </b>" + str(gps_sat) + "<br/>"
			+ "<b>HDOP: </b>" + str(gps_hdop) + "<br/>"
			+ "<b>SF: </b>" + str(sp) + "<br/>"
			+ "<b>Gateways: </b>" + ", ".join(gateways) + "<br/>"
			+ "<b>RSSI: </b>" + str(rssi) + "<br/>"
			+ "<b>SNR: </b>" + str(snr) + "<br/>"
			+ "<b>ESP: </b>" + str(esp) + "<br/>"
			+ "<b>Cluster: </b>" + str(cluster) + "<br/>"
			+ "<b><a href=https://spaghetti.scapp.io/query?delpoint="+timeformat[0]+" target=\"_blank\">Delete point</a></b>",
			color='',
			fill_color=pick_color(heat,int(cluster),color_clusters),
			fill_opacity=pick_opacity(heat)))

	map.add_child(ftr1)
	print("Map: "+layerName+" rendered!")

def print_map_from_pandas(df,nb_cl,path):
	reduced = df.loc[:,['rLat','rLon','Label2']].values.tolist()

	#seperate list by clusters
	cluster_array = [[] for i in range(nb_cl)]
	distance_list = df.loc[:,['Label2','rLat','rLon']].values.tolist()
	
	#split for every cluster
	for point in distance_list:
		cluster_array[int(point[0])].append(point)

	#filter for points which are contours
	cluster_array_contour = []
	for cluster in cluster_array:
		contours = []
		for p1 in cluster:
			n = s = e = w = False
			lat1 = p1[1]
			lon1 = p1[2]
			for p2 in cluster:
				lat2 = p2[1]
				lon2 = p2[2]
				if lat2 > lat1:
					n = True
				if lat2 < lat1:
					s = True
				if lon2 > lon1:
					e = True
				if lon2 < lon1:
					w = True
			#if there is no other point in one of the directions N,S,E,W, keep it as a contour
			if (n and s and e and w) == False:
				contours.append(p1)
		cluster_array_contour.append(contours)

	#draw every cluster as a polygon
	#draw polygon around all those points


	for idx,cluster in enumerate(cluster_array):
		ftr1 = folium.FeatureGroup(name='Cluster {}'.format(idx))
		color = random_color()
		for point in cluster:
			if point[0] > -1:
				ftr1.add_child(folium.CircleMarker(location=[point[1],point[2]],
					fill=True,radius=10,color='',
					popup="Cluster " + str(int(point[0])),
					fill_color=color,
					fill_opacity=0.7
					))
		map.add_child(ftr1)
	output_map(path)
	print("Clustering map saved at " + path)

#try to norm heat value to have better results
def add_heatmap(pts, layerName='Heatmap', gateway= '0B030153', minSatellites = 1, minHDOP = 500):

	fhmap = folium.FeatureGroup(name=layerName)
	lat, lon, esp, heat = ([] for i in range(4))

	for count, trk in enumerate(pts):
		#only consider points with at least 5 satellites and antenna 1
		if(trk['gps_sat']>=minSatellites and trk['gps_hdop']<minHDOP) and trk['gateway_id'][0] == gateway:
			lat.append(trk['gps_lat'])
			lon.append(trk['gps_lon'])
			esp.append(trk['gateway_esp'])

			#first gateway heatmap production. Store ESP value if the EPFL gateway has received the signal.
			if(trk['gateway_id'][0] == gateway):
				heat.append(-1000000000000000/trk['gateway_esp'][0])
			else:
				heat.append(0)
	data=list(zip(lat,lon,heat))

	fhmap.add_child(HeatMap(data,radius=25,blur=50,min_opacity=0.7,gradient={0.1:'grey',0.3: 'red', 0.6: 'orange', 1: 'green'}))
	map.add_child(fhmap)




