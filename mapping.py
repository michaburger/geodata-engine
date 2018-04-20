import json
import folium
from folium.plugins import HeatMap
import numpy as np
import pandas
import geometry as geo

coord_list = [(46.520312, 6.565633),(46.519374, 6.569038),(46.517747, 6.569007),(46.516938, 6.563536),(46.522087, 6.563415),(46.521034, 6.571053),(46.518912, 6.566103),(46.518215, 6.563403),(46.521293, 6.568626)]

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

def add_gateway_layer(gatewayList, layerName='Gateways'):
	gtw = json.loads(gatewayList.decode('utf-8'))
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

def pick_color(heat):
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


def pick_opacity(heat):
	if heat == 0:
		return 0
	else:
		return 0.7

def add_point_layer(pointList, layerName='PointLayer', gateway= '0B030153', minSatellites = 1, minHDOP = 500):
	pts = json.loads(pointList.decode('utf-8'))
	ftr1 = folium.FeatureGroup(name=layerName)
	lat, lon, time, timest, dev, hum, temp, sp, gps_sat, gps_hdop, gateways, rssi, snr, esp, heat = ([] for i in range(15))

	for count, trk in enumerate(pts):
		#only consider points with at least 5 satellites and antenna 1
		if(trk['gps_sat']>=minSatellites and trk['gps_hdop']<minHDOP) and trk['gateway_id'][0] == gateway:
			lat.append(trk['gps_lat'])
			lon.append(trk['gps_lon'])
			timest.append(trk['timestamp']['$date']) #todo: format time
			if 'time' in trk: #old points of track 1 don't have time string yet
				time.append(trk['time'])
			dev.append(trk['deviceType'])
			hum.append(trk['humidity'])
			temp.append(trk['temperature'])
			sp.append(trk['sp_fact'])
			gps_sat.append(trk['gps_sat'])
			gps_hdop.append(trk['gps_hdop'])
			gateways.append(trk['gateway_id'])
			rssi.append(trk['gateway_rssi'])
			snr.append(trk['gateway_snr'])
			esp.append(trk['gateway_esp'])
			#first gateway heatmap production. Store ESP value if the EPFL gateway has received the signal.
			if(trk['gateway_id'][0] == gateway):
				heat.append((int)(-1*trk['gateway_rssi'][0]))
			else:
				heat.append(0)

	for lat,lon,time,timest,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat in zip(lat,lon,time,timest,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat):
		#print("heat: "+str(heat)+", color: "+pick_color(heat))
		ftr1.add_child(folium.CircleMarker(location=[lat,lon],
			fill=True,radius=10,
			popup="<b>Timestamp: </b>" + str(timest) + "<br/>"
			+ "<b>Time: </b>" + str(time) + "<br/>"
			+ "<b>Device: </b>" + str(dev) + "<br/>"
			+ "<b>Temperature: </b>" + str(temp) + "°C<br/>"
			+ "<b>Humidity: </b>" + str(hum) + "% RH</br>"
			+ "<b>Satellites: </b>" + str(gps_sat) + "<br/>"
			+ "<b>HDOP: </b>" + str(gps_hdop) + "<br/>"
			+ "<b>SF: </b>" + str(sp) + "<br/>"
			+ "<b>Gateways: </b>" + ", ".join(gateways) + "<br/>"
			+ "<b>RSSI: </b>" + str(rssi) + "<br/>"
			+ "<b>SNR: </b>" + str(snr) + "<br/>"
			+ "<b>ESP: </b>" + str(esp) + "<br/>",
			color='',
			fill_color=pick_color(heat),
			fill_opacity=pick_opacity(heat)))

	map.add_child(ftr1)
	print("Map: "+layerName+" rendered!")

#try to norm heat value to have better results
def add_heatmap(pointList, layerName='Heatmap', gateway= '0B030153', minSatellites = 1, minHDOP = 500):
	pts = json.loads(pointList.decode('utf-8'))

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




