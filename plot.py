import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objs as go
from matplotlib import pylab
from scipy.optimize import curve_fit
import numpy as np
import json
from datetime import datetime
import pandas as pd
from scipy import stats, integrate
import geopy.distance
import seaborn as sns
import re

coord_list = [(46.520312, 6.565633),(46.519374, 6.569038),(46.517747, 6.569007),(46.516938, 6.563536),(46.522087, 6.563415),(46.521034, 6.571053),(46.518912, 6.566103),(46.518215, 6.563403),(46.521293, 6.568626)]

def temperature_rssi(data, plot_title):
	bunch = json.loads(data.decode('utf-8'))
	LRRs = ['0B030153','080E0FF2','080E05AD','080E04C4','080E05AD','080E1006','080E0669','080E1007','080E0FF2','080E1005']

	temp = []
	hum = []

	#rssi data for every gateway 
	rssi = [[np.nan for x in range(len(bunch))] for y in range(len(LRRs))] 

	#print(bunch)
	for i, element in enumerate(bunch):
		if(element['temperature']):
			temp.append(element['temperature'])
			hum.append(element['humidity'])
		else:
			temp.append(np.nan)
			hum.append(np.nan)

		#attribute received signal strengths (gtw_data) to the fixed list (gtw_list) of gateways
		for idx, gtw_list in enumerate(LRRs):
			for idy, gtw_data in enumerate(element['gateway_id']):
				if(gtw_list == gtw_data):
					rssi[idx][i] = element['gateway_rssi'][idy]
	plt.figure()
	plt.subplot(211)
	plt.xlabel('Temperature °C')
	plt.ylabel('RSSI')

	plt.title(plot_title)

	for idx in range(len(LRRs)):
		plt.scatter(temp, rssi[idx], label=LRRs[idx])

	plt.legend()

	plt.subplot(212)
	plt.xlabel('Humidity')
	plt.ylabel('RSSI')

	#plot all Gateway data
	for idx in range(len(LRRs)):
		plt.scatter(hum, rssi[idx], label=LRRs[idx])

	plt.legend()

	plt.show()

def time_graph_rssi(data, plot_title):
	bunch = json.loads(data.decode('utf-8'))
	LRRs = ['0B030153','080E0FF2','080E05AD','080E04C4','080E05AD','080E1006','080E0669','080E1007','080E0FF2','080E1005']

	time, temp, hum = ([] for i in range(3))

	#esp data for every gateway 
	rssi = [[np.nan for x in range(len(bunch))] for y in range(len(LRRs))] 

	#print(bunch)
	for i, element in enumerate(bunch):
		#print(element['time'])
		time_sub = re.sub(':','',element['time'])
		#print(time_sub)
		time.append(datetime.strptime(time_sub,"%Y-%m-%dT%H%M%S.%f%z"))
		temp.append(element['temperature'])
		hum.append(element['humidity'])

		#attribute received signal strengths (gtw_data) to the fixed list (gtw_list) of gateways
		for idx, gtw_list in enumerate(LRRs):
			for idy, gtw_data in enumerate(element['gateway_id']):
				if(gtw_list == gtw_data):
					rssi[idx][i] = element['gateway_rssi'][idy]
	plt.figure()
	plt.subplot(211)
	plt.ylabel('°C / %RH')
	plt.legend()

	plt.title(plot_title)

	plt.plot(time, temp, label='temperature')
	plt.plot(time, hum, label='humidity')

	plt.subplot(212)
	#plot all Gateway data
	for idx in range(len(LRRs)):
		fill_gaps(rssi[idx])
		plt.plot(time, rssi[idx], label=LRRs[idx])


	plt.xlabel('Date')
	plt.ylabel('ESP (dB)')

	plt.legend()

	plt.show()

def trilat_quick_plot(gateway_list, point_list, ref_point,txpow,sf):
	MAX_DISPLAY = 9

	p_dict = json.loads(point_list.decode('utf-8'))
	g_dict = json.loads(gateway_list.decode('utf-8'))

	#print(p_dict)
	#print(g_dict)

	gtw_cnt_dict, occurrences, gtw_frequent = ([] for i in range(3))

	for g in g_dict:
		counter = 0
		eui = g['gateway_id']
		for p in p_dict:
			for j, inner in enumerate(p['gateway_id']):
				if eui in p['gateway_id']:
					counter += 1
		if(counter):
			gtw_cnt_dict.append({
				'gtw_id': eui,
				'occurrence': counter
			})

		#to filter for the 9 highest values only
		occurrences.append(counter)

	occurrences.sort(reverse=True)

	#create array of MAX_DISPLAY most frequent gateways
	for g in gtw_cnt_dict:
		if int(g['occurrence']>occurrences[MAX_DISPLAY]) and int(g['occurrence']>0):
			gtw_frequent.append(g['gtw_id'])

	#plot histograms for the MAX_DISPLAY most frequent gateways

	plt.figure()
	plt.suptitle('RSSI histograms for place #'+str(ref_point)+" SF"+str(sf)+" TXpower: "+str(txpow))
	gtw_cnt=0
	for g in g_dict:
		if g['gateway_id'] in gtw_frequent:
			print("******")
			print("EUI: "+g['gateway_id'])
			gtw_cnt += 1
			gtw_coords = (g['gateway_lat'], g['gateway_lon'])
			point_coords = coord_list[ref_point-3] #offset: 3
			real_distance = geopy.distance.vincenty(gtw_coords,point_coords).km
			print("Distance: "+str(real_distance))
			counter = 0
			mean_total = 0
			distance, rssi, esp = ([] for i in range(3))
			for p in p_dict:
				for j, eui in enumerate(p['gateway_id']):
					if g['gateway_id'] == p['gateway_id'][j]:
						#distance.append(real_distance)
						rssi.append(p['gateway_rssi'][j])
						esp.append(p['gateway_esp'][j])
						mean_total += p['gateway_rssi'][j]
						counter += 1
			if len(rssi)>2:
				plt.subplot(3,3,gtw_cnt)
				plt.xlabel('RSSI')
				plt.ylabel('Occurrence')
				plt.title(g['gateway_id'] + ', packets: ' + str(counter))
				sns.distplot(rssi)
			mean = mean_total / counter
			print("Mean RSSI: "+str(mean))
			print("Packets: " + str(counter))

	plt.tight_layout()
	plt.show()

	plt.figure()
	plt.suptitle('ESP histograms for place #'+str(ref_point)+" SF"+str(sf) + " TXpower: "+str(txpow))
	gtw_cnt=0
	for g in g_dict:
		if g['gateway_id'] in gtw_frequent:
			gtw_cnt += 1
			#gtw_coords = (g['gateway_lat'], g['gateway_lon'])
			#point_coords = coord_list[ref_point-3] #offset: 3
			#real_distance = geopy.distance.vincenty(gtw_coords,point_coords).km
			counter = 0
			distance, rssi, esp = ([] for i in range(3))
			for p in p_dict:
				for j, eui in enumerate(p['gateway_id']):
					if g['gateway_id'] == p['gateway_id'][j]:
						#distance.append(real_distance)
						rssi.append(p['gateway_rssi'][j])
						esp.append(p['gateway_esp'][j])
						counter += 1
			if len(esp)>2:
				plt.subplot(3,3,gtw_cnt)
				plt.xlabel('ESP')
				plt.ylabel('Occurrence')
				plt.title(g['gateway_id'] + ', packets: ' + str(counter))
				sns.distplot(esp)
	plt.tight_layout()
	plt.show()

def distance_plot(point_list, gtw_list, gateway_eui):
	pts = json.loads(point_list.decode('utf-8'))
	g_dict = json.loads(gtw_list.decode('utf-8'))

	dist = []
	rssi = []
	esp = []

	gtw_coords = (0,0)
	for g in g_dict:
		if g['gateway_id'] == gateway_eui:
			gtw_coords = (g['gateway_lat'], g['gateway_lon'])
			print(gtw_coords)

	for p in pts:
		for j, eui in enumerate(p['gateway_id']):
			if gateway_eui == p['gateway_id'][j]:
				real_distance = geopy.distance.vincenty(gtw_coords,(p['gps_lat'],p['gps_lon'])).km * 1000

				#filter points with coords 0,0
				if real_distance < 18000:
					dist.append(real_distance)
					rssi.append(p['gateway_rssi'][j])
					esp.append(p['gateway_esp'][j])
					#print('Dist: '+str(real_distance)+ " | RSSI: "+str(p['gateway_rssi'][j]) + " | ESP: "+str(p['gateway_esp'][j]))

	plt.figure()
	plt.subplot(211)
	plt.xlabel('Distance (m)')
	plt.ylabel('RSSI')
	plt.title('RSSI vs distance - SF'+ str(sf)+' - TXpower '+str(txpow))
	plt.scatter(dist, rssi)

	plt.subplot(212)
	plt.xlabel('Distance (m)')
	plt.ylabel('ESP')
	plt.title('ESP vs distance - SF'+ str(sf)+' - TXpower '+str(txpow))
	plt.scatter(dist, esp)

	plt.show()

def logfunc(x, a, b):
    return a*np.log10(1/x)+b

def device_comparison(track_dev1,track_dev2):
	print(track_dev1)

#plots all gateways on the same plot
def distance_plot_all(point_list, fix1, fix2, fix3, fix4, fix5, fix6,gtw_list,txpow,sf):
	#coord list of fixed measurement points
	NB_FIXPOINTS = 6
	fixed_points = [fix1, fix2, fix3, fix4, fix5, fix6]
	fpts = []
	pts = json.loads(point_list.decode('utf-8'))
	for i in range(NB_FIXPOINTS):
		fpts.append(json.loads(fixed_points[i].decode('utf-8')))
	g_dict = json.loads(gtw_list.decode('utf-8'))

	dist = []
	rssi_scatter = []
	esp_scatter = []
	rssi_all = []
	esp_all = []
	rssi_fixed = []
	esp_fixed = []


	weight_scatter = len(pts)
	fixpoint_weight = 3*int(weight_scatter / NB_FIXPOINTS)

	gtw_coords = (0,0)
	for g in g_dict:
		gtw_coords = (g['gateway_lat'], g['gateway_lon'])

		#store points of the scatter plot (all points at EPFL campus)
		for p in pts:
			for j, eui in enumerate(p['gateway_id']):
				if g['gateway_id'] == p['gateway_id'][j]:
					real_distance = geopy.distance.vincenty(gtw_coords,(p['gps_lat'],p['gps_lon'])).km * 1000
					
					#filter points with coords 0,0
					if real_distance < 10000:
						dist.append(real_distance)
						rssi_scatter.append(p['gateway_rssi'][j])
						esp_scatter.append(p['gateway_esp'][j])
						rssi_all.append(p['gateway_rssi'][j])
						esp_all.append(p['gateway_esp'][j])
						#to still correctly attribute the array to the distances, we have to fill these values with nan
						rssi_fixed.append(np.nan)
						esp_fixed.append(np.nan)
						#print('Dist: '+str(real_distance)+ " | RSSI: "+str(p['gateway_rssi'][j]) + " | ESP: "+str(p['gateway_esp'][j]))

		#store fixed point mean values in the right weight
		for i in range(NB_FIXPOINTS):
			esp_cnt = 0
			rssi_cnt = 0
			cnt = 0
			for p in fpts[i]:
				for j, eui in enumerate(p['gateway_id']):
					if g['gateway_id'] == p['gateway_id'][j]:
						real_distance = geopy.distance.vincenty(gtw_coords,coord_list[i]).km * 1000
						cnt += 1
						rssi_cnt += p['gateway_rssi'][j]
						esp_cnt += p['gateway_esp'][j]
			if(cnt and real_distance < 10000):
				rssi_mean = rssi_cnt / cnt
				esp_mean = esp_cnt / cnt
				for k in range(fixpoint_weight):
					dist.append(real_distance)
					rssi_scatter.append(np.nan)
					esp_scatter.append(np.nan)
					rssi_all.append(rssi_mean)
					esp_all.append(esp_mean)
					rssi_fixed.append(rssi_mean)
					esp_fixed.append(esp_mean)


	y_rssi = rssi_all
	y_esp = esp_all
	rssi_par = []
	esp_par = []

	xx = np.linspace(50,7000,2000)

	popt_rssi, pcov_rssi = curve_fit(logfunc, dist, y_rssi)
	popt_esp, pcov_esp = curve_fit(logfunc, dist, y_esp)

	#standard deviation sigma
	perr_rssi = np.sqrt(np.diag(pcov_rssi))
	perr_esp = np.sqrt(np.diag(pcov_esp))

	#format log parameters
	for i in range(2):
		rssi_par.append("{:2.2f}".format(popt_rssi[i]))
		esp_par.append("{:2.2f}".format(popt_esp[i]))
	
	rssi_func = rssi_par[0]+"log(1/x)"+rssi_par[1]
	esp_func = esp_par[0]+"log(1/x)"+esp_par[1]

	plt.figure()
	plt.subplot(211)
	plt.xlabel('Distance (m)')
	plt.ylabel('RSSI')
	plt.title('RSSI vs distance - SF'+ str(sf)+' - TXpower '+str(txpow))
	plt.scatter(dist, rssi_scatter, marker="x", label="All campus")
	plt.scatter(dist, rssi_fixed, marker="D", c='#ff7d00',label="Fixed points")
	plt.plot(xx,logfunc(xx, *popt_rssi),c='r', label=rssi_func)
	#plt.plot(xx,logfunc(xx,popt_rssi[0]-3*perr_rssi[0],popt_rssi[1]+3*perr_rssi[1]),c='g',label="+3sigma")
	#plt.plot(xx,logfunc(xx,popt_rssi[0]+3*perr_rssi[0],popt_rssi[1]-3*perr_rssi[1]),c='g',label="-3sigma")
	plt.legend()


	plt.subplot(212)
	plt.xlabel('Distance (m)')
	plt.ylabel('ESP')
	plt.title('ESP vs distance - SF'+ str(sf)+' - TXpower '+str(txpow))
	plt.scatter(dist, esp_scatter, marker="x", label="All campus")
	plt.scatter(dist, esp_fixed, marker="D", c='#ff7d00',label="Fixed points")
	plt.plot(xx,logfunc(xx, *popt_esp),c='r', label=esp_func)
	#plt.plot(xx,logfunc(xx,popt_esp[0]-3*perr_esp[0],popt_esp[1]+3*perr_esp[1]),c='g',label="+3sigma")
	#plt.plot(xx,logfunc(xx,popt_esp[0]+3*perr_esp[0],popt_esp[1]-3*perr_esp[1]),c='g',label="-3sigma")
	plt.legend()

	plt.show()

def fill_gaps(data):
	for i in range(1,len(data)):
		if(np.isnan(data[i])):
			data[i] = data[i-1]



