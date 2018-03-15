import numpy as np
import json
import plot
import geopy.distance


def distance_list(gateway_list, point_list, gateway):
	p_dict = json.loads(point_list.decode('utf-8'))
	g_dict = json.loads(gateway_list.decode('utf-8'))

	#print(p_dict)
	#print(g_dict)

	#to do: Automatically detect all gateways
	for g in g_dict:
		if gateway == g['gateway_id']:
			print('**********')
			eui = gateway
			print(eui)
			#do mean, later: do variance
			esp_total = 0
			esp_counter = 0
			gtw_coords = (g['gateway_lat'], g['gateway_lon'])
			point_coords = (46.520312, 6.565633)
			print('Distance: ' + str(geopy.distance.vincenty(gtw_coords,point_coords).km))
			for p in p_dict:
				for j, inner in enumerate(p['gateway_id']):
					esp_total += p['gateway_esp'][j]
					esp_counter += 1
			esp_mean = esp_total / esp_counter
			print('ESP mean value: ' + str(esp_mean))
