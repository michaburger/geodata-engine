import numpy as np
import database as db
import json
import geopy.distance
import math
import plot

d2r = math.pi/180
coord_list = [(46.520312, 6.565633),(46.519374, 6.569038),(46.517747, 6.569007),(46.516938, 6.563536),(46.522087, 6.563415),(46.521034, 6.571053)]

#returns the approximate distance in meters according to trilateration report, April 2018
def distance(esp):
	a=17.7
	b=53.32
	dist = 1/(np.power(10,(esp+b)/a))
	return dist

def deltax(esp,variance):
	maximum = distance(esp-variance)
	minimum = distance(esp+variance)
	return (maximum - minimum) / 2

def trilat_extract_info(point_list, gateway_list, ref_point):
	p_dict = json.loads(point_list.decode('utf-8'))
	g_dict = json.loads(gateway_list.decode('utf-8'))

	distance, rssi, esp = ([] for i in range(3))

	distance_data_dict = []

	for g in g_dict:
		eui = g['gateway_id']
			
		esp_total = rssi_total = snr_total = 0
		esp_var = rssi_var = snr_var = 0
		esp_var_tot = rssi_var_tot = snr_var_tot = 0
		esp_mean = rssi_mean = snr_mean = 0

		gtw_coords = (g['gateway_lat'], g['gateway_lon'])
		point_coords = coord_list[ref_point-3] #offset: 3

		counter = 0
		for p in p_dict:
			for j, inner in enumerate(p['gateway_id']):
				if eui == p['gateway_id'][j]:
					esp_total += p['gateway_esp'][j]
					rssi_total += p['gateway_rssi'][j]
					snr_total += p['gateway_snr'][j]
					counter += 1
		if counter: 
			esp_mean = esp_total / counter
			rssi_mean = rssi_total / counter
			snr_mean = snr_total / counter
		else: 
			esp_mean = snr_mean = rssi_mean = 0


		#second loop for variance
		counter = 0
		for p in p_dict:
			for j, inner in enumerate(p['gateway_id']):
				if eui == p['gateway_id'][j]:
					esp_var_tot += math.pow(p['gateway_esp'][j]-esp_mean , 2)
					rssi_var_tot += math.pow(p['gateway_rssi'][j]-rssi_mean , 2)
					snr_var_tot += math.pow(p['gateway_snr'][j]-snr_mean , 2)
					counter += 1
		if counter: 
			esp_var = math.sqrt(esp_var_tot / counter)
			rssi_var = math.sqrt(rssi_var_tot / counter)
			snr_var = snr_var_tot / counter
		else: 
			esp_var = snr_var = rssi_var = 0
		if counter>=10:
			distance_data_dict.append({'Track':ref_point,'EUI':eui,'Lat':g['gateway_lat'],'Lon':g['gateway_lon'],'ESP-mean':esp_mean,'ESP-var':esp_var,'packets':counter})
	#print(distance_data_dict)
	return distance_data_dict

#distance optimizing function over x (labelled trilateration tracks)
def trilat_opt_foo(x,params):
	#x input: array/tensor with: lat, lon, all params
	#weights input
	#output: distance 
	w1,w2,w3,w4,w5,r1,r2 = params
	distance = 0
	#for every different track
	for trk in x:
		intersections = trilateration(db.request_track(trk,0,7),db.request_gateways(30),trk,(r1,r2))
		mean = mean_coords(intersections,w1,w2,w3,w4,w5)
		distance += geopy.distance.vincenty(mean,coord_list[trk-3]).km
	return distance

def trilat_opt():
	tracks = [3,4,5,6,7,8]
	weight_array = []
	for i in range (1,11):
		weight_array.append(i/10.0)

	#try: minimize trilat_opt_foo over all labelled tracks with brute-force
	best_params = [1,1,1,1,1,3,3]
	min_dist = trilat_opt_foo(tracks,(best_params[0],best_params[1],best_params[2],best_params[3],best_params[4],best_params[5],best_params[6]))

	for w1 in weight_array:
		print("Param_loop1: "+str(w1))
		for w2 in weight_array:
			print("Param_loop2: "+str(w2))
			for w3 in weight_array:
				print("Param_loop3: "+str(w3))
				for w4 in weight_array:
					print("Param_loop4: "+str(w4))
					for w5 in weight_array:
						for w6 in weight_array:
							for w7 in weight_array:
								r1 = w6*5
								r2 = w7*5
								dist = trilat_opt_foo(tracks,(w1,w2,w3,w4,w5,r1,r2))
								if(dist<min_dist):
									min_dist = dist
									best_params = [w1,w2,w3,w4,w5,r1,r2]
	print("*******")
	print("Best parameters: "+str(best_params))
	print("Mean deviation: "+str(min_dist/len(tracks)))


def trilateration(point_list, gateway_list, ref_point, filter):

	#filter: tuple(inner_filter,outer_filter)
	#read all point and gateway data
	data = trilat_extract_info(point_list, gateway_list, ref_point)

	#create a zero reference point to be used for the circle coordinate system
	total_lat = total_lon = c = 0
	for gtw in data:
		total_lat += float(gtw['Lat'])
		total_lon += float(gtw['Lon'])
		c+= 1
	zero_point = total_lat/c, total_lon/c
	#print(zero_point)
	#print(data)

	gtw_pairs = []
	intersect_points = []

	#for every pair of gateways:
	for gtw1 in data:
		for gtw2 in data:
			if gtw1['EUI'] != gtw2['EUI']:
				#Every pair exists twice. Filter.
				gtw_pairs.append({'EUI1':gtw1['EUI'],'EUI2':gtw2['EUI']})
				#if the inverse pair does not exist yet
				if {'EUI1':gtw2['EUI'],'EUI2':gtw1['EUI']} not in gtw_pairs:
					#print("Gateway pair: "+gtw1['EUI']+ " + "+gtw2['EUI'])
					'''
					#testing my own functions against geopy. Test ok, almost corresponds to great circle
					p1_geopy = geopy.distance.great_circle((gtw1['Lat'],gtw1['Lon']),zero_point).km * 1000
					p1_lat = coord_to_m('lat',zero_point[0]-gtw1['Lat'],gtw1['Lat'])
					p1_lon = coord_to_m('lon',zero_point[1]-gtw1['Lon'],gtw1['Lat'])
					p1_homemade = math.sqrt(p1_lat*p1_lat+p1_lon*p1_lon)
					print("Geopy distance: " + str(p1_geopy))
					print("Own function dist: " + str(p1_homemade))
					
					#Test: conversion to refcoords and back --> works
					print("Reference point: " + str(gtw1['Lat'])+", "+str(gtw1['Lon']))
					banana = latlon_to_ref((gtw1['Lat'],gtw1['Lon']),zero_point)
					print("Point in reference coordinates: "+str(banana))
					avocado = ref_to_latlon(banana,zero_point)
					print("Conversion back to lat/lon: " + str(avocado))
					'''

					#build rings with sigma_rings time the variance
					for param in range(0,61):
						sigma_ring = 0.1*param-3

						if distance(gtw1['ESP-mean']) < distance(gtw2['ESP-mean']):
							closer_gateway = gtw1['EUI']
							closer_distance = distance(gtw1['ESP-mean'])
							closer_variance = distance(gtw1['ESP-var'])
						else:
							closer_gateway = gtw2['EUI']
							closer_distance = distance(gtw2['ESP-mean'])
							closer_variance = distance(gtw2['ESP-var'])

						mean_distance = distance((gtw1['ESP-mean']+gtw2['ESP-mean'])/2)
						mean_variance = distance((gtw1['ESP-var']+gtw2['ESP-var'])/2)

						circle1 = latlon_to_ref((gtw1['Lat'],gtw1['Lon']),zero_point) + (distance(gtw1['ESP-mean']+sigma_ring*gtw1['ESP-var']),)
						circle2 = latlon_to_ref((gtw2['Lat'],gtw2['Lon']),zero_point) + (distance(gtw2['ESP-mean']+sigma_ring*gtw2['ESP-var']),)
						p1_refsystem, p2_refsystem = circle_intersection(circle1,circle2)
						if p1_refsystem != None: 
							p1_latlon = ref_to_latlon(p1_refsystem,zero_point)
							p2_latlon = ref_to_latlon(p2_refsystem,zero_point)

							#check validity of the point (within 3 sigma of every gateway)
							valid1 = valid2 = True
							for g in data:
								#for every gateway, if distance point-gateway is bigger than dist+3sigma, point is invalid
								#only one gateway not fitting into the criteria is setting it to false.

								#outside circle
								if geopy.distance.vincenty(p1_latlon,(g['Lat'],g['Lon'])).km*1000 > distance(g['ESP-mean']-filter[1]*g['ESP-var']):
									valid1 = False
								if geopy.distance.vincenty(p2_latlon,(g['Lat'],g['Lon'])).km*1000 > distance(g['ESP-mean']-filter[1]*g['ESP-var']):
									valid2 = False

								#inside circle
								if geopy.distance.vincenty(p1_latlon,(g['Lat'],g['Lon'])).km*1000 < distance(g['ESP-mean']+filter[0]*g['ESP-var']):
									valid1 = False
								if geopy.distance.vincenty(p2_latlon,(g['Lat'],g['Lon'])).km*1000 < distance(g['ESP-mean']+filter[0]*g['ESP-var']):
									valid2 = False

							if(valid1):
								intersect_points.append({'Ref':ref_point,'Gateway':closer_gateway, 'Intersection':p1_latlon, 'Closer distance':closer_distance, 'Closer variance':closer_variance, 'Mean distance':mean_distance, 'Mean variance':mean_variance, 'Sigma ring':sigma_ring})
							if(valid2):
								intersect_points.append({'Ref':ref_point,'Gateway':closer_gateway, 'Intersection':p2_latlon, 'Closer distance':closer_distance, 'Closer variance':closer_variance, 'Mean distance':mean_distance, 'Mean variance':mean_variance, 'Sigma ring':sigma_ring})

	return intersect_points

#returns an estimated lat, lon using arbitrary weights for the model
def mean_coords(intersect_points,w1,w2,w3,w4,w5):

	c = 0
	dist_max = var_max = dist_m_max = var_m_max= 0.0

	for i in intersect_points:
		c += 1
		dist_weight = 1.0/i['Closer distance']
		var_weight = 1.0/i['Closer variance']
		dist_m_weight = 1.0/i['Mean distance']
		var_m_weight = 1.0/i['Mean variance']

		#norming everything. sigma ring already uses an equation which is normed to 1
		if dist_weight > dist_max:
			dist_max = dist_weight
		if var_weight > var_max:
			var_max = var_weight
		if dist_m_weight > dist_m_max:
			dist_m_max = dist_m_weight
		if var_m_weight > var_m_max:
			var_m_max = var_m_weight

	#now using the calculated maximum values to norm everything
	lat_t = lon_t = 0.0
	param_sum = 0.0
	c=0
	for i in intersect_points:
		#normed values
		dist_n = (1.0/i['Closer distance'])/dist_max
		var_n = (1.0/i['Closer variance'])/var_max
		dist_m_n = (1.0/i['Mean distance'])/dist_m_max
		var_m_n = (1.0/i['Mean variance'])/var_m_max
		ring_n = 1.0/(np.power(2.0,abs(i['Sigma ring'])))

		c+=1
		multiplier = w1*dist_n + w2*var_n + w3*ring_n + w4*dist_m_n + w5*var_m_n
		param_sum += multiplier

		lat_t += i['Intersection'][0]*multiplier
		lon_t += i['Intersection'][1]*multiplier
	if c:
		lat, lon = (lat_t,lon_t)/param_sum
	else:
		lat,lon=0,0
	
	return lat,lon	


	
def latlon_to_ref(point, refpoint):
	'''
	@summary: calculates the reference coordinates respective to the reference point, in meters
	@param point: tuple(lat,lon)
	@param refpoint: tuple(lat,lon)
	@result: tuple of reference coordinates (which are (x,y) tuple)
	'''

	plat, plon = point
	rlat, rlon = refpoint

	clat = plat - rlat
	clon = plon - rlon

	return coord_to_m('lon',clon,rlat),coord_to_m('lat',clat,rlat)

def ref_to_latlon(point, refpoint):
	'''
	@summary: calculates the absolute lat-lon coordinates from the reference coordinates
	@param point: tuple(x,y)
	@param refpoint: tuple(lat,lon)
	@result: tuple of reference coordinates (which are (lat,lon) tuple)
	'''
	rlat, rlon = refpoint
	x, y = point

	lat = rlat + m_to_coord('lat',y,rlat)
	lon = rlon + m_to_coord('lon',x,rlat)

	return lat,lon

#because geopy.distance doesn't offer an inverse function. 
#Results compareable to geopy.distance.great_circle
def m_to_coord(latlon, meter, deglat):
	R = 40030173
	if latlon == 'lon':
		return (meter/(np.cos(np.radians(deglat))*R))*360.0
	elif latlon == 'lat':
		return (meter/R)*360.0
	else:
		return 0

#because geopy.distance doesn't offer an inverse function
def coord_to_m(latlon, degrees, deglat):
	R = 40030173
	if latlon == 'lon':
		return (degrees/360.0)*(np.cos(np.radians(deglat))*R)
	elif latlon == 'lat':
		return (degrees/360.0)*R
	else:
		return 0

def distance_list(point_list, gateway_list, ref_point):
	p_dict = json.loads(point_list.decode('utf-8'))
	g_dict = json.loads(gateway_list.decode('utf-8'))

	#print(p_dict)
	#print(g_dict)

	distance, rssi, esp = ([] for i in range(3))

	#to do: Automatically detect all gateways
	for g in g_dict:
		eui = g['gateway_id']
			
		esp_total = rssi_total = snr_total = 0
		esp_var = rssi_var = snr_var = 0
		esp_var_tot = rssi_var_tot = snr_var_tot = 0
		esp_mean = rssi_mean = snr_mean = 0

		gtw_coords = (g['gateway_lat'], g['gateway_lon'])
		point_coords = coord_list[ref_point-3] #offset: 3

		counter = 0
		for p in p_dict:
			for j, inner in enumerate(p['gateway_id']):
				if eui == p['gateway_id'][j]:
					esp_total += p['gateway_esp'][j]
					rssi_total += p['gateway_rssi'][j]
					snr_total += p['gateway_snr'][j]
					counter += 1
		if counter: 
			esp_mean = esp_total / counter
			rssi_mean = rssi_total / counter
			snr_mean = snr_total / counter
		else: 
			esp_mean = snr_mean = rssi_mean = 0


		#second loop for variance
		counter = 0
		for p in p_dict:
			for j, inner in enumerate(p['gateway_id']):
				if eui == p['gateway_id'][j]:
					esp_var_tot += math.pow(p['gateway_esp'][j]-esp_mean , 2)
					rssi_var_tot += math.pow(p['gateway_rssi'][j]-rssi_mean , 2)
					snr_var_tot += math.pow(p['gateway_snr'][j]-snr_mean , 2)
					counter += 1
		if counter: 
			esp_var = esp_var_tot / counter
			rssi_var = rssi_var_tot / counter
			snr_var = snr_var_tot / counter
		else: 
			esp_var = snr_var = rssi_var = 0
		if counter>=10:
			
			print('***')
			print(ref_point)
			print(eui)
			print(geopy.distance.vincenty(gtw_coords,point_coords).km)
			print(math.sqrt(rssi_var))
			print(math.sqrt(esp_var))
			print(counter)
			
			'''
			print('**********')
			print(eui)
			print('Real distance: ' + str(geopy.distance.vincenty(gtw_coords,point_coords).km))
			print('ESP mean value: ' + str(esp_mean))
			print('ESP variance: ' + str(math.sqrt(esp_var)))
			print('RSSI mean value: ' + str(rssi_mean))
			print('RSSI standard deviation: ' + str(math.sqrt(rssi_var)))
			print('SNR mean value: ' + str(snr_mean))
			print('SNR standard deviation: ' + str(math.sqrt(snr_var)))
			print('Number of measures: ' + str(counter))
			'''

#circle intersection utility by xaedes https://gist.github.com/xaedes/974535e71009fa8f090e
def circle_intersection(circle1, circle2):
        '''
        @summary: calculates intersection points of two circles
        @param circle1: tuple(x,y,radius)
        @param circle2: tuple(x,y,radius)
        @result: tuple of intersection points (which are (x,y) tuple)
        '''
        # return self.circle_intersection_sympy(circle1,circle2)
        x1,y1,r1 = circle1
        x2,y2,r2 = circle2
        # http://stackoverflow.com/a/3349134/798588
        dx,dy = x2-x1,y2-y1
        d = math.sqrt(dx*dx+dy*dy)
        if d > r1+r2:
            return None,None # no solutions, the circles are separate
        if d < abs(r1-r2):
            return None,None # no solutions because one circle is contained within the other
        if d == 0 and r1 == r2:
            return None,None # circles are coincident and there are an infinite number of solutions

        a = (r1*r1-r2*r2+d*d)/(2*d)
        h = math.sqrt(r1*r1-a*a)
        xm = x1 + a*dx/d
        ym = y1 + a*dy/d
        xs1 = xm + h*dy/d
        xs2 = xm - h*dy/d
        ys1 = ym - h*dx/d
        ys2 = ym + h*dx/d

        return (xs1,ys1),(xs2,ys2)