import numpy as np
import pandas as pd
import fingerprinting as fp
import mapping as mp
import random
import math
import datetime

N_SAMPLE = 250
CLUSTER_R = 20
SPEED = 0
DISCARD = 0.3 #historical discard
FLATTEN_PROBABILITY = 0.5 #take n-root after the min-max probability calculation
FIRST_VALUES = 10 #how many of the first guesses to consider

pf_store = pd.DataFrame(columns=['lat','lon','age'])

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

#returns a random position within the circle with radius CLUSTER_R
def get_random_position(lat,lon):
	#in meters
	vector_l = random.uniform(0,CLUSTER_R)
	angle = random.uniform(0,360)
	dx = vector_l * math.cos(angle)
	dy = vector_l * math.sin(angle)

	#coords conversion
	dxc = m_to_coord('lon', dx, lat)
	dyc = m_to_coord('lat', dy, lat)

	return (lat+dyc, lon+dxc)

def create_time_series(validation, nb_meas):
	#split validation track into sub-tracks, always NB_MEAS points
	TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+02:00" #parse with a different parser to take into account the time zone
	last_time = datetime.datetime.now()
	current_serie = []
	all_series = []
	for point in validation:
		point_time = datetime.datetime.strptime(point['time'],TIME_FORMAT)
		time_difference = point_time - last_time
		#if non concecutive transmission, start new serie
		if(time_difference.total_seconds()>10 or len(current_serie)>=nb_meas):
			if(len(current_serie)>=nb_meas):
				all_series.append(current_serie)
			current_serie = []
		current_serie.append(point)
		last_time = point_time
	return all_series

def get_particle_distribution(sample_feature_space,database,nncl,layer_name,**kwargs):
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	metrics_probability = kwargs['metrics_probability'] if 'metrics_probability' in kwargs else True
	best_classes = fp.cosine_similarity_classifier_knn(database,sample_feature_space,nncl,first_values=FIRST_VALUES,flatten=FLATTEN_PROBABILITY)
	print(best_classes)
	global pf_store
	particles = []
	#for every cluster, sample p*N_SAMPLE points with random position inside cluster
	for idx, line in best_classes.iterrows():
		if metrics_probability:
			nb_particles = int(round(line.loc['Probability']*N_SAMPLE))
		else:
			nb_particles = int(round(line.loc['Mean Similarity']*N_SAMPLE))
		#print("Cluster {}, Generating {} particles".format(idx,nb_particles))
		for p in range(nb_particles):
			lat, lon = get_random_position(line.loc['Lat'],line.loc['Lon'])
			particles.append((lat,lon,0))
	new_particles = pd.DataFrame(data=particles,columns=['lat','lon','age'])

	if pf_store.empty == False:
		#resample historical data
		pf_store = pf_store.sample(frac=1).reset_index(drop=True).loc[:int(pf_store.shape[0]*(1-DISCARD)),:]
		pf_store['age'] = pf_store['age']+1
	pf_store = pf_store.append(new_particles,ignore_index=True)
	#print(pf_store)

	if render_map:
		mp.print_particles(pf_store,layer_name)



	#TODO for dynamical algorithm: increase cluster size if speed != 0

	#print current particle distribution for double check



