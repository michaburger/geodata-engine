import numpy as np
import pandas as pd
import fingerprinting as fp
import geopy.distance
import mapping as mp
import random
import math
import datetime

N_SAMPLE = 250
CLUSTER_R = 30
SPEED = 1.5 #m/s
F_SAMPLING = 30 #seconds between 2 transmissions
DISCARD = 0.5 #historical discard
MAX_AGE = 5 #discard particles older than this
FILTER_AGE = 3 #number of historical values to be used for filtering
FLATTEN_PROBABILITY = 0.5 #take n-root after the min-max probability calculation
FIRST_VALUES = 5 #how many of the first guesses to consider
CLASSIFIER_FUNCTION = 'euclidean' #euclidean, cosine, manhattan or correlation

pf_store_particles = pd.DataFrame(columns=['lat','lon','age','clat','clon'])
pf_store_clusters = pd.DataFrame(columns=['age','Probability','Lat','Lon'])

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
def get_random_position(lat,lon,r):
	#in meters
	vector_l = random.uniform(0,r)
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
		#print(time_difference)
		#if non concecutive transmission, start new serie
		if(time_difference.total_seconds()>10 or len(current_serie)>=nb_meas):
			if(len(current_serie)>=nb_meas):
				all_series.append(current_serie)
			current_serie = []
		current_serie.append(point)
		last_time = point_time
	return all_series

def get_particle_distribution(sample_feature_space,database,nncl,age,real_pos,**kwargs):
	global pf_store_particles
	global pf_store_clusters
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	metrics_probability = kwargs['metrics_probability'] if 'metrics_probability' in kwargs else True
	best_classes = fp.similarity_classifier_knn(database,sample_feature_space,nncl,first_values=FIRST_VALUES,flatten=FLATTEN_PROBABILITY,function=CLASSIFIER_FUNCTION)
	#print(best_classes)
	#print(pf_store_clusters)

	#resampling filter: remove new particles with impossible positions (too far away)
	if pf_store_clusters.empty == False:
		for idx, cluster in best_classes.iterrows():
			discard = True
			for idy, old_cluster in pf_store_clusters.iterrows():
				distance = geopy.distance.vincenty((cluster['Lat'],cluster['Lon']),(old_cluster['Lat'],old_cluster['Lon'])).km*1000
				if distance < CLUSTER_R+F_SAMPLING*SPEED*old_cluster['age']:
					discard = False
			#discard row if too far away from previous measures
			if (discard):
				best_classes.drop(index=idx,inplace=True)	
				#print("Cluster {} dropped".format(idx))

	particles = []
	#for every cluster, sample p*N_SAMPLE points with random position inside cluster
	for idx, line in best_classes.iterrows():
		if metrics_probability:
			nb_particles = int(round(line.loc['Probability']*N_SAMPLE))
		else:
			nb_particles = int(round(line.loc['Mean Similarity']*N_SAMPLE))
		#print("Cluster {}, Generating {} particles".format(idx,nb_particles))
		for p in range(nb_particles):
			lat, lon = get_random_position(line.loc['Lat'],line.loc['Lon'],CLUSTER_R)
			particles.append((lat,lon,0,line.loc['Lat'],line.loc['Lon']))
	new_particles = pd.DataFrame(data=particles,columns=['lat','lon','age','clat','clon'])
	new_clusters = best_classes.drop(['Cluster ID','Variance','Mean Similarity'],axis=1)
	new_clusters['age']=0

	if pf_store_particles.empty == False:
		pf_store_particles['age'] = pf_store_particles['age']+1
		pf_store_clusters['age'] = pf_store_clusters['age']+1

		#remove old points
		pf_store_particles = pf_store_particles.loc[pf_store_particles['age']<=MAX_AGE]
		pf_store_clusters = pf_store_clusters.loc[pf_store_clusters['age']<=FILTER_AGE]

		#resample historical data --> density of particles representing probability
		pf_store_particles = pf_store_particles.sample(frac=1).reset_index(drop=True).loc[:int(pf_store_particles.shape[0]*(1-DISCARD)),:]

		#increase past radius according to device velocity
		for i, particle in pf_store_particles.iterrows():
			lat_dynamic, lon_dynamic = get_random_position(particle.loc['clat'],particle.loc['clon'],CLUSTER_R+F_SAMPLING*SPEED*age)
			pf_store_particles['lat'] = lat_dynamic
			pf_store_particles['lon'] = lon_dynamic

	pf_store_clusters = pf_store_clusters.append(new_clusters,ignore_index=True)
	pf_store_particles = pf_store_particles.append(new_particles,ignore_index=True)

	mp.print_particles(pf_store_particles,"t = {}".format(-1*age),real_pos,heatmap=True,particles=False)
	return pf_store_particles
	#print(pf_store_particles)

#returns mean particle error (the mean distance particle-real point for all particles) and error distance to estimate
#(distance mean coordinates of particles - real point)
def get_mean_error(pd_distribution, real_pos):
	distances = []
	lat = []
	lon = []
	for idx, particle in pd_distribution.iterrows():
		distances.append(geopy.distance.vincenty((particle['lat'],particle['lon']),real_pos).km*1000)
		lat.append(particle['lat'])
		lon.append(particle['lon'])
	return np.mean(distances),geopy.distance.vincenty((np.mean(lat),np.mean(lon)),real_pos).km*1000



