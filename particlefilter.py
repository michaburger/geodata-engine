"""
Author: Micha Burger, 24.07.2018
https://micha-burger.ch
LoRaWAN Localization algorithm used for Master Thesis 


This file contains the particle filter of the algorithm. Based on the most 
probable position estimates from the fingerprinting algorithm (fingerprinting.py), 
a particle distribution is calculated and adapted step-by-step. Past particle 
distributions and best clusters are stored as global variables. Every time the 
function get_particle_distribution is called, the past values are adapted. 
This file also contains functions used to evaluate the performance of the algorithm.
"""

import numpy as np
import pandas as pd
import fingerprinting as fp
import geopy.distance
import mapping as mp
import random
import math
import datetime

#Definition of the particle filter's parameters to change them easily
#number of particles
N_SAMPLE = 500 

#geographical radius of a cluster in meters
CLUSTER_R = 30 

#speed of movement for dynamical filter in m/s
SPEED = 1.0 

#seconds between 2 transmissions
F_SAMPLING = 60 

#historical discard fraction for particles older than t=-1
DISCARD = 1.0 

#discard particles older than this
MAX_AGE = 5 

#number of historical values to be used for cluster filtering
FILTER_AGE = 1 

#enable / disable dynamical filter
DYNAMICAL_FILTER_ON = True 

 #minimum occurrency of a certain cluster in the historical particles. 
 #clusters with less occurrency will be deleted for noise.
MIN_OCCURR_FRAC = 0.1

#minimum fraction of new particles that have to be in the prediction. 
#When less particles are available (due to filtering / wrong prediction) 
#use position of the last transmission
MIN_NEW_PARTICLES_FRAC = 0.2 

#take n-root after the min-max probability calculation
FLATTEN_PROBABILITY = 1.5 

#how many of the first guesses to consider
FIRST_VALUES = 5 

#dynamical filter: cut off the N clusters furthest away from historical clusters, 
#but keep a certain proportion of particles in all the cases
PARTICLES_KEEP_FRAC = 1.0 

#euclidean, cosine, manhattan or correlation
CLASSIFIER_FUNCTION = 'euclidean' 

#Output intermediate results
VERBOSE = 0 # 0 1 or 2

#global variables for storage of the historical clusters and particles, 
#storage between two filtering steps.
pf_store_particles = pd.DataFrame(columns=['lat','lon','age','cluster','clat','clon'])
pf_store_clusters = pd.DataFrame(columns=['age','Probability','Lat','Lon'])

#because geopy.distance doesn't offer an inverse function. 
#Results compareable to geopy.distance.great_circle
def m_to_coord(latlon, meter, deglat):
	"""Calculates the distance in degrees from a distance in meters.

	The result is only valid for a certain latitude that has to be specified 
	with the parameter deglat.

    Args:
        latlon (string): 	Either 'lat' or 'lon' to define if the coordinate is a
							latitude or longitude
		meter (float): 		Distance in meters
		deglat (float): 	The latitude for which the result will be valid

    Returns:
        float: Distance in degrees
    """

    #radius of the earth in meters
	R = 40030173

	if latlon == 'lon':
		return (meter/(np.cos(np.radians(deglat))*R))*360.0
	elif latlon == 'lat':
		return (meter/R)*360.0
	else:
		return 0

#because geopy.distance doesn't offer an inverse function
def coord_to_m(latlon, degrees, deglat):
	"""Calculates the distance in meters from a distance in degrees.

	The result is only valid for a certain latitude that has to be specified 
	with the parameter deglat.

    Args:
        latlon (string): 	Either 'lat' or 'lon' to define if the coordinate is a
							latitude or longitude
		degrees (float): 	Distance in degrees
		deglat (float): 	The latitude for which the result will be valid

    Returns:
        float: Distance in meters
    """

    #radius of the earth in meters
	R = 40030173

	if latlon == 'lon':
		return (degrees/360.0)*(np.cos(np.radians(deglat))*R)
	elif latlon == 'lat':
		return (degrees/360.0)*R
	else:
		return 0


def get_random_position(lat,lon,r):
	"""returns a coordinate around a center point within a circle of radius r.

	Used to generate random particles to simulate a cluster in the 
	particle filter algorithm

    Args:
        lat (float):		Latitude in degrees
		lon (float):		Longitude in degrees
		r (float): 			Maximum distance to the center point in meters

    Returns:
        tuple: (latitude, longitude) coordinates of the new point
    """

	#in meters
	vector_l = random.uniform(0,r)
	angle = random.uniform(0,360)
	dx = vector_l * math.cos(angle)
	dy = vector_l * math.sin(angle)

	#coords conversion
	dxc = m_to_coord('lon', dx, lat)
	dyc = m_to_coord('lat', dy, lat)

	return (lat+dyc, lon+dxc)

#Used for the validation on campus
def create_time_series(validation, nb_meas):
	"""Rearrange an array of packet transmissions for the validation

	The function takes an array of packet transmissions and rearranges them
	according to the timestamp, in order to always have nb_meas consecutive 
	packet transmissions that can create a feature space. At least nb_meas 
	packets have to be sent in a row for this case, all the rest will be deleted.

    Args:
        validation (array):	Array of points (python dict) as received from the db
		nb_meas (int): 		Size of the feature space to be created

    Returns:
        array of arrays of python dict: The re-arranged array, can be used as 
        track array to be backward-compatible
    """

    #TODO: parse with a different parser to take into account the time zone
	TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f+02:00" 
	last_time = datetime.datetime.now()
	current_serie = []
	all_series = []
	for point in validation:
		point_time = datetime.datetime.strptime(point['time'],TIME_FORMAT)
		time_difference = point_time - last_time
		if VERBOSE == 2: print(time_difference)
		#if non concecutive transmission, start new serie
		if(time_difference.total_seconds()>10 or len(current_serie)>=nb_meas):
			if(len(current_serie)>=nb_meas):
				all_series.append(current_serie)
			current_serie = []
		current_serie.append(point)
		last_time = point_time
	return all_series


def get_particle_distribution(sample_feature_space,database,nncl,age,
	real_pos,**kwargs):
	"""Execute one step of the particle filter algorithm for dynamical localization.

	Take the new feature space as input, execute the similarity classifier with 
	the parameters specified in this file, perform the filter using a global 
	storage of past values and plot the current result on the heatmap as a layer.

    Args:
        validation (array):	Array of points (python dict) as received from the db
		nb_meas (int): 		Size of the feature space to be created

    Returns:
        array of arrays of python dict: The re-arranged array, can be used as 
        track array to be backwarrd-compatible
    """
	global pf_store_particles
	global pf_store_clusters
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	metrics_probability = kwargs['metrics_probability'] if 'metrics_probability' in kwargs else True

	#get the most likely classes from the classifier
	best_classes = fp.similarity_classifier_knn(database,sample_feature_space,nncl,
		first_values=FIRST_VALUES,flatten=FLATTEN_PROBABILITY,
		function=CLASSIFIER_FUNCTION)

	#try this new version of a dynamical filter. If the possible cluster is too 
	#far away from the last position estimate, remove it from the list.
	if DYNAMICAL_FILTER_ON and pf_store_particles.empty == False:

		#remove low-density particle noise
		count_discard = 0
		counts = pf_store_particles['cluster'].value_counts(normalize=True)
		drop_clusters = counts.index[counts<MIN_OCCURR_FRAC].tolist()
		for i, line in pf_store_particles.iterrows():
			if(line['cluster'] in drop_clusters):
				pf_store_particles.drop(index=i,inplace=True)

		#Distance list, discard the points the furthest away from the last point.
		dist_idx_list = []
		for idx, cluster in best_classes.iterrows():
			cldist = []			
			for idy, old_cluster in pf_store_clusters.iterrows():
				cldist.append(geopy.distance.vincenty((cluster['Lat'],cluster['Lon']),
					(old_cluster['Lat'],old_cluster['Lon'])).km*1000)
			#take minimum distance to old cluster
			dist_idx_list.append((np.min(cldist),idx))
		#feed into pandas dataframe
		dist_idx_list_pd = pd.DataFrame(data=dist_idx_list,columns=['dist','index'])
		#take the  highest distances
		dist_idx_list_pd = dist_idx_list_pd.sort_values(by='dist',ascending=False).reset_index(drop=True)

		#drop a fraction of clusters which are too far away from the 
		#previous measurement.
		n = 0
		while True:
			clusters_cut = dist_idx_list_pd.loc[n:n]

			#cut the clusters which are the furthest away
			for idx, clcut in clusters_cut.iterrows():
				best_classes.drop(index=clcut['index'],inplace=True)
			if (np.sum(best_classes['Probability'].tolist())) < PARTICLES_KEEP_FRAC: break
			n += 1

	#for every cluster, sample p*N_SAMPLE points with random position inside cluster
	particles = []
	for idx, line in best_classes.iterrows():
		if metrics_probability:
			nb_particles = int(round(line.loc['Probability']*N_SAMPLE))
		else:
			nb_particles = int(round(line.loc['Mean Similarity']*N_SAMPLE))
		#print("Cluster {}, Generating {} particles".format(idx,nb_particles))
		for p in range(nb_particles):
			lat, lon = get_random_position(line.loc['Lat'],line.loc['Lon'],CLUSTER_R)
			particles.append((lat,lon,0,line.loc['Cluster ID'],
				line.loc['Lat'],line.loc['Lon']))
	new_particles = pd.DataFrame(data=particles,columns=['lat','lon','age',
		'cluster','clat','clon'])
	if VERBOSE: print("New particles: {}".format(len(new_particles)))
	new_clusters = best_classes.drop(['Cluster ID','Variance','Mean Similarity'],
		axis=1)
	new_clusters['age']=0
	if VERBOSE: print("Historical particles: {}".format(len(pf_store_particles)))

	
	#Sample the historical values
	if pf_store_particles.empty == False:
		pf_store_particles['age'] = pf_store_particles['age']+1
		pf_store_clusters['age'] = pf_store_clusters['age']+1

		#remove old points
		pf_store_particles = pf_store_particles.loc[pf_store_particles['age']<=MAX_AGE+1]
		pf_store_clusters = pf_store_clusters.loc[pf_store_clusters['age']<=FILTER_AGE-1]

		#resample historical data --> density of particles representing probability
		#remove pre-defined proportion of particles
		pf_store_particles = pf_store_particles.sample(frac=1).reset_index(drop=True).loc[:int(pf_store_particles.shape[0]*(1-DISCARD))-1,:]

		#increase past radius according to device velocity
		for i, particle in pf_store_particles.iterrows():
			lat_dynamic, lon_dynamic = get_random_position(particle.loc['clat'],
				particle.loc['clon'],CLUSTER_R+F_SAMPLING*SPEED*age)
			pf_store_particles['lat'] = lat_dynamic
			pf_store_particles['lon'] = lon_dynamic

	#Store historical data of the model
	pf_store_clusters = pf_store_clusters.append(new_clusters,ignore_index=True)
	pf_store_particles = pf_store_particles.append(new_particles,ignore_index=True)

	mp.print_particles(pf_store_particles,"t = {}".format(-1*age),
		real_pos,heatmap=True,particles=False)
	return pf_store_particles

#Returns the mean particle error metrics
def get_mean_error(pd_distribution, real_pos):
	"""This function evaluates the mean error between a particle distribution 
	and a real ground-truth position

    Args:
        pd_distribution (pandas):	Particle distribution to evaluate
		real_pos (tuple): 			Tuple of floats (lat,lon). Real position of 
									the evaluation

    Returns:
        float: Distance error in meters (mean particle error)
    """
	distances = []
	lat = []
	lon = []
	for idx, particle in pd_distribution.iterrows():
		distances.append(geopy.distance.vincenty((particle['lat'],particle['lon']),
			real_pos).km*1000)
		lat.append(particle['lat'])
		lon.append(particle['lon'])
	return np.mean(distances),geopy.distance.vincenty((np.mean(lat),np.mean(lon)),
		real_pos).km*1000

def get_position_estimate(pd_distribution):
	"""This function estimates the position from a particle distribution

	It is a very basic start and calculates only the mean between all the particles.
	Todo: Rule if some particle clusters are far away, rule for older particles, etc

    Args:
        pd_distribution (pandas):	Particle distribution to evaluate

    Returns:
        tuple: (lat,lon): Estimated position
    """
	lat = []
	lon = []
	for idx, particle in pd_distribution.iterrows():
		lat.append(particle['lat'])
		lon.append(particle['lon'])
	return np.mean(lat),np.mean(lon)



