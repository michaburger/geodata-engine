import numpy as np
import pandas as pd
import fingerprinting as fp
import geopy.distance
import mapping as mp
import random
import math
import datetime

N_SAMPLE = 500
CLUSTER_R = 30
SPEED = 1.0 #m/s
F_SAMPLING = 60 #seconds between 2 transmissions
DISCARD = 1.0 #historical discard for particles older than t=-1
MAX_AGE = 5 #discard particles older than this
FILTER_AGE = 1 #number of historical values to be used for cluster filtering
DYNAMICAL_FILTER_ON = True
MIN_OCCURR_FRAC = 0.1 #minimum occurrency of a certain cluster in the historical particles. clusters with less occurrency will be deleted for noise.
MIN_NEW_PARTICLES_FRAC = 0.2 #minimum fraction of new particles that have to be in the prediction. When less particles are available (due to filtering / wrong prediction) use position of the last transmission
FLATTEN_PROBABILITY = 1.5 #take n-root after the min-max probability calculation
FIRST_VALUES = 10 #how many of the first guesses to consider
PARTICLES_KEEP_FRAC = 0.5 #dynamical filter: cut off the N clusters furthest away from historical clusters, but keep a certain proportion of particles in all the cases
CLASSIFIER_FUNCTION = 'euclidean' #euclidean, cosine, manhattan or correlation

VERBOSE = 0 # 0 1 or 2

pf_store_particles = pd.DataFrame(columns=['lat','lon','age','cluster','clat','clon'])
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

#Version 2, simplified to understand and debug
def get_particle_distribution(sample_feature_space,database,nncl,age,real_pos,**kwargs):
	global pf_store_particles
	global pf_store_clusters
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	metrics_probability = kwargs['metrics_probability'] if 'metrics_probability' in kwargs else True

	#get the most likely classes from the classifier
	best_classes = fp.similarity_classifier_knn(database,sample_feature_space,nncl,first_values=FIRST_VALUES,flatten=FLATTEN_PROBABILITY,function=CLASSIFIER_FUNCTION)

	#try this new version of a dynamical filter. If the possible cluster is too far away from the last position estimate, remove it from the list.
	if DYNAMICAL_FILTER_ON and pf_store_particles.empty == False:

		#remove low-density particle noise
		count_discard = 0
		counts = pf_store_particles['cluster'].value_counts(normalize=True)
		drop_clusters = counts.index[counts<MIN_OCCURR_FRAC].tolist()
		for i, line in pf_store_particles.iterrows():
			if(line['cluster'] in drop_clusters):
				pf_store_particles.drop(index=i,inplace=True)

		#Distance list, discard the points the furthest away from the last transmission.
		dist_idx_list = []
		for idx, cluster in best_classes.iterrows():
			cldist = []			
			for idy, old_cluster in pf_store_clusters.iterrows():
				cldist.append(geopy.distance.vincenty((cluster['Lat'],cluster['Lon']),(old_cluster['Lat'],old_cluster['Lon'])).km*1000)
			#take minimum distance to old cluster
			dist_idx_list.append((np.min(cldist),idx))
		#feed into pandas dataframe
		dist_idx_list_pd = pd.DataFrame(data=dist_idx_list,columns=['dist','index'])
		#take the  highest distances
		dist_idx_list_pd = dist_idx_list_pd.sort_values(by='dist',ascending=False).reset_index(drop=True)

		#drop a fraction of clusters which are too far away from the previous measurement.
		n = 0
		while True:
			clusters_cut = dist_idx_list_pd.loc[n:n]

			#cut the clusters which are the furthest away
			for idx, clcut in clusters_cut.iterrows():
				best_classes.drop(index=clcut['index'],inplace=True)
			if (np.sum(best_classes['Probability'].tolist())) < PARTICLES_KEEP_FRAC: break
			n += 1

		'''
		#look at the dispersion of the clusters
		lat = []
		lon = []
		for idx, clss in best_classes.iterrows():
			lat.append(clss['Lat'])
			lon.append(clss['Lon'])
		dispersion = 5000*(np.std(lat)+np.std(lon))
		print("cluster dispersion (std deviation): {}".format(dispersion))
		'''

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
			particles.append((lat,lon,0,line.loc['Cluster ID'],line.loc['Lat'],line.loc['Lon']))
	new_particles = pd.DataFrame(data=particles,columns=['lat','lon','age','cluster','clat','clon'])
	print("New particles: {}".format(len(new_particles)))
	new_clusters = best_classes.drop(['Cluster ID','Variance','Mean Similarity'],axis=1)
	new_clusters['age']=0
	print("Historical particles: {}".format(len(pf_store_particles)))

	
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
			lat_dynamic, lon_dynamic = get_random_position(particle.loc['clat'],particle.loc['clon'],CLUSTER_R+F_SAMPLING*SPEED*age)
			pf_store_particles['lat'] = lat_dynamic
			pf_store_particles['lon'] = lon_dynamic

	#Store historical data of the model
	pf_store_clusters = pf_store_clusters.append(new_clusters,ignore_index=True)
	pf_store_particles = pf_store_particles.append(new_particles,ignore_index=True)

	mp.print_particles(pf_store_particles,"t = {}".format(-1*age),real_pos,heatmap=True,particles=False)
	return pf_store_particles

#particle distribution with dynamical filter, trials
def get_particle_distribution_deprecated(sample_feature_space,database,nncl,age,real_pos,**kwargs):
	global pf_store_particles
	global pf_store_clusters
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	metrics_probability = kwargs['metrics_probability'] if 'metrics_probability' in kwargs else True
	best_classes = fp.similarity_classifier_knn(database,sample_feature_space,nncl,first_values=FIRST_VALUES,flatten=FLATTEN_PROBABILITY,function=CLASSIFIER_FUNCTION)
	#print(best_classes)
	#print(pf_store_clusters)

	#resampling filter: remove new particles with impossible positions (too far away)
	if DYNAMICAL_FILTER_ON and pf_store_clusters.empty == False:

		#remove low-density particle noise
		count_discard = 0
		counts = pf_store_particles['cluster'].value_counts(normalize=True,dropna=False)
		drop_clusters = counts.index[counts<MIN_OCCURR_FRAC].tolist()
		for i, line in pf_store_particles.iterrows():
			if(line['cluster'] in drop_clusters):
				pf_store_particles.drop(index=i,inplace=True)

		for idx, cluster in best_classes.iterrows():			
			discard = True
			for idy, old_cluster in pf_store_clusters.iterrows():
				distance = geopy.distance.vincenty((cluster['Lat'],cluster['Lon']),(old_cluster['Lat'],old_cluster['Lon'])).km*1000
				if distance < CLUSTER_R+F_SAMPLING*SPEED*old_cluster['age']:
					discard = False
			#discard row if too far away from previous measures
			if (discard):
				best_classes.drop(index=idx,inplace=True)
				count_discard += 1	
				#print("Cluster {} dropped".format(idx))
		print("{} clusters out of {} dropped".format(count_discard,FIRST_VALUES))
	print("Historical particles: {}".format(len(pf_store_particles)))

	#number of particles transferred from the last time, before removal by filtering algorithm.
	nb_historical = len(pf_store_particles)

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
			particles.append((lat,lon,0,line.loc['Cluster ID'],line.loc['Lat'],line.loc['Lon']))
	new_particles = pd.DataFrame(data=particles,columns=['lat','lon','age','cluster','clat','clon'])
	print("New particles: {}".format(len(new_particles)))
	new_clusters = best_classes.drop(['Cluster ID','Variance','Mean Similarity'],axis=1)
	new_clusters['age']=0


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
			lat_dynamic, lon_dynamic = get_random_position(particle.loc['clat'],particle.loc['clon'],CLUSTER_R+F_SAMPLING*SPEED*age)
			pf_store_particles['lat'] = lat_dynamic
			pf_store_particles['lon'] = lon_dynamic


	#In case of a wrong prediction (too many clusters dropped due to impossible position)
	if len(new_particles) < nb_historical*MIN_NEW_PARTICLES_FRAC:
		#use previous estimation
		old_particles_revamp = pf_store_particles
		old_particles_revamp['age'] = old_particles_revamp['age']-1

		old_clusters_revamp = pf_store_clusters
		old_clusters_revamp['age'] = old_clusters_revamp['age']-1

		#Resampling to have at least N_SAMPLE particles, to avoid ending up with 0 particles
		nnn = int(N_SAMPLE/len(pf_store_particles))
		for i in range(nnn):
			pf_store_particles = pf_store_particles.append(old_particles_revamp,ignore_index=True)

		pf_store_clusters = pf_store_clusters.append(old_clusters_revamp,ignore_index=True)
		print("Debug info: Old values retaken")

		
	else:
		#Store historical data of the model
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

def get_position_estimate(pd_distribution):
	lat = []
	lon = []
	for idx, particle in pd_distribution.iterrows():
		lat.append(particle['lat'])
		lon.append(particle['lon'])
	return np.mean(lat),np.mean(lon)



