import numpy as np
import pandas as pd
import fingerprinting as fp
import mapping as mp
import random
import math

N_SAMPLE = 100
CLUSTER_R = 60
SPEED = 0
DISCARD = 0.3 #historical discard

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

def create_artificial_time_series():
	print(0)

def get_particle_distribution(sample_feature_space,database,nncl,**kwargs):
	render_map = kwargs['render_map'] if 'render_map' in kwargs else False 
	best_classes = fp.cosine_similarity_classifier_knn(database,sample_feature_space,nncl,first_values=20)
	global pf_store
	particles = []
	#for every cluster, sample p*N_SAMPLE points with random position inside cluster
	for idx, line in best_classes.iterrows():
		nb_particles = int(round(line.loc['Probability']*N_SAMPLE))
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
	print(pf_store)

	if render_map:
		mp.print_particles(pf_store)
		print("Particle map rendered!")


	#TODO for dynamical algorithm: increase cluster size if speed != 0

	#print current particle distribution for double check



