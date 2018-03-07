import folium
from folium.plugins import HeatMap
import pandas
import urllib.request
import json
import datetime


track_query_url = "https://spaghetti.scapp.io/query?track="
gateway_query_url = "https://spaghetti.scapp.io/gateways"

#request json for track
def requestTrack(track):
	url = track_query_url + str(track)
	req = urllib.request.Request(url)
	r = urllib.request.urlopen(req).read()
	return r

def requestGateways():
	req = urllib.request.Request(gateway_query_url)
	r = urllib.request.urlopen(req).read()
	return r


#gateway list as python object (dict)
gatewayList = json.loads(requestGateways().decode('utf-8'))

gtw_id=[]
gtw_lat=[]
gtw_lon=[]

#count: remove index from gtw and put it in count
for count, gtw in enumerate(gatewayList):
	gtw_id.append(gtw['gateway_id'])
	gtw_lon.append(gtw['gateway_lon'])
	gtw_lat.append(gtw['gateway_lat'])

map = folium.Map(location=[46.52, 6.565],zoom_start=14)#,tiles='Stamen Toner')
fgtw = folium.FeatureGroup(name='Gateways')

for gtw_id,gtw_lat,gtw_lon in zip(gtw_id,gtw_lat,gtw_lon):
	fgtw.add_child(folium.Marker(location=[gtw_lat,gtw_lon],popup="ID: "+str(gtw_id),icon=folium.Icon(color='darkblue',prefix='fa',icon='rss')))

#add points of track 1
track1 = json.loads(requestTrack(1).decode('utf-8'))
#print(track1)

lat=[]
lon=[]
time=[]
dev=[]
hum=[]
temp=[]
sp=[]
gps_sat=[]
gps_hdop=[]
gateways=[]
rssi=[]
snr=[]
esp=[]
heat=[]

for count, trk in enumerate(track1):
	#only consider points with at least 5 satellites
	if(trk['gps_sat']>4):
		lat.append(trk['gps_lat'])
		lon.append(trk['gps_lon'])
		time.append(trk['timestamp']['$date']) #todo: format time
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
		if(trk['gateway_id'][0] == '0B030153'):
			heat.append(trk['gateway_esp'][0])
		else:
			heat.append(0)

def color_producer(heat):
    if heat< -100:
        return 'orange'
    elif -100<=heat<0:
        return 'green'
    else:
        return 'red'

ftr1 = folium.FeatureGroup(name='Track1')
for lat,lon,time,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat in zip(lat,lon,time,dev,hum,temp,sp,gps_sat,gps_hdop,gateways,rssi,snr,esp,heat):
	ftr1.add_child(folium.CircleMarker(location=[lat,lon],
		fill=True,radius=20,
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
		+ "<b>Easteregg: </b>Dinosaur<br/>",
		color='',
		fill_color=color_producer(heat),
		fill_opacity=0.7))


#fhmap = folium.FeatureGroup(name='Heatmap')
#for lat,lon,time,dev,hum,heat in zip(lat,lon,time,dev,hum,heat):
#	fhmap.add_child(HeatMap([lat,lon],radius=25,gradient={.4: 'blue', .65: 'lime', 1:'red'}))


#generate map
map.add_child(fgtw)
map.add_child(ftr1)
#map.add_child(fhmap)
map.add_child(folium.LayerControl())
map.save("Map1.html")

'''
data=pandas.read_csv("Volcanoes.txt")
lat=list(data['LAT'])
lon=list(data['LON'])
elev=list(data['ELEV'])

def color_producer(elevation):
    if elevation<1000:
        return 'green'
    elif 1000<=elevation<3000:
        return 'orange'
    else:
        return 'red'

for lt,ln,el in zip(lat,lon,elev):
    fgv.add_child(folium.CircleMarker(location=[lt,ln],fill=True,radius=6,popup=str(el)+" m",color='grey',fill_color=color_producer(el),fill_opacity=0.7))
fgp=folium.FeatureGroup(name="Population")

fgp.add_child(folium.GeoJson(data=open('world.json','r',encoding='utf-8-sig').read(),style_function=lambda x: {'fillColor':'green' if x['properties']['POP2005']<10000000 else 'orange' if 10000000 <=x['properties']['POP2005'] < 20000000 else 'red'}))
map.add_child(fgv)
map.add_child(fgp)
map.add_child(folium.LayerControl())
map.save("Map1.html")

'''