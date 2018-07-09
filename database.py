import urllib.request
from urllib.error import URLError, HTTPError
import datetime as dt
import pandas as pd
import json
import time

track_query_url = "https://spaghetti.scapp.io/query?track="
gateway_query_url = "https://spaghetti.scapp.io/gateways"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
starttime = dt.datetime.now() - dt.timedelta(days=365)


def request_track(track,txpow=0,sf=7,dev='78AF580300000485',hdop=500,start="2018-01-01_00:00:00",end=str(dt.datetime.now().strftime(TIME_FORMAT))):
	if dev=='ALL':
		url = track_query_url + str(track) + "&start=" + start + "&end=" + end + "&sf=" + str(sf) + "&txpow=" + str(txpow) + "&hdop=" + str(hdop)
	else:
		url = track_query_url + str(track) + "&start=" + start + "&end=" + end + "&sf=" + str(sf) + "&txpow=" + str(txpow) + "&device=" + str(dev) + "&hdop=" + str(hdop)
	tries = 5
	for i in range(tries):
		try:
			req = urllib.request.Request(url)
			r = urllib.request.urlopen(req).read()
			return json.loads(r.decode('utf-8'))
		except HTTPError as e:
			print("WARNING: HTTPError in trial {} of total {} retries".format(i,tries))
			time.sleep(60)
		except URLError as e:
			print("WARNING: URLError in trial {} of total {} retries".format(i,tries))
			time.sleep(60)
	return "[]"

def request_track_no_params(track,start="2018-01-01_00:00:00",end=str(dt.datetime.now().strftime(TIME_FORMAT))):
	url = track_query_url + str(track) + "&start=" + start + "&end=" + end
	tries = 5
	for i in range(tries):
		try:
			req = urllib.request.Request(url)
			r = urllib.request.urlopen(req).read()
			return json.loads(r.decode('utf-8'))
		except HTTPError as e:
			print("WARNING: HTTPError in trial {} of total {} retries".format(i,tries))
			time.sleep(60)
		except URLError as e:
			print("WARNING: URLError in trial {} of total {} retries".format(i,tries))
			time.sleep(60)
	return "[]"

def request_gateways(rad_km=250,lat=46.52,lon=6.56):
	rad_m = rad_km * 1000
	url = gateway_query_url + '?lat=' + str(lat) + '&lon=' + str(lon) + '&radius=' + str(rad_m)
	for i in range(5):
		try:
			req = urllib.request.Request(url)
			r = urllib.request.urlopen(req).read()
			return json.loads(r.decode('utf-8'))
		except HTTPError as e:
			print("HTTPError in trial {}".format(i))
			time.sleep(60)
		except URLError as e:
			print("URLError in trial {}".format(i))
			time.sleep(60)
	return "[]"

def add_gateway(eui,lat,lon):
	url = gateway_query_url + '?id=' + eui + '&lat=' + lat + '&lon=' + lon
	for i in range(5):
		try:
			req = urllib.request.Request(url,{})
			r = urllib.request.urlopen(req).read()
			return json.loads(r.decode('utf-8'))
		except HTTPError as e:
			print("HTTPError in trial {}".format(i))
			time.sleep(60)
		except URLError as e:
			print("URLError in trial {}".format(i))
			time.sleep(60)
	return "[]"

#transform external data to python dict to imitate it came directly from the server
def transform_antwerp(data_pd):
	track_list = []
	NB_BASE_STATIONS = 68

	for idx, row in data_pd.iterrows():
		#only import SF7 for the moment
		if row["'SF'"] == 7:
			p_dict = {'humidity': 0.0, 'gps_speed': 0.0, 'devEUI': 'ANTWERP', 'track_ID': 0, 'temperature': 0.0, 'sub_band': 'n/a', 'tx_pow': 0, 'channel': 'n/a', 'gps_sat': 5, 'deviceType': 'ANTWERP', 'gps_course':0}
			p_dict['time'] = row["'RX Time'"][1:-1] #rip off the simple quotes
			p_dict['gps_hdop'] = int(100*row["'HDOP'"])
			p_dict['sp_fact'] = row["'SF'"]
			p_dict['gps_lat'] = row["'Latitude'"]
			p_dict['gps_lon'] = row["'Longitude'"]
			p_dict['timestamp'] = {'$date': 0}


			gtw_id = []
			rssi = []
			snr = []
			esp = []
			for bs in range(1,NB_BASE_STATIONS+1):
				if row["'BS {}'".format(bs)] > -200:
					rssi_val = row["'BS {}'".format(bs)]
					gtw_id.append("BS {}".format(bs))
					snr.append(0)
					rssi.append(rssi_val)
					esp.append(rssi_val)
			p_dict['gateway_rssi'] = rssi
			p_dict['gateway_snr'] = snr
			p_dict['gateway_esp'] = esp
			p_dict['gateway_id'] = gtw_id

			track_list.append(p_dict)
	return track_list


