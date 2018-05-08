import urllib.request
from urllib.error import URLError, HTTPError
import datetime as dt
import json
import time

track_query_url = "https://spaghetti.scapp.io/query?track="
gateway_query_url = "https://spaghetti.scapp.io/gateways"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
starttime = dt.datetime.now() - dt.timedelta(days=365)


def request_track(track,txpow=0,sf=7,dev='78AF580300000485',hdop=500,start="2018-01-01_00:00:00",end=str(dt.datetime.now().strftime(TIME_FORMAT))):
	url = track_query_url + str(track) + "&start=" + start + "&end=" + end + "&sf=" + str(sf) + "&txpow=" + str(txpow) + "&device=" + str(dev) + "&hdop=" + str(hdop)
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

def request_track_no_params(track,start="2018-01-01_00:00:00",end=str(dt.datetime.now().strftime(TIME_FORMAT))):
	url = track_query_url + str(track) + "&start=" + start + "&end=" + end
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
