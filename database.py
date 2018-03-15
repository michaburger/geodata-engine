import urllib.request
import datetime as dt
import json

track_query_url = "https://spaghetti.scapp.io/query?track="
gateway_query_url = "https://spaghetti.scapp.io/gateways"
TIME_FORMAT = "%Y-%m-%d_%H:%M:%S"
starttime = dt.datetime.now() - dt.timedelta(days=365)


def request_track(track,start="2018-01-01_00:00:00",end=str(dt.datetime.now().strftime(TIME_FORMAT))):
	url = track_query_url + str(track) + "&start=" + start + "&end=" + end
	req = urllib.request.Request(url)
	r = urllib.request.urlopen(req).read()
	return r

def request_gateways(rad_km=250,lat=46.52,lon=6.56):
	rad_m = rad_km * 1000
	url = gateway_query_url + '?lat=' + str(lat) + '&lon=' + str(lon) + '&radius=' + str(rad_m)
	req = urllib.request.Request(url)
	r = urllib.request.urlopen(req).read()
	return r

def add_gateway(eui,lat,lon):
	url = gateway_query_url + '?id=' + eui + '&lat=' + lat + '&lon=' + lon
	req = urllib.request.Request(url,{})
	r = urllib.request.urlopen(req).read()
	return r
