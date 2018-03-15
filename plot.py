import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import json
from datetime import datetime

def time_graph_rssi(data, plot_title):
	bunch = json.loads(data.decode('utf-8'))
	LRRs = ['0B030153','080E0FF2','080E05AD','080E04C4','080E05AD','080E1006','080E0669','080E1007','080E0FF2','080E1005']

	time, temp, hum = ([] for i in range(3))

	#esp data for every gateway 
	esp = [[np.nan for x in range(len(bunch))] for y in range(len(LRRs))] 

	#print(bunch)
	for i, element in enumerate(bunch):
		time.append(datetime.strptime(element['time'],"%Y-%m-%dT%H:%M:%S.%f+01:00"))
		temp.append(element['temperature'])
		hum.append(element['humidity'])

		#attribute received signal strengths (gtw_data) to the fixed list (gtw_list) of gateways
		for idx, gtw_list in enumerate(LRRs):
			for idy, gtw_data in enumerate(element['gateway_id']):
				if(gtw_list == gtw_data):
					esp[idx][i] = element['gateway_esp'][idy]

	plt.figure(1)
	plt.subplot(211)
	plt.ylabel('°C / %RH')
	plt.legend()

	plt.title(plot_title)

	plt.plot(time, temp, label='temperature')
	plt.plot(time, hum, label='humidity')

	plt.subplot(212)
	#plot all Gateway data
	for idx in range(len(LRRs)):
		fill_gaps(esp[idx])
		np.power(10,esp[idx])
		plt.plot(time, esp[idx], label=LRRs[idx])


	plt.xlabel('Date')
	plt.ylabel('ESP (dB)')

	plt.legend()

	plt.show()

def fill_gaps(data):
	for i in range(1,len(data)):
		if(np.isnan(data[i])):
			data[i] = data[i-1]



