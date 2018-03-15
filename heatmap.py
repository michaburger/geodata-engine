import os
import folium

print(folium.__version__)

import numpy as np

data = (np.random.normal(size=(100, 3)) *
        np.array([[1, 1, 1]]) +
        np.array([[48, 5, 1]])).tolist()

print(data)

from folium.plugins import HeatMap

m = folium.Map([48., 5.], tiles='stamentoner', zoom_start=6)

HeatMap(data).add_to(m)

m.save('Heatmap.html')

m