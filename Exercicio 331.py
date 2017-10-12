from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os as os
from sklearn import manifold
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# draw state boundaries.
# data from U.S Census Bureau
# http://www.census.gov/geo/www/cob/st2000.html
os.chdir("C:/Users/hespo/OneDrive/Documentos/GitHub/Trab2_DataScience")
cwd = os.getcwd()
print (cwd)
# Distance file available from RMDS project:
#    https://github.com/cheind/rmds/blob/master/examples/european_city_distances.csv
reader = csv.reader(open("city_distances.csv", "r"), delimiter=';')
data = list(reader)

dists = []
cities = []
for d in data:
    cities.append(d[0])
    dists.append(map(float , d[1:]))

adist = np.array(dists)
print (adist.shape)
amax = np.amax(adist)
adist /= amax
print (adist.shape)

mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
results = mds.fit(adist)

#coords = results.embedding_ * 100
coords_aux = results.embedding_

fatlat = 856.6217032112617
fatlong= 4665.870709846797
coords=np.zeros((20,2))
cont = 0
for i in coords_aux:
    lat_global = i[0]*fatlat
    long_global = i[1]*fatlong
    print (cont,"x global:"+ str(lat_global))
    print ("y global:" + str(long_global))
    coords[cont][0] = lat_global
    coords[cont][1] = long_global
    cont = cont + 1


#plt.subplots_adjust(bottom = 0.1)
#plt.scatter(
 #    coords[:, 1],  coords[:, 0], marker = 'o'
  #  )
#for label, x, y in zip(cities, coords[:, 1],  coords[:, 0]):
#    plt.annotate(
#        label,
#        xy = (x, y), xytext = (-20, 20),
#        textcoords = 'offset points', ha = 'right', va = 'bottom',
#        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

lons = [-95.94043, -74.00597, -122.33207]
lats = [41.2562, 40.71427, 47.60621]


coords[0][0] = 41.2562
coords[0][1] = -95.94043
x, y = m(coords[:, 1], coords[:, 0])
print("x e y ",x,y)
m.plot(x, y, 'bo', markersize=10)

labels = ['Omaha', 'Nova York', 'Seattle']
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt, ypt, label)

#plt.show()


shp_info = m.readshapefile('C:/Users/hespo/OneDrive/Documentos/GitHub/Trab2_DataScience/st99_d00','states',drawbounds=True)
# population density by state from
# http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
popdensity = {
'New Jersey':  438.00,
'Rhode Island':   387.35,
'Massachusetts':   312.68,
'Connecticut':	  271.40,
'Maryland':   209.23,
'New York':    155.18,
'Delaware':    154.87,
'Florida':     114.43,
'Ohio':	 107.05,
'Pennsylvania':	 105.80,
'Illinois':    86.27,
'California':  83.85,
'Hawaii':  72.83,
'Virginia':    69.03,
'Michigan':    67.55,
'Indiana':    65.46,
'North Carolina':  63.80,
'Georgia':     54.59,
'Tennessee':   53.29,
'New Hampshire':   53.20,
'South Carolina':  51.45,
'Louisiana':   39.61,
'Kentucky':   39.28,
'Wisconsin':  38.13,
'Washington':  34.20,
'Alabama':     33.84,
'Missouri':    31.36,
'Texas':   30.75,
'West Virginia':   29.00,
'Vermont':     25.41,
'Minnesota':  23.86,
'Mississippi':	 23.42,
'Iowa':	 20.22,
'Arkansas':    19.82,
'Oklahoma':    19.40,
'Arizona':     17.43,
'Colorado':    16.01,
'Maine':  15.95,
'Oregon':  13.76,
'Kansas':  12.69,
'Utah':	 10.50,
'Nebraska':    8.60,
'Nevada':  7.03,
'Idaho':   6.04,
'New Mexico':  5.79,
'South Dakota':	 3.84,
'North Dakota':	 3.59,
'Montana':     2.39,
'Wyoming':      1.96,
'Alaska':     0.42}
print(shp_info)
# choose a color for each state based on population density.
colors={}
statenames=[]
cmap = plt.cm.hot # use 'hot' colormap
vmin = 0; vmax = 450 # set range.
print(m.states_info[0].keys())
for shapedict in m.states_info:
    statename = shapedict['NAME']
    # skip DC and Puerto Rico.
    if statename not in ['District of Columbia','Puerto Rico']:
        pop = popdensity[statename]
        # calling colormap with value between 0 and 1 returns
        # rgba value.  Invert color range (hot colors are high
        # population), take sqrt root to spread out colors more.
        #colors[statename] = cmap(1.-np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
        colors[statename] = cmap(1.)[:3]
    statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
    # skip DC and Puerto Rico.
    if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
        color = rgb2hex(colors[statenames[nshape]])
        poly = Polygon(seg,facecolor=color,edgecolor=color)
        ax.add_patch(poly)
# draw meridians and parallels.
#m.drawparallels(np.arange(25,65,20),labels=[1,0,0,0])
#m.drawmeridians(np.arange(-120,-40,20),labels=[0,0,0,1])
plt.title('Filling State Polygons by Population Density')

plt.show()