from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

#map = Basemap(projection='merc', lat_0=57, lon_0=-135,resolution = 'h', area_thresh = 0.1,llcrnrlon = -136.25, llcrnrlat = 56.0,urcrnrlon = -134.25, urcrnrlat = 57.75)
# Lambert Conformal map of lower 48 states.
map = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

map.drawcoastlines()
map.drawcountries()
map.fillcontinents(color='coral')
map.drawmapboundary()


lons = [-95.94043, -74.00597, -122.33207]
lats = [41.2562, 40.71427, 47.60621]
x, y = map(lons, lats)
map.plot(x, y, 'bo', markersize=10)

labels = ['Omaha', 'Nova York', 'Seattle']
for label, xpt, ypt in zip(labels, x, y):
    plt.text(xpt, ypt, label)



plt.show()