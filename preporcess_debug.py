#%%

import json 
import numpy as np
import geopandas as gpd
import pandas as pd

from shapely import wkt
from shapely.geometry import Polygon,LineString

def xbd2gdf(f_in):
    """
    Convert an xBD json vector file to a geo-dataframe and calculates
    the angle, major, and minor axis of the minimum oriented bounding box
    of each polygon in the file.

    Args:
        f_in (str): The path to the xBD json file

    Returns:
        gdf (geopandas.GeoDataFrame): The geo-dataframe containing the polygons
        (with the uid, damage, and geometry columns from the xBD json file)
        and the calculated **angle, major, and minor axis** of the minimum oriented
        bounding box of each polygon.
    """

    # adapted from https://github.com/DIUx-xView/xView2_baseline/blob/master/utils/view_polygons.ipynb
    with open(f_in, 'rb') as image_json_file:
        image_json = json.load(image_json_file)

    # coords can be in xy or in lng_lat format 
    coords = image_json['features']['lng_lat']
    coords_xy = image_json['features']['xy']

    wkt_polygons = []

    for coord in coords:
        if 'subtype' in coord['properties']:
            damage = coord['properties']['subtype']
        else:
            damage = 'no-damage'

        uid = coord['properties']['uid']
        wkt_polygons.append((uid, damage, coord['wkt']))

    wkt_polygons_xy = []

    for coord in coords_xy:
        if 'subtype' in coord['properties']:
            damage = coord['properties']['subtype']
        else:
            damage = 'no-damage'

        uid = coord['properties']['uid']
        wkt_polygons_xy.append((uid, damage, coord['wkt']))

    polygons = []

    for uid, damage, swkt in wkt_polygons:
        polygons.append((uid, damage, wkt.loads(swkt), ))

    polygons_xy = []
    
    for uid, damage, swkt in wkt_polygons_xy:
        polygons_xy.append((uid, damage, wkt.loads(swkt), ))

    # adapted from https://gis.stackexchange.com/questions/380499/calculate-azimuth-from-polygon-in-geopandas
    def get_angle(poly, mode='degrees'):
        # g = poly.geometry
        a = poly.minimum_rotated_rectangle
        l = a.boundary
        coords = [c for c in l.coords]
        segments = [LineString([a, b]) for a, b in zip(coords,coords[1:])]
        longest_segment = max(segments, key=lambda x: x.length)
        shortest_segment = min(segments, key=lambda x: x.length)
        p1, p2 = [c for c in longest_segment.coords]

        if mode=='degrees':
            a = np.degrees(np.arctan2(p2[1]-p1[1], p2[0]-p1[0]))
        else:
            a = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])

        # REPLACE THIS LINE
        return {'angle': a, 'longest_segment': longest_segment.length/2, 'shortest_segment': shortest_segment.length/2}
        
    gdf = gpd.GeoDataFrame(polygons, columns=['uid', 'damage', 'geometry'], geometry='geometry', crs='EPSG:4326')
    gdf = gdf.to_crs('EPSG:3395')

    gdf_xy = pd.DataFrame(polygons_xy, columns=['uid', 'damage', 'geometry'])
    
    gdf[['angle','major', 'minor']] = gdf.apply(lambda x: get_angle(x.geometry, mode='radians'), axis=1, result_type='expand')

    # inner join to remove the polygons that are not in the xy format
    gdf = gdf.merge(gdf_xy, on='uid', how='inner', suffixes=('', '_xy'))

    gdf = gpd.GeoDataFrame(gdf, columns=['uid', 'damage', 'geometry', 'angle', 'major', 'minor', 'geometry_xy'], geometry='geometry', crs='EPSG:3395')
    # print(gdf.head())
    return gdf

def rasterize_poly(gdf, px=128, gsd=0.5):
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    """
    Rasterize the polygons in the geo-dataframe

    Args:
        gdf (geopandas.GeoDataFrame): The geo-dataframe containing the polygons
        px (int): The number of pixels in the raster
        gsd (float): The ground sample distance of the raster (in map units) i.e. the size of the pixel in linear map units

    Returns:
        raster (numpy.ndarray): The rasterized polygons
    """

    xy = gdf.geometry.centroid
    x, y = xy.x.item(), xy.y.item()
    
    r = gsd*px/2
    xmin, ymin, xmax, ymax = x-r, y-r, x+r, y+r

    transform = from_bounds(xmin, ymin, xmax, ymax, px, px)
    
    # BUG: gdf.geometry.item() does not work with pd.apply when implemented downstream (see main method) but it works when using a loop with DataFrame.iterrows()

    # raster = rasterize([(gdf.geometry.item(), 1),], out_shape=(px, px), fill=0, all_touched=True, transform=transform)
    raster = rasterize([(gdf.geometry, 1),], out_shape=(px, px), fill=0, all_touched=True, transform=transform)
    return raster, transform

def draw_ellipsoid(c, M, m, a):
    """
    Draw an ellipsoid

    Args:
        c (tuple): The center of the ellipsoid
        M (float): The major axis of the ellipsoid
        m (float): The minor axis of the ellipsoid
        a (float): The angle of the ellipsoid

    Returns:
        ellipsoid (shapely.geometry.Polygon): The ellipsoid
    """
    from shapely.geometry import Polygon

    t = np.linspace(0, 2*np.pi, 100)
    x = c[0] + M*np.cos(t)*np.cos(a) - m*np.sin(t)*np.sin(a)
    y = c[1] + M*np.cos(t)*np.sin(a) + m*np.sin(t)*np.cos(a)
    ellipsoid = Polygon(zip(x, y))
    return ellipsoid

def poly2density(gdf, kernel, px=128, mode='centroid'):
    ''' 
    Calculate the kernel density of the polygons in a geo-dataframe.
    '''

    from scipy.signal import convolve2d
    from rasterio.transform import from_bounds

    if mode not in ['centroid', 'poly', 'ellipsoid']:
        raise ValueError('mode must be "centroid", "poly", or "ellipsoid"')
    
    

    if mode == 'centroid':
        xy = gdf.geometry.centroid
        x, y = xy.x, xy.y

        empty_grid = np.zeros((px,px))
        i = int(px/2)
        xmin, ymin, xmax, ymax = x-i, y-i, x+i, y+i

        empty_grid[i, i] = 1
        empty_grid[i-1, i] = 1
        empty_grid[i-1, i-1] = 1
        empty_grid[i, i-1] = 1
        
        r = empty_grid
        t = from_bounds(xmin, ymin, xmax, ymax, px, px)

    
    if mode == 'poly':
        # next line may be necessary if gdf is a pandas.Series object
        # g = gpd.GeoDataFrame([gdf.iloc[i]], columns=gdf.columns, geometry='geometry', crs='crs')
        r, t = rasterize_poly(gdf, px=px)

    if mode == 'ellipsoid':

        a, M, m = gdf.angle.item(), gdf.major.item(), gdf.minor.item()

        xy = gdf.geometry.centroid
        x, y = xy.x.item(), xy.y.item()

        # a, M, m = gdf.angle, gdf.major.length/2, gdf.minor.length/2
        # a, M, m = gdf.angle.item(), gdf.major.item().length/2, gdf.minor.item().length/2

        # xy = gdf.geometry.centroid
        # x, y = xy.x.item(), xy.y.item()
        
        e = draw_ellipsoid((x,y), M, m, a)
        r, t = rasterize_poly(gpd.GeoDataFrame({'geometry': [e]}, crs=gdf.crs), px=px)
        
    r = convolve2d(r, kernel, mode='same', boundary='fill', fillvalue=0)
        
    return r, t

def io_wrap(name, r, _, target='./temp_densities'):
    """
    Save the raster to a file

    Args:
        name (str): The name of the file
        r (numpy.ndarray): The raster
        target (str): The target directory
    """
    import matplotlib.image
    import os

    if not os.path.exists(target):
        os.makedirs(target)

    matplotlib.image.imsave(f"{target}/{name}.png", r, cmap='gray')

def rgb_io_wrap(name, r, target='./temp_rgb'):
    """
    Save the raster to a file

    Args:
        name (str): The name of the file
        r (numpy.ndarray): The raster
        target (str): The target directory
    """
    import matplotlib.image
    import os

    if not os.path.exists(target):
        os.makedirs(target)

    matplotlib.image.imsave(f"{target}/{name}.png", r, cmap='gray')

def poly2rgbClip(gdf, rgb, px=128):

    name = gdf.uid
    print(gdf.geometry_xy)
    xy = gdf.geometry_xy.centroid
    
    x, y = xy.x.item(), xy.y.item()

    r = px/2
    xmin, ymin, xmax, ymax = x-r, y-r, x+r, y+r

    rgb_slice = rgb[int(ymin):int(ymax), int(xmin):int(xmax), :] 
    return name, rgb_slice


# Kernel functions. Can be customized with different functions
# check https://en.wikipedia.org/wiki/Kernel_(statistics)

def gaussianKernel(x):
    return np.divide(1, 1*np.sqrt(2*np.pi))*np.exp(-0.5*(np.divide(x, 1))**2)

def epanechnikovKernel(x):
    return np.where(np.abs(x) <= 1, np.divide(3, 4*1)*(1-np.divide(x, 1)**2), 0)
     

def tricubeKernel(x):
    return np.divide(70, 81)*np.power(1-np.abs(np.power(x,3)),3)


def generate_kernel(r, f, limit=1):
    """
    Generate a 2D kernel for a kernel density estimation

    Args:
        r (int): The resolution of the kernel
        f (function): The kernel function
        limit (float): The domain of the kernel function (x,y) most kernels are defined in the range [-1,1] only. Adjust r to change the size of the filter.

    Returns:
        xx*yy (numpy.ndarray): The 2D kernel
    """
    x, y = np.linspace(-limit, limit, r), np.linspace(-limit, limit, r)
    # x, y = np.arange(m-r, m+r+1, 1), np.arange(m-r, m+r+1, 1)

    xx, yy = f(np.meshgrid(x, y))
    return xx*yy
    
if __name__ == "__main__":

    import os

    gk = generate_kernel(15,gaussianKernel)
    ek = generate_kernel(15,epanechnikovKernel)
    tk = generate_kernel(15,tricubeKernel)

    import matplotlib.pyplot as plt
    import geopandas as gpd

    f_in = './data/woolsey-fire_00000645_post_disaster.json'
    img_in = './data/woolsey-fire_00000645_post_disaster.png'

    gdf = xbd2gdf(f_in)
    gdf = gdf.to_crs('EPSG:3395')

    img = plt.imread(img_in)

    os.makedirs('./temp', exist_ok=True)
    gdf.apply(lambda x: io_wrap(x.uid, *poly2density(x, gk, px=128, mode='poly')), axis=1)
    gdf.apply(lambda x: rgb_io_wrap(*poly2rgbClip(x, img, px=128)), axis=1)

# %%
