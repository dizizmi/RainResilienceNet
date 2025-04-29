import ee
import os 
os.environ['OMP_NUM_THREADS'] = '1'
import geemap
import webbrowser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import requests 
from datetime import datetime, timedelta
import time

def load_singapore_boundary():
    return ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1") \
        .filter(ee.Filter.eq('ADM0_NAME', 'Singapore')) \
        .geometry()


#MODIS11A2 for Land Surface Temperature (LST) 8-day avg 2024 till PRESENT
def load_lst(singapore_boundary, start_date='2024-01-01', end_date='2025-04-20'):

    #load modis
    collection = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterDate(start_date, end_date) 
        
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError("No images found for the specified date range.")

    lst = collection.mean() \
        .select('LST_Day_1km') \
        .multiply(0.02).subtract(273.15) \
        .rename('LST')

    return lst.clip(singapore_boundary)

def lst_to_numpy(lst_ee, singapore_boundary, scale=1000):
    arr = geemap.ee_to_numpy(
        ee_object=lst_ee,
        region=singapore_boundary,
        scale=scale
    )

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr

def normalize_list_zscore(lst_array):
    mean_lst = np.nanmean(lst_array)
    std_lst = np.nanstd(lst_array)

    z_scores = (lst_array - mean_lst) / std_lst

    print(f"LST Mean: {mean_lst:.2f} °C")
    print(f"LST Std Dev: {std_lst:.2f} °C")

    return z_scores

'''def plot_z_scores(z_scores):
    plt.figure(figsize=(10, 6))
    plt.imshow(z_scores, cmap='coolwarm', vmin=-3, vmax=3)
    plt.colorbar(label='Z-score')
    plt.title('Normalized LST (Z-scores) for Singapore')
    plt.axis('off')
    plt.show()  
'''
def detect_hotspots(z_scores, threshold=1.5):
    '''so zscore -3 to 3, if > 2 then it is hotter, less than 2 is 'cooler' 
        now binary mask 0 to 1, 1 hotspot, 0 is normal 
        set threshold to 1 since it is 0% for 2
    '''
    hotspots = np.where(z_scores > threshold, 1, 0)

    hotspot_percent = np.sum(hotspots) / hotspots.size * 100
    print(f"Detected hotspot {hotspot_percent:.2f}% of Singapore.")

    return hotspots

#NEA rainfall and weather station coordinates in 120H (5 DAYS) till PRESENT
def load_rainfall(hours=120):
    base_url = "https://api.data.gov.sg/v1/environment/rainfall"
    
    #first fetch station metadata
    station_meta_resp = requests.get(base_url)
    if station_meta_resp.status_code != 200:
        raise Exception("Could not load station metadata")
    
    station_meta = station_meta_resp.json()['metadata']['stations']
    station_dict = {s['id']: {
        'name': s['name'],
        'lat': s['location']['latitude'],
        'lon': s['location']['longitude']
    } for s in station_meta}


def main():
    ee.Initialize(project='ee-alyshabm000')

    #load sg boundary
    singapore_boundary = load_singapore_boundary()
    '''singapore_boundary = ee.FeatureCollection( "FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
    .filter(ee.Filter.eq('ADM0_NAME', 'Singapore')).geometry()'''

    #Load LST image to sg boundary
    lst_image = load_lst(singapore_boundary)
    #lst_image = lst_image.clip(singapore_boundary)

    #Loading geemap 
    Map = geemap.Map()
    vis_params = {
        'min': 0,
        'max': 40,
        'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
    
    }
    #add lst image to map and display
    Map.centerObject(singapore_boundary, 10)
    Map.addLayer(singapore_boundary, {}, "Singapore Boundary")
    Map.addLayer(lst_image, vis_params, 'Singapore LST (°C)')
    

    #savemap to html
    '''html_file = "lst_map.html"
    Map.to_html(html_file)
    print(f"Map has been saved to {html_file}.")
    webbrowser.open(html_file)
    
    #Convert to numpy array
    lst_array = lst_to_numpy(lst_image, singapore_boundary)

    #normalize zscore
    z_scores = normalize_list_zscore(lst_array)
    plot_z_scores(z_scores)

    #detect hotspots
    hotspot_mask = detect_hotspots(z_scores)
    plt.figure(figsize=(10, 6))
    plt.imshow(hotspot_mask, cmap='hot')
    plt.title('Detected hotspots')
    plt.axis('off')
    plt.show()'''



if __name__ == "__main__":
    main()