import ee

import os 
os.environ['OMP_NUM_THREADS'] = '1'
import geemap
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt 

def load_lst(start_date='2024-01-01', end_date='2025-04-20'):
    collection = ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterDate(start_date, end_date) 
        
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError("No images found for the specified date range.")
    
    return collection.mean() \
        .select('LST_Day_1km') \
        .multiply(0.02).subtract(273.15) \
        .rename('LST')

def main():
    ee.Initialize()
    lst_image = load_lst()
    #Loading geemap 
    Map = geemap.Map(center=[0, 0], zoom=2)
    vis_params = {
        'min': 0,
        'max': 40,
        'palette': ['blue', 'cyan', 'green', 'yellow', 'red']
    }
    #add lst image to map and display
    Map.addLayer(lst_image, vis_params, 'LST (°C)')
    Map.add_colorbar(vis_params, label="LST (°C)")
    
    #savemap to html
    html_file = "lst_map.html"
    Map.to_html(html_file)
    print(f"Map has been saved to {html_file}.")
    webbrowser.open(html_file)


if __name__ == "__main__":
    main()