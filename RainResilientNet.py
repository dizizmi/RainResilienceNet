import ee
import os 
os.environ['OMP_NUM_THREADS'] = '1'
import geemap
import webbrowser
import pandas as pd
import matplotlib.pyplot as plt 

#MODIS11A2 for Land Surface Temperature (LST) 8-day avg 2024 till PRESENT
def load_lst(start_date='2024-01-01', end_date='2025-04-20'):

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

    return lst
        

def main():
    ee.Initialize(project='ee-alyshabm000')

    #load sg boundary
    singapore_boundary = ee.FeatureCollection( "FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
    .filter(ee.Filter.eq('ADM0_NAME', 'Singapore')).geometry()

    #Load LST image to sg boundary
    lst_image = load_lst()
    lst_image = lst_image.clip(singapore_boundary)

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
    html_file = "lst_map.html"
    Map.to_html(html_file)
    print(f"Map has been saved to {html_file}.")
    webbrowser.open(html_file)
    
    #exporting lst to geotiff
    geemap.ee_export_image_to_drive(
        image=lst_image,
        description='Singapore_LST',
        folder='earthengine',
        fileNamePrefix='SG_LST_2024',
        region=singapore_boundary,
        scale=1000,    # 1km MODIS native resolution
        crs='EPSG:4326',   # Standard lat/lon CRS
        maxPixels=1e13
)


if __name__ == "__main__":
    main()