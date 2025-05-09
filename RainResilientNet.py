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
import seaborn as sns
import geopandas as gpd
import rasterio
from skimage.transform import resize

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

#MODIS13Q1 for NDVI 
def load_ndvi(singapore_boundary, start_date='2024-01-01', end_date='2025-04-20'):
    collection = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start_date, end_date) 
        
    count = collection.size().getInfo()
    if count == 0:
        raise ValueError("No images found for the specified date range.")

    ndvi = collection.mean() \
        .select('NDVI') \
        .rename('NDVI')

    return ndvi.clip(singapore_boundary)

#ASTER DEM v3 for elevation of singapore (1 granule of sg)
def load_elevation(elev_path: str, normalize: bool = True):
    '''
    note: elev path is the path to the tiff file, boundary path is path to GEOjson to clip sg boundary, normalise between 0 to 1
    returns np.ndarray the clipped and normalized
    '''

    with rasterio.open(elev_path) as src:
       elev_array = src.read(1)

    if normalize:
        elev_array = (elev_array - np.nanmin(elev_array)) / (np.nanmax(elev_array) - np.nanmin(elev_array))
        elev_array = np.nan_to_num(elev_array)

    return elev_array

#resample elevation to fit CNN
def resample_elevation(elev_array, target_shape=(256,256)):
    '''
    note: elev_array is 2d elevation array, target shape is the height width of output
    returnsn np.ndarray resampled elevation array
    '''
    elev_resampled  = resize(
        elev_array,
        target_shape,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True

    )
    return elev_resampled

'''
#URA Land Plan 2019, load geoJSON file (WIP, not done w URA)
def load_ura(ura_path):
    
    # Load the GeoJSON file
    gdf = gpd.read_file(ura_path)

    # Convert to GeoJSON format
    geojson_str = gdf.to_json()

    # Load the GeoJSON data into an Earth Engine FeatureCollection
    feature_collection = geemap.geojson_to_ee(geojson_str)

    return feature_collection

def assign_ura_class():
    #map the land use to codes
    return {
        'Residential': 1,
        'Commercial': 2,
        'Industrial': 3,
        'Agricultural': 4,
        'Park': 5,
        'Waterbody': 6,
        'Forest': 7,
        'Other': 0
    }

def final_load(ura_fc):

    ee_class_map = ee.Dictionary(assign_ura_class())

    def assign_class(feature):
        return feature.set('LU_CODE', ee_class_map.get(feature.get('LU_DESC'), 0))
     
  
    return ura_fc.map(assign_class)

def rasterize_land(ura_fc):
    return ura_fc.reduceToImage(
        properties=['LU_CODE'],
        reducer=ee.Reducer.first()
    ).rename('LandUse') 

'''


def lst_to_numpy(lst_ee, singapore_boundary, scale=1000):
    arr = geemap.ee_to_numpy(
        ee_object=lst_ee,
        region=singapore_boundary,
        scale=scale
    )

    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr

#zscoring for LST
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
#binary mask from zcsoring for LST
def detect_hotspots(z_scores, threshold=1.5):
    '''so zscore -3 to 3, if > 2 then it is hotter, less than 2 is 'cooler' 
        now binary mask 0 to 1, 1 hotspot, 0 is normal 
        set threshold to 1 since it is 0% for 2
    '''
    hotspots = np.where(z_scores > threshold, 1, 0)

    hotspot_percent = np.sum(hotspots) / hotspots.size * 100
    print(f"Detected hotspot {hotspot_percent:.2f}% of Singapore.")

    return hotspots

#NEA rainfall and weather station coordinates in 120H (5 DAYS) till PRESENT, thinking if it should be shorter...? ~3h-24?
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

    rainfall_records = []

    #get rainfall data
    for i in range(hours):
        timestamp = (datetime.utcnow() - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            r = requests.get(base_url, params={"date_time": timestamp})
            if r.status_code == 200:
                data = r.json()
                readings = data['items'][0]['readings']
                for reading in readings:
                    sid = reading['station_id']
                    value = reading['value']
                    info = station_dict.get(sid)

                    if info:
                        rainfall_records.append({
                            'timestamp': timestamp,
                            'station_id': sid,
                            'station_name': info['name'],
                            'lat': info['lat'],
                            'lon': info['lon'],
                            'rainfall_mm': value
                        })
        except Exception as e:
            print(f"Error at {timestamp}: {e}")

        time.sleep(0.1)  # avoid overloading API   
    return pd.DataFrame(rainfall_records)

def rainfall_raster(latlons, z_scores, hotspot_mask, bounds):

    #zcoring for rainfall data
    '''convert latitude and longt to row col for raster array for pixel coordinates'''
    def rainfall_to_pixel(lat, lon, bounds, array_shape):

        lat_min, lat_max, lon_min, lon_max = bounds
        height, width = array_shape

        row = int((lat_max - lat) / (lat_max - lat_min) * height)
        col = int((lon - lon_min) / (lon_max - lon_min) * width)
        return row, col
    
    samples = []

    for lat, lon in latlons:
        row, col = rainfall_to_pixel(lat, lon, bounds, z_scores.shape)
        if 0 <= row < z_scores.shape[0] and 0 <= col < z_scores.shape[1]:
            z_val = z_scores[row, col]
            hotspot = hotspot_mask[row, col]

        else:
            z_val = np.nan
            hotspot = np.nan

        samples.append({
            'lat': lat,
            'lon': lon,
            'z_score': z_val,
            'hotspot': hotspot
        })
    
    return samples

def prepare_rainfall_data(rainfall_120h_df, z_scores, hotspot_mask, bounds):
    rainfall_120h_df['timestamp'] = pd.to_datetime(rainfall_120h_df['timestamp'])
    rainfall_120h_df.sort_values(by=['station_id', 'timestamp'], inplace=True)

    # Summarize 5-day rainfall per station
    rainfall_summary = rainfall_120h_df.groupby(['station_id', 'station_name', 'lat', 'lon']) \
        .agg(total_rainfall_mm=('rainfall_mm', 'sum')) \
        .reset_index()

    latlon_list = list(zip(rainfall_summary['lat'], rainfall_summary['lon']))

    samples = rainfall_raster(latlon_list, z_scores, hotspot_mask, bounds)
    sample_df = pd.DataFrame(samples)

    rainfall_summary['lst_zscore'] = sample_df['z_score']
    rainfall_summary['in_hotspot'] = sample_df['hotspot']

    return rainfall_summary

'''
#XGBoost regression model- test rainfall on urban heat
def prepare_features(df):
    df = df.dropna(subset=['lst_zscore', 'total_rainfall_mm'])


    features = ['lst_zscore', 'in_hotspot', 'lat', 'lon']
    X = df[features]
    y = df['total_rainfall_mm']

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_evaluate_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"XGBoost R² score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f} mm")

    return model

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=feature_names, palette='viridis')
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

def run_xgboost_pipeline(rainfall_summary):
    print("Preparing features...")
    X_train, X_test, y_train, y_test = prepare_features(rainfall_summary)

    print("Training XGBoost...")
    model = train_evaluate_xgboost(X_train, X_test, y_train, y_test)

    print("Plotting feature importance...")
    plot_feature_importance(model, X_train.columns)
'''

def main():
    ee.Initialize(project='ee-alyshabm000')

    #load sg boundary
    singapore_boundary = load_singapore_boundary()
    '''singapore_boundary = ee.FeatureCollection( "FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
    .filter(ee.Filter.eq('ADM0_NAME', 'Singapore')).geometry()'''

    #Load LST image to sg boundary
    lst_image = load_lst(singapore_boundary)
    #lst_image = lst_image.clip(singapore_boundary)

    '''#load URA
    ura_fc = load_ura("MasterPlan2019LandUselayer.geojson")
    ura_fc_mapped = map_land_use(ura_fc)
    ura_raster = rasterize_land(ura_fc_mapped)

    '''

    #load elevation
    elev = load_elevation(
        elev_path="AST14DEM_00308102024025318_20250508075518_368746.tif"
    )

    elev_resized = resample_elevation(elev, target_shape=(256, 256))
    print(f"Elevation Array Shape: {elev_resized.shape}")

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
    #Map.addLayer(ura_raster, {}, 'URA Land Use')
    

    #savemap to html
    '''html_file = "lst_map.html"
    Map.to_html(html_file)
    print(f"Map has been saved to {html_file}.")
    webbrowser.open(html_file)
    '''
    #Convert to numpy array
    lst_array = lst_to_numpy(lst_image, singapore_boundary)

    #normalize zscore
    z_scores = normalize_list_zscore(lst_array)
   # plot_z_scores(z_scores)

    #detect hotspots
    hotspot_mask = detect_hotspots(z_scores)
    ''' plt.figure(figsize=(10, 6))
    plt.imshow(hotspot_mask, cmap='hot')
    plt.title('Detected hotspots')
    plt.axis('off')
    plt.show()
    '''
    #get rainfall data
    rainfall_120h_df = load_rainfall(120)
    bounds = (1.22, 1.48, 103.6, 104.0)
    rainfall_summary = prepare_rainfall_data(
        rainfall_120h_df,
        z_scores,
        hotspot_mask,
        bounds
    )   
   
    #plotting
    '''sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rainfall_summary,
    x='lst_zscore',
    y='total_rainfall_mm',
    hue='in_hotspot',
    palette={0: 'blue', 1: 'red'},
    s=80)

    plt.xlabel("LST Z-Score (Urban Heat Intensity)")
    plt.ylabel("Total Rainfall (last 120h) [mm]")
    plt.title("Rainfall vs Urban Heat Intensity at NEA Stations")
    plt.axvline(x=1.5, color='gray', linestyle='--', label='Z = 1.5 Threshold')
    plt.legend(title='In Hotspot')
    plt.tight_layout()
    print(plt.show())
    '''

    # run_xgboost_pipeline(rainfall_summary)



if __name__ == "__main__":
    main()