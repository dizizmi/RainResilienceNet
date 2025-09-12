import ee
import geemap
import numpy as np, pandas as pd, rasterio, re, json, os
from glob import glob
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from pyproj import Transformer
from datetime import datetime, timedelta

#generating daily LST images for Singapore from MODIS LST but realised modis lst is 8 day everaged
ee.Initialize(project='ee-alyshabm000')

singapore = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1") \
    .filter(ee.Filter.eq('ADM0_NAME', 'Singapore'))

start_date = '2025-01-01'
end_date = '2025-09-01'

modis = ee.ImageCollection('MODIS/061/MOD11A1') \
    .filterBounds(singapore) \
    .filterDate(start_date, end_date) \
    .select('LST_Day_1km')


def export_image(img):
    date_str = ee.Date(img.get('system:time_start')).format('YYYY_MM_dd').getInfo()
    img_scaled = img.multiply(0.02).subtract(273.15).rename('LST_Celsius')

    task = ee.batch.Export.image.toDrive(
        image=img_scaled.clip(singapore),
        description=f'LST_MODIS_{date_str}',
        folder='LST_Daily',
        fileNamePrefix=f'LST_MODIS_{date_str}',
        region=singapore.geometry(),
        scale=1000,
        maxPixels=1e13
    )
    task.start()
    print(f"Started task: LST_MODIS_{date_str}")

def main():

    img_list = modis.toList(modis.size())
    n = img_list.size().getInfo()

    for i in range(n):
        image = ee.Image(img_list.get(i))
        export_image(image)
  
if __name__ == '__main__':
    main()
  