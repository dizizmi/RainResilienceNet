import ee
ee.Initialize()
import os 
os.environ['OMP_NUM_THREADS'] = '1'
import geemap
import pandas as pd
import matplotlib.pyplot as plt 

def load_lst(start_date='2024-01-01', end_date='2025-04-20'):
    return ee.ImageCollection("MODIS/061/MOD11A2") \
        .filterDate(start_date, end_date) \
        .mean() \
        .select('LST_Day_1km') \
        .multiply(0.02).subtract(273.15) \
        .rename('LST')

def main():
    lst_image = load_list()

if __name__ == "__main__":
    main()