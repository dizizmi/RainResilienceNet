import ee
import geemap

def main():

    ee.Initialize(project='ee-alyshabm000')
    singapore_boundary = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1").filter(ee.Filter.eq('ADM0_NAME', 'Singapore'))

    elev = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(singapore_boundary)

    Map = geemap.Map()
    Map.centerObject(singapore_boundary, 10)

    zone1 = elev.gte(0).And(elev.lt(10)).multiply(1)
    zone2 = elev.gte(10).And(elev.lt(30)).multiply(2)
    zone3 = elev.gte(30).And(elev.lt(60)).multiply(3)
    zone4 = elev.gte(60).And(elev.lt(100)).multiply(4)
    zone5 = elev.gte(100).And(elev.lte(165)).multiply(5)

    # Combine all zones
    elevation_zones = zone1.add(zone2).add(zone3).add(zone4).add(zone5)

    # Visualize by zone number
    zone_vis = {
        'min': 1,
        'max': 5,
        'palette': ['#0000ff', '#00ff00', '#ffff00', '#ffa500', '#8b0000']
    }

    Map = geemap.Map()
    Map.centerObject(singapore_boundary, 10)
    Map.addLayer(elevation_zones, zone_vis, "Elevation Zones")

    Map.add_legend(
        title="Elevation Zones (m)",
        labels=[
            '0–10 (Very Low)',
            '10–30 (Low)',
            '30–60 (Moderate)',
            '60–100 (High)',
            '100–165 (Very High)'
        ],
        colors=['#0000ff', '#00ff00', '#ffff00', '#ffa500', '#8b0000']
    )

    Map.to_html("singapore_dem_map3.html")

if __name__ == '__main__':
    main()
  