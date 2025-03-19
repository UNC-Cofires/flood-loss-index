#!/bin/bash

HUC6=$1
cd "/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/${HUC6}"

# Read in elevation raster and use it to set region of interest 
gdalwarp "${HUC6}_elev.tif" "${HUC6}_elev_northup.tif" -overwrite
r.in.gdal input="${HUC6}_elev_northup.tif" output=elev
g.region raster=elev -p

## Read in data on stream network and floodplain boundaries

# Waterbodies
gdalwarp "${HUC6}_waterbodies_nanvalued.tif" "${HUC6}_waterbodies_nanvalued_northup.tif" -overwrite
r.in.gdal input="${HUC6}_waterbodies_nanvalued_northup.tif" output=waterbodies
r.null map=waterbodies null=0
r.mapcalc "waterbodies = int(waterbodies)" --overwrite

# 100y floodplain
gdalwarp "${HUC6}_floodplain_100y_nanvalued.tif" "${HUC6}_floodplain_100y_nanvalued_northup.tif" -overwrite
r.in.gdal input="${HUC6}_floodplain_100y_nanvalued_northup.tif" output=floodplain100y
r.null map=floodplain100y null=0
r.mapcalc "floodplain100y = int(floodplain100y)" --overwrite

# 500y floodplain
gdalwarp "${HUC6}_floodplain_500y_nanvalued.tif" "${HUC6}_floodplain_500y_nanvalued_northup.tif" -overwrite
r.in.gdal input="${HUC6}_floodplain_500y_nanvalued_northup.tif" output=floodplain500y
r.null map=floodplain500y null=0
r.mapcalc "floodplain500y = int(floodplain500y)" --overwrite

# Get flow direction raster 
r.watershed elevation=elev threshold=1000 accumulation=flow_acc drainage=flow_dir stream=ws_stream
r.out.gdal input=flow_acc output="${HUC6}_grass_flow_acc.tif" --overwrite
r.out.gdal input=flow_dir output="${HUC6}_grass_flow_dir.tif" --overwrite

# Calculate distance to and height above waterbodies 
r.stream.distance stream_rast=waterbodies direction=flow_dir elevation=elev method=downstream distance=waterbodies_dist difference=waterbodies_diff
r.out.gdal input=waterbodies_dist output="${HUC6}_grass_waterbodies_dist.tif" --overwrite
r.out.gdal input=waterbodies_diff output="${HUC6}_grass_waterbodies_diff.tif" --overwrite

# Calculate distance to and height above 100y floodplain 
r.stream.distance stream_rast=floodplain100y direction=flow_dir elevation=elev method=downstream distance=floodplain100y_dist difference=floodplain100y_diff
r.out.gdal input=floodplain100y_dist output="${HUC6}_grass_floodplain_100y_dist.tif" --overwrite
r.out.gdal input=floodplain100y_diff output="${HUC6}_grass_floodplain_100y_diff.tif" --overwrite

# Calculate distance to and height above 500y floodplain 
r.stream.distance stream_rast=floodplain500y direction=flow_dir elevation=elev method=downstream distance=floodplain500y_dist difference=floodplain500y_diff
r.out.gdal input=floodplain500y_dist output="${HUC6}_grass_floodplain_500y_dist.tif" --overwrite
r.out.gdal input=floodplain500y_diff output="${HUC6}_grass_floodplain_500y_diff.tif" --overwrite
