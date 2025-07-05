#!/bin/bash

RPU=$1
cd "/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/${RPU}"

# Read in elevation raster and use it to set project region 
r.in.gdal input="${RPU}_elev_cm.tif" output=elev
g.region raster=elev -p

# Convert elevation from cm to m
r.mapcalc "elev = float(elev)/100.0" --overwrite

# Calculate terrain forms and slope from DEM
r.geomorphon -m elevation=elev forms=terrainforms search=1000.0
r.slope.aspect elevation=elev slope=slope format=percent

# Calculate topographic position index (TPI)
r.neighbors -c input=elev output=neighborhood_elev size=9 method=average
r.mapcalc "tpi = elev - neighborhood_elev"

# Convert to integer data types to save disk space
r.mapcalc "slope = int(10*slope)" --overwrite
r.mapcalc "tpi = int(100*tpi)" --overwrite

# Specify nodata value
r.null map=slope null=-999999 setnull=-999999
r.null map=terrainforms null=-999999 setnull=-999999
r.null map=tpi null=-999999 setnull=-999999

# Save results
r.out.gdal input=terrainforms output="${RPU}_geomorphon.tif" nodata=-999999 type=Int32 --overwrite
r.out.gdal input=slope output="${RPU}_slope_x1000.tif" nodata=-999999 type=Int32 --overwrite
r.out.gdal input=tpi output="${RPU}_tpi_cm.tif" nodata=-999999 type=Int32 --overwrite
