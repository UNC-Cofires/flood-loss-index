#!/bin/bash

RPU=$1
cd "/proj/characklab/projects/kieranf/flood_damage_index/data/rasters/${RPU}"

# Read in elevation raster and use it to set project region 
r.in.gdal input="${RPU}_elev_cm.tif" output=elev
g.region raster=elev -p

# Read in flow direction and waterbody rasters 
r.in.gdal input="${RPU}_fdr.tif" output=fdr
r.in.gdal input="${RPU}_wb.tif" output=wb

# Calculate distance to and height above waterbodies 
r.stream.distance stream_rast=wb direction=fdr elevation=elev method=downstream distance=wb_dist difference=wb_diff

# Convert to integer raster to save disk space
r.mapcalc "wb_dist = int(wb_dist)" --overwrite
r.mapcalc "wb_diff = int(wb_diff)" --overwrite

# Specify nodata value
r.null map=wb_dist null=-999999
r.null map=wb_diff null=-999999

# Save results
r.out.gdal input=wb_dist output="${RPU}_flowpath_dist_m.tif" --overwrite
r.out.gdal input=wb_diff output="${RPU}_flowpath_diff_cm.tif" --overwrite