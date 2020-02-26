# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: 'Python 3.7.4 64-bit (''dev'': pipenv)'
#     language: python
#     name: python37464bitdevpipenvfd4c3ff9cb90482ea2b536dc79a995a8
# ---

# %%
import pandas as pd
import numpy as np
from haversine import haversine
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points
from utils import count_line, map_shapely_objects, num_road_crossed

# %% [markdown]
# ## Reading in the data

# %%
# filename = "data/block_with_geoloc.csv" # Old data
filename = "data/NSE_release_with_latlong.csv" # New data

# %%
df = pd.read_csv(filename)
num_blocks = len(df)
longlat = list(zip(df["long"], df["lat"]))
# Converting it to geopandas
geometry = [Point(xy) for xy in longlat]
crs = {"init": "epsg:4326"}
gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry).to_crs(epsg=3414)


# %% [markdown]
# # Getting the euclidean distance
# Will be using this instead of the walking distance as the walking distance is not accurate.

# %%
euclidean_dist = np.zeros((num_blocks, num_blocks))
for idx, loc in enumerate(longlat):
    euclidean_dist[idx] = [
        haversine((loc[1], loc[0]), (loc_[1], loc_[0])) * 1000 for loc_ in longlat
    ]

# %% [markdown]
# # Localising the roads and waterbody
# This will speed up the computation

# %%
buffer = 2000 # buffering by 2km of centroid

# %%
# Getting the centroid from the points
pts = MultiPoint([row['geometry'] for _, row in gdf.iterrows()])
near_cent = nearest_points(pts.centroid, pts)
centroid = gdf.distance(near_cent[0]).idxmin()

# %%
area_of_interest = gdf.loc[centroid, 'geometry'].buffer(buffer)  # Buffering the centroid by buffer

# loading the road network and waterbody (NOTE: THEY ARE IN LOCAL COORDINATE)
sg_road = gpd.read_file("data/road.json")
waterbody = gpd.read_file("data/waterbody.json")

# localising
waterbody_localised = waterbody.geometry[0].intersection(area_of_interest)
road_localised = sg_road.geometry[0].intersection(area_of_interest)

# %% [markdown]
# ## Getting all the linestrings
# %%
lines = []
for loc in longlat:
    lines += [LineString([loc, x]) for x in longlat]
lines_gpd = gpd.GeoSeries(lines, crs={"init": "epsg:4326"})

# Converting to local coordinates so that buffering distance is more intuitive
sg_lines_gpd = lines_gpd.copy()
sg_lines_gpd = sg_lines_gpd.to_crs(epsg=3414)

# %% [markdown]
# # Getting The number of water body crossed

# %%
num_water = sg_lines_gpd.intersection(waterbody_localised.buffer(30).buffer(-30))
counts_num_water = num_water.apply(count_line)
water_crossed = counts_num_water.to_numpy().reshape(num_blocks, num_blocks)

# %%
map_shapely_objects(sg_lines_gpd[counts_num_water.idxmax()], waterbody_localised.buffer(30).buffer(-30))

# %% [markdown]
# # Getting the numbers of roads crossed!

# %% [markdown]
# ## Method 1:
# Number of intersection of line with the road.
#
# Problems: diagonal crossing counted as one road crossing

# %%
num_roads = sg_lines_gpd.intersection(road_localised)
counts_num_roads = num_roads.apply(count_line)
road_crossed_lines = counts_num_roads.to_numpy().reshape(num_blocks, num_blocks)
# %%
counts_num_roads.idxmax()

# %% [markdown]
# ## Method 2:
# Getting the connected components and filter distance for those under 60m. Diagonal component filtered out based on area of overlap after buffering.
#
# Problems:
# - Some roads have small pop up, leading to small overlap. (Accepting those under 30m without the overlap filtering)
# - Components might have roads in it, just that it is not spliting it into two component. (ignoring it for now)

# %%
area_cc = (
    area_of_interest
    .difference(sg_road.geometry[0])
)  # buffering to merge the polygons together


# %% [markdown]
# Getting the connected components with points on it

# %%
land_mass = gpd.GeoDataFrame(
    pd.Series(range(len(area_cc)), name="cluster"),
    crs={"init": "epsg:3414", "no_defs": True},
    geometry=list(area_cc),
)
gdf = gpd.sjoin(gdf, land_mass, how="left", op="within")

# %%
gdf["cluster"] += len(gdf.cluster.unique())
region = []
for idx, num in enumerate(gdf.cluster.unique()):
    region.append(land_mass.iloc[num - len(gdf.cluster.unique())].geometry)
    gdf.loc[gdf["cluster"] == num, "cluster"] = idx

# %% [markdown]
# Current method of determining if component is adjacent or not.
# - filter distance below 60 and at least $10000m^2$
# - for those below 30m apart consider them as side by side
# - those between 30-60 m apart will buffer one component and check the area of overlap (if they are diagonal the area of overlap will be small (less than $150m^2$)

# %%
adj_mat = np.zeros((len(gdf.cluster.unique()), len(gdf.cluster.unique())))
for idx, ply in enumerate(region):
    for idx_ in range(idx + 1, len(region)):
        ply_ = region[idx_]
        dist = ply.distance(ply_)
        if dist < 30:
            adj_mat[idx, idx_] = 1
            adj_mat[idx_, idx] = 1
        elif dist < 60 and ply.intersection(ply_.buffer(dist + 5)).area > 150:
            adj_mat[idx, idx_] = 1
            adj_mat[idx_, idx] = 1

# %%
road_crossed_region = np.zeros((num_blocks, num_blocks))
cluster_road = num_road_crossed(adj_mat)
house_to_cluster = list(gdf["cluster"])
for idx, cluster in enumerate(house_to_cluster):
    for idx_ in range(idx + 1, num_blocks):
        road_crossed_region[idx, idx_] = cluster_road[(cluster, house_to_cluster[idx_])]
        road_crossed_region[idx_, idx] = cluster_road[(cluster, house_to_cluster[idx_])]

# %%
# Viewing the regions
map_shapely_objects(*region)

# %% [markdown]
# ## Method 3
# - Taking the max of both in case of weird shaped polygon

# %%
road_crossed = np.maximum(road_crossed_lines, road_crossed_region)

# %% [markdown]
# ## Saving!

# %%
road_crossed.dump('data/road_crossed.pickle')
water_crossed.dump('data/water_crossed.pickle')
euclidean_dist.dump('data/distance.pickle')
