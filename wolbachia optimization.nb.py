# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loading the processed data

# %%
import geopandas as gpd
import numpy as np
# %%
import pandas as pd
from shapely.geometry import LineString, MultiPoint, Point
from shapely.ops import nearest_points
from utils import add_release_penalty, map_shapely_objects, vrp_solver, generate_readable_solution, set_up_map

# %%
road_crossed = np.load('data/road_crossed.pickle', allow_pickle=True)
water_crossed = np.load('data/water_crossed.pickle', allow_pickle=True)
euclidean_dist = np.load('data/distance.pickle', allow_pickle=True)

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
# Getting the centriod from the points

# %%
pts = MultiPoint([row['geometry'] for _, row in gdf.iterrows()])
near_cent = nearest_points(pts.centroid, pts)
centriod = gdf.distance(near_cent[0]).idxmin()

# %% [markdown]
# ## Generating the time/distance matrix
# Adding in penalty for road crossing. 
#
# In the future can modify the release_penalty as well

# %%
penalty_road = 300 # Time penalty for road crossing
penalty_water = 450 # time penalty for crossing water body
penalty_release = 240 # time needed for each release
depot = centriod


# %%
walking_speed = 90  # m/min
euclidean_time_matrix = (euclidean_dist / walking_speed) * 60
euclid_matrix = euclidean_dist
time_matrix = (
    add_release_penalty(euclidean_time_matrix, penalty=penalty_release)
    + road_crossed * penalty_road + water_crossed * penalty_water
)
dist_matrix = euclidean_dist

# %% [markdown]
# # Plotting the routes

# %%
current_time_matrix = time_matrix
current_distance_matrix = euclidean_dist

# %%
# %%time
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data["distance_matrix"] = current_time_matrix
data["num_vehicles"] = 6
data["depot"] = depot

solution, manager, routing = vrp_solver(data)

# Print solution on console.
data["time_matrix"] = current_time_matrix
data["dist_matrix"] = current_distance_matrix
data["df"] = df
data["euclid_matrix"] = euclidean_dist
if solution:
    full_route = generate_readable_solution(
        data,
        manager,
        routing,
        solution,
    )

# %%
mapa = set_up_map(data, manager, routing, solution)
mapa

# %% [markdown]
# # Arbitrary start and end point

# %%
current_time_matrix = np.zeros((num_blocks + 1, num_blocks + 1))
current_time_matrix[:num_blocks, :num_blocks] = (
    add_release_penalty(euclidean_time_matrix, penalty=penalty_release)
    + road_crossed * penalty_road + water_crossed * penalty_water
)
current_distance_matrix = np.zeros((num_blocks + 1, num_blocks + 1))
current_distance_matrix[:num_blocks, :num_blocks] = euclidean_dist
current_euclid_matrix = np.zeros((num_blocks + 1, num_blocks + 1))
current_euclid_matrix[:num_blocks, :num_blocks] = euclidean_dist
df_append = df.copy()
df_append = df_append.append(
    {"Block": "Dummy", "lat": None, "long": None}, ignore_index=True
)

# %%
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data["distance_matrix"] = current_time_matrix
data["num_vehicles"] = 7
data["depot"] = num_blocks

# Create the routing index manager.
solution, manager, routing = vrp_solver(data)
data["time_matrix"] = current_time_matrix
data["dist_matrix"] = current_distance_matrix
data["df"] = df_append
data["euclid_matrix"] = current_euclid_matrix
# Print solution on console.
if solution:
    full_route = generate_readable_solution(
        data,
        manager,
        routing,
        solution,
    )

mapa = set_up_map(data, manager, routing, solution)
display(mapa)

# %%
mapa = set_up_map(data, manager, routing, solution)
display(mapa)

# %% [markdown]
# # Looping to minimise distance travelled for all
#
# Current method only optimise the longest travelling time and not of the others, so looping to minimise the distance should give better results. 
#
# *Unfortunately this is not the case.*

# %%
current_time_matrix = time_matrix_10min

# %%
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data["distance_matrix"] = current_time_matrix
data["num_vehicles"] = 6
data["depot"] = 37
df_tmp = df.copy()

while data["num_vehicles"] > 2:
    # Create the routing index manager.
    solution, manager, routing = vrp_solver(data)

    # Print solution on console.
    if solution:
        data["df"] = df_tmp
        data["time_matrix"] = current_time_matrix
        print("SOLUTIONS\n")
        full_route = generate_readable_solution(data, manager, routing, solution)
        max_route_id = np.argmax(full_route["time"])
        print("\nMax\n")
        print(full_route["route_print"][max_route_id])
        max_route = full_route["route"][max_route_id][1:-1]
        df_tmp = df_tmp.drop(max_route)
        df_tmp = df_tmp.reset_index(drop=True)
        current_time_matrix = np.delete(current_time_matrix, max_route, 0)
        current_time_matrix = np.delete(current_time_matrix, max_route, 1)
        data["distance_matrix"] = current_time_matrix
        data["num_vehicles"] = data["num_vehicles"] - 1
        data["depot"] = data["depot"] - sum([(i < data["depot"]) for i in max_route])
    else:
        break
