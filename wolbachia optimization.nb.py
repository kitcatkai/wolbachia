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
import pandas as pd
import json
import requests
import math
import numpy as np
from haversine import haversine
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import nearest_points
import folium

# %%
# filename = "data/block_with_geoloc.csv" # Old data
filename = "data/NSE_release_with_latlong.csv" # New data

# %% [markdown]
# ## Reading in the data

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
euclid_matrix = euclidean_dist

# %%
walking_speed = 90  # m/min
euclidean_time_matrix = (euclidean_dist / walking_speed) * 60

# %% [markdown]
# # Getting the number of roads crossed

# %%
buffer = 2000 # buffering by 2km of centriod

# %% [markdown]
# Getting the centriod from the points

# %%
pts = MultiPoint([row['geometry'] for _, row in gdf.iterrows()])
near_cent = nearest_points(pts.centroid, pts)
centriod = gdf.distance(near_cent[0]).idxmin()

# %%
# Getting the first point and converting it to local coordinate
area_of_interest = gdf.loc[centriod, 'geometry'].buffer(buffer)  # Buffering the point by buffer

# loading the road network and waterbody (NOTE: THEY ARE IN LOCAL COORDINATE)
sg_road = gpd.read_file("data/road.json")
waterbody = gpd.read_file("data/waterbody.json")

# localising to reduce the computation
waterbody_localised = waterbody.geometry[0].intersection(area_of_interest)
road_localised = sg_road.geometry[0].intersection(area_of_interest)

# %%
# getting all the linestrings
lines = []
for loc in longlat:
    lines += [LineString([loc, x]) for x in longlat]
lines_gpd = gpd.GeoSeries(lines, crs={"init": "epsg:4326"})

# Converting to local coordinates so that buffering distance is more intuitive
sg_lines_gpd = lines_gpd.copy()
sg_lines_gpd = sg_lines_gpd.to_crs(epsg=3414)

# %% [markdown]
# Getting The number of water body crossed

# %%
num_water = sg_lines_gpd.intersection(waterbody_localised)
counts_num_water = num_water.apply(lambda x: 1 if x.type == "LineString" else len(x))
water_crossed = counts_num_water.to_numpy().reshape(num_blocks, num_blocks)

# %% [markdown]
# ## Method 1:
# Number of intersection of line with the road.
#
# Problems: diagonal crossing counted as one road crossing

# %%
num_roads = sg_lines_gpd.intersection(road_localised)
counts_num_roads = num_roads.apply(lambda x: 1 if x.type == "LineString" else len(x))
road_crossed = counts_num_roads.to_numpy().reshape(num_blocks, num_blocks)
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
def num_road_crossed(adj_mat):
    """Special case of Dijkstra algorithm for shortest path 
    """
    num_vert = len(adj_mat)
    num_road = np.ones((num_vert, num_vert)) * num_vert
    np.fill_diagonal(num_road, 0)
    num_road[adj_mat > 0] = 1
    count = 1
    while count < num_vert and np.any(num_road == num_vert):
        old_road = num_road.copy()
        count += 1
        for i in range(num_vert):
            for j in range(num_vert):
                num_road[i, j] = min(old_road[i] + old_road[j])
    if np.any(num_road == num_vert):
        print("Graph is not connected! Please check!")
    return num_road


# %%
road_crossed = np.zeros((num_blocks, num_blocks))
cluster_road = num_road_crossed(adj_mat)
house_to_cluster = list(gdf["cluster"])
for idx, cluster in enumerate(house_to_cluster):
    for idx_ in range(idx + 1, num_blocks):
        road_crossed[idx, idx_] = cluster_road[(cluster, house_to_cluster[idx_])]
        road_crossed[idx_, idx] = cluster_road[(cluster, house_to_cluster[idx_])]

# %%
# Viewing the regions
mapa = folium.Map([1.43296,103.8386047], zoom_start=16, tiles="cartodbpositron")
mp_road = folium.FeatureGroup(name="mp road")
mp_road.add_child(
    folium.GeoJson(
        gpd.GeoSeries(region, crs={"init": "epsg:3414"}).to_crs(epsg=4326).to_json()
    )
)
mapa.add_child(mp_road)
display(mapa)

# %% [markdown]
# ## Generating the time/distance matrix
# Adding in penalty for road crossing. 
#
# In the future can modify the release_penalty as well

# %%
# Time penalty for road crossing
penalty_road = 500
penalty_water = 1000
depot = centriod
# penalty_lift =
# penalty_release_pts =


# %%
def add_release_penalty(time_matrix, penalty=600, method="fixed"):
    """ Adds time penalty for the release points
    """
    # Currently only fixed penalty is implemented, but possible to extend it to cater
    # for more cases such as lift penalty or based on number of release points.
    if method == "fixed":
        final_matrix = time_matrix.copy() + penalty
    else:
        print("No other method implemented.")
        final_matrix = time_matrix.copy()
    return final_matrix


# %%
elucid_matrix = euclidean_dist
time_matrix = (
    add_release_penalty(euclidean_time_matrix, penalty=240)
    + road_crossed * penalty_road + water_crossed * penalty_water
)
dist_matrix = euclidean_dist

# %%
"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def generate_readable_solution(
    data,
    manager,
    routing,
    solution,
    df=df,
    time_matrix=time_matrix,
    dist_matrix=dist_matrix,
    euclid_matrix=euclid_matrix,
):
    """Get the route details (path, time, distance)"""
    full_route = {"route": [], "route_print": [], "distance": [], "time": []}
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = "Route for vehicle {}:\n".format(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        pair_pts = list(zip(route[:-1], route[1:]))
        walk_dist = 0
        walk_time = 0
        euclid_dist = 0
        for num, (i, j) in enumerate(pair_pts[1:]):  # Dropping the first point
            if num == 0:
                plan_output += "Blk {}".format(df.loc[i, "Block"])
            plan_output += " -> Blk {}".format(df.loc[j, "Block"])
            walk_dist += dist_matrix[i, j]
            walk_time += time_matrix[i, j]
            euclid_dist += euclid_matrix[i, j]
        plan_output += "\n"
        plan_output += "Walking distance of the route: {}m\n".format(walk_dist)
        plan_output += "Walking time of the route: {}mins\n".format(walk_time / 60)
        plan_output += "Euclidean distance of the route: {}m\n".format(euclid_dist)
        plan_output += "Numbers of block: {}".format(len(route[1:]))
        full_route["route"].append(route)
        full_route["route_print"].append(plan_output)
        full_route["distance"].append(walk_dist)
        full_route["time"].append(walk_time)
    return full_route


def print_solution(full_route):
    """ Print the solution
    """
    for i in full_route["route_print"]:
        print(i)
    print("Maximum of the euclidean distances: {}m".format(max(full_route["distance"])))
    print("Maximum of the route duration: {}mins".format(max(full_route["time"]) / 60))


# %% [markdown]
# # Plotting the routes

# %% [markdown]
# For 5min in each building

# %%
current_time_matrix = time_matrix
current_distance_matrix = euclidean_dist

# %%
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data["distance_matrix"] = current_time_matrix
data["num_vehicles"] = 6
data["depot"] = depot

# Create the routing index manager.


def vrp_solver(data):
    """ Solved the VRP with the given distance matrix, num_vehicles and depot node
    INPUT
    data : dictionary with keys "distance_matrix", "num_vehicles" and "depot"
           "depot" is the index of the starting node in the distance_matrix 
    """
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        100000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    #     search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    #     search_parameters.local_search_metaheuristic = (
    #         routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    #     )
    #     search_parameters.time_limit.seconds = 30euclid_matrix
    #     search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    return solution, manager, routing


solution, manager, routing = vrp_solver(data)

# Print solution on console.
if solution:
    full_route = generate_readable_solution(
        data,
        manager,
        routing,
        solution,
        df=df,
        time_matrix=current_time_matrix,
        dist_matrix=current_distance_matrix,
        euclid_matrix=euclidean_dist,
    )
    print_solution(full_route)

# %%
import folium

# %%
color = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
]


# %%
def set_up_map(data, df, routing, manager, color=color):
    mapa = folium.Map([1.43296,103.8386047], zoom_start=16, tiles="cartodbpositron")
    for vehicle_id in range(data["num_vehicles"]):
        loc = []
        index = routing.Start(vehicle_id)
        route_distance = 0
        if not np.isnan(df.loc[manager.IndexToNode(index), "lat"]):
            loc.append((float(df.loc[manager.IndexToNode(index), "lat"]), float(df.loc[manager.IndexToNode(index), "long"])))
        while not routing.IsEnd(index):
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not np.isnan(df.loc[manager.IndexToNode(index), "lat"]):
                loc.append((float(df.loc[manager.IndexToNode(index), "lat"]), float(df.loc[manager.IndexToNode(index), "long"])))
        folium.Marker(loc[1], popup=str(index)).add_to(mapa)
#         print(loc)
        folium.PolyLine(loc[1:], color=color[vehicle_id], weight=2.5, opacity=1).add_to(
            mapa
        )
    return mapa


# %%
mapa = set_up_map(data, df, routing, manager)
mapa

# %% [markdown]
# # Arbitrary start and end point

# %%
penalty_water = 1000

# %%
current_time_matrix = np.zeros((num_blocks + 1, num_blocks + 1))
current_time_matrix[:num_blocks, :num_blocks] = (
    add_release_penalty(euclidean_time_matrix, penalty=300)
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

# Print solution on console.
if solution:
    full_route = generate_readable_solution(
        data,
        manager,
        routing,
        solution,
        df=df_append,
        time_matrix=current_time_matrix,
        dist_matrix=current_distance_matrix,
        euclid_matrix=current_euclid_matrix,
    )
    print_solution(full_route)

# %%
import folium

mapa = set_up_map(data, df_append, routing, manager)
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
        full_route = generate_readable_solution(
            data, manager, routing, solution, time_matrix=current_time_matrix, df=df_tmp
        )
        print("SOLUTIONS\n")
        print_solution(full_route)
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
