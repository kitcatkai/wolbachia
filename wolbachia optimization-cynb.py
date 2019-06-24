# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import json
import requests
import math
import numpy as np
from haversine import haversine

with open('config.json') as f:
    settings = json.load(f)
df = pd.read_csv("block.csv")
# -

# # Getting the distance and time Matrix
# Should only be run once and the output is saved to reduce call to Google api.

release_pts = pd.read_excel('Yishun Wolbachia Release Points.xlsx')

release_pts['NumRelease'] = release_pts['ReleaseLocation'].str.split(',').apply(len)
release_pts['Block'] = release_pts['Block'].astype('str')

release_pts['Block'] = release_pts['Block'].str.replace('.Bin Centre', '')

release_pts


def count_release(df):
    output = {
        'Road' : df.iloc[0,0],
        'NumRelease' : df['NumRelease'].sum(),
    }
    return pd.Series(output)


df = release_pts.groupby('Block').apply(count_release).reset_index()

df['address'] = df['Block'] + '+' + df['Road'].str.replace(' ','+')

df['location'] = None

# +
url= "https://maps.googleapis.com/maps/api/geocode/json"

for each in range(len(df)):
    if not df.loc[each, 'location']:
        params = dict(
            address = df.loc[each, 'address'],
            key = settings['api_key']
        )

        resp = requests.get(url=url, params=params)
        data = resp.json()

        try:
            df.loc[each, 'location'] = str(data['results'][0]['geometry']['location']['lat']) + "," + str(data['results'][0]['geometry']['location']['lng'])
        except:
            print(each)
            print(data)
# -


df.to_pickle('block_with_geoloc.pickle')

# +
arr,arr1,arr2,arr3,arr4 = "","", "", "", ""

arr_total = []

for index, row in df.iterrows():
    arr += str(row['address']) + "|"

for index, row in df.iterrows():
    if(index < 25):
        arr1 += str(row['address']) + "|"
        
for index, row in df.iterrows():
    if(25 <= index and index < 50):
        arr2 += str(row['address']) + "|"
        
for index, row in df.iterrows():
    if(50 <= index and index < 75):
        arr3 += str(row['address']) + "|"
    
    
for index, row in df.iterrows():
    if(75 <= index and index < 84):
        arr4 += str(row['address']) + "|"
        
arr_total.append(arr1[:-1])
arr_total.append(arr2[:-1])
arr_total.append(arr3[:-1])
arr_total.append(arr4[:-1])

arr_total

# +
# WILL RUN FOR 2 MINS AT MAX

time_matrix = np.zeros((84,84))
time_array = []
dist_matrix = np.zeros((84,84))
dist_array = []

url_google_route = "https://maps.googleapis.com/maps/api/distancematrix/json"


for each_index in range(len(df)):
    time_array = []
    dist_array = []
    print("Inititaing " + str(each_index) + "/83 first...")
    for index, dest_input in enumerate(arr_total):
#         print("Working on " + str(index) + "/3 first...")
#         print(dest_input)
        params = dict(
            destinations = dest_input,
            origins = df.iloc[each_index]['address'],
            key = settings['api_key'],
            mode = "walking"
        )
        resp = requests.get(url=url_google_route, params=params)
        data = resp.json()
        for each in range(len(data['rows'][0]['elements'])):
            if data['rows'][0]['elements'][each]['status'] == 'OK':
                time_array.append(data['rows'][0]['elements'][each]['duration']['value'])
                dist_array.append(data['rows'][0]['elements'][each]['distance']['value'])
            else:
                print(each)
                print("Origin: {}, Destination: ".format(data['origin_addresses'][0]))
                print(data)
    print(each_index)
    time_matrix[each_index] = time_array
    dist_matrix[each_index] = dist_array

# -

np.save('time_matrix.npy', time_matrix)
np.save('dist_matrix.npy', dist_matrix)

# # Loading the processed data

# +
import pandas as pd
import json
import requests
import math
import numpy as np
from haversine import haversine

with open('config.json') as f:
    settings = json.load(f)
df = pd.read_csv("block.csv")
# -

time_matrix = np.load('time_matrix.npy')
dist_matrix = np.load('dist_matrix.npy')
df = pd.read_pickle('block_with_geoloc.pickle')

time_matrix[37, :] = 0 #Setting going out time to zero

time_matrix[:, 37]

# + {"active": ""}
# time_matrix[:, 37] = 0 # Setting return to base time to zero.
# -

time_matrix.shape

# + {"active": ""}
# # Adding in the additional time for release
# for idx, value in df['NumRelease'].iteritems():
#     time_matrix[idx, :] += value*60

# +
"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))
    
def print_readable_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    max_route_time = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        pair_pts = list(zip(route[:-1], route[1:]))
        walk_dist = 0
        walk_time = 0
        elucid_dist = 0
        for num, (i, j) in enumerate(pair_pts[1:]): #Dropping the first point
            if num == 0:
                plan_output += "Blk {}".format(df.loc[i, "Block"])
            plan_output += " -> Blk {}".format(df.loc[j, "Block"])
            walk_dist += dist_matrix[i, j]
            walk_time += time_matrix[i, j]
            loc_i = df.loc[i, 'location'].split(',')
            loc_j = df.loc[j, 'location'].split(',')
            elucid_dist += haversine((float(loc_i[0]), float(loc_i[1])), (float(loc_j[0]), float(loc_j[1]))) * 1000
        plan_output += "\n"
        plan_output += 'Walking distance of the route: {}m\n'.format(walk_dist)
        plan_output += 'Walking time of the route: {}mins\n'.format(walk_time/60)
        plan_output += 'Euclidean distance of the route: {}m\n'.format(elucid_dist)
        print(plan_output)
        max_route_distance = max(elucid_dist, max_route_distance)
        max_route_time = max(walk_time, max_route_time)
    print('Maximum of the euclidean distances: {}m'.format(max_route_distance))
    print('Maximum of the route duration: {}mins'.format(max_route_time/60))


# -

# # Plotting the routes 

# +
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data['distance_matrix'] = time_matrix
data['num_vehicles'] = 5
data['depot'] = 37

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(
    len(data['distance_matrix']), data['num_vehicles'], data['depot'])

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)


# Create and register a transit callback.
def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define cost of each arc.
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Distance constraint.
dimension_name = 'Distance'
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    100000,  # vehicle maximum travel distance
    True,  # start cumul to zero
    dimension_name)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if solution:
    print_readable_solution(data, manager, routing, solution)
#     print_solution(data, manager, routing, solution)
# -

import folium
color = ['red', 'blue', 'green', 'purple', 'orange', 'darkred','lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue']
mapa = folium.Map([1.3,103.9],
                  zoom_start=4,
                  tiles='cartodbpositron')
for vehicle_id in range(data['num_vehicles']):
    loc = []
    index = routing.Start(vehicle_id)
    print('\nRoute for vehicle {}:'.format(vehicle_id))
    route_distance = 0
    tmp = df.loc[manager.IndexToNode(index), 'location'].split(',')
    loc.append((float(tmp[0]), float(tmp[1])))
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        tmp = df.loc[manager.IndexToNode(index), 'location'].split(',')
        loc.append((float(tmp[0]), float(tmp[1])))
    folium.Marker(loc[1], popup=str(index)).add_to(mapa)
    folium.PolyLine(loc[1:], color=color[vehicle_id], weight=2.5, opacity=1).add_to(mapa)
mapa

# # Arbitrary start and end point 

new_time_matrix = np.zeros((85, 85))
new_time_matrix[:84,:84] = time_matrix

# +
"""Solve the CVRP problem."""
# Instantiate the data problem.
"""Stores the data for the problem."""
data = {}
data['distance_matrix'] = new_time_matrix
data['num_vehicles'] = 6
data['depot'] = 84

# Create the routing index manager.
manager = pywrapcp.RoutingIndexManager(
    len(data['distance_matrix']), data['num_vehicles'], data['depot'])

# Create Routing Model.
routing = pywrapcp.RoutingModel(manager)


# Create and register a transit callback.
def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Define cost of each arc.
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add Distance constraint.
dimension_name = 'Distance'
routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    100000,  # vehicle maximum travel distance
    True,  # start cumul to zero
    dimension_name)
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# Setting first solution heuristic.
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem.
solution = routing.SolveWithParameters(search_parameters)

# Print solution on console.
if solution:
    print_solution(data, manager, routing, solution)
# -

import folium
color = ["red", "blue", "green", "yellow", "black", "grey"]
mapa = folium.Map([1.3,103.9],
                  zoom_start=4,
                  tiles='cartodbpositron')
for vehicle_id in range(data['num_vehicles']):
    loc = []
    index = routing.Start(vehicle_id)
    print('\nRoute for vehicle {}:'.format(vehicle_id))
    route_distance = 0
#     tmp = df.loc[manager.IndexToNode(index), 'location'].split(',')
#     loc.append((float(tmp[0]), float(tmp[1])))
    while not routing.IsEnd(index):
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        if manager.IndexToNode(index) < len(df):
            tmp = df.loc[manager.IndexToNode(index), 'location'].split(',')
            loc.append((float(tmp[0]), float(tmp[1])))
        else:
            print(manager.IndexToNode(index))
    folium.PolyLine(loc, color=color[vehicle_id], weight=2.5, opacity=1).add_to(mapa)
mapa


