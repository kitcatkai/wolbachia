import folium
import geopandas as gpd
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# GENERIC

def count_line(x):
    """count the number of LineString after intersection
    to get the number of lines cut
    """
    if x.type == "LineString":
        if x.is_empty:
            return 0
        else:
            return 1
    else:
        return len(x)


def map_shapely_objects(*args, center=(1.43296, 103.8386047), zoom_start=16, epsg=3414):
    """Takes in a list of shapely objects (with epsg given),
    plotting them on folium with center and zoom_start given.
    """
    area = folium.Map(center, zoom_start=zoom_start, tiles="cartodbpositron")
    mp_road = folium.FeatureGroup(name="mp road")
    mp_road.add_child(
        folium.GeoJson(
            gpd.GeoSeries(args, crs={"init": "epsg:{}".format(epsg)})
            .to_crs(epsg=4326)
            .to_json()
        )
    )
    area.add_child(mp_road)
    display(area)


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


# FOR VRP

def generate_readable_solution(data, manager, routing, solution, print_output=True):
    """Get the route details (path, time, distance)"""
    full_route = {"route": [], "route_print": [], "distance": [], "time": []}
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = "Route for vehicle {}:\n".format(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        pair_pts = list(zip(route[:-1], route[1:]))
        walk_dist = 0
        walk_time = 0
        euclid_dist = 0
        for num, (i, j) in enumerate(pair_pts[1:]):  # Dropping the first point
            if num == 0:
                plan_output += "Blk {}".format(data["df"].loc[i, "Block"])
            plan_output += " -> Blk {}".format(data["df"].loc[j, "Block"])
            walk_dist += data["dist_matrix"][i, j]
            walk_time += data["time_matrix"][i, j]
            euclid_dist += data["euclid_matrix"][i, j]
        plan_output += "\n"
        plan_output += "Walking distance of the route: {}m\n".format(walk_dist)
        plan_output += "Walking time of the route: {}mins\n".format(walk_time / 60)
        plan_output += "Euclidean distance of the route: {}m\n".format(euclid_dist)
        plan_output += "Numbers of block: {}".format(len(route[1:]))
        full_route["route"].append(route)
        full_route["route_print"].append(plan_output)
        if print_output:
            print(plan_output)
        full_route["distance"].append(walk_dist)
        full_route["time"].append(walk_time)
    print("Maximum of the euclidean distances: {}m".format(max(full_route["distance"])))
    print("Maximum of the route duration: {}mins".format(max(full_route["time"]) / 60))
    return full_route


def vrp_solver(data, method="cheapest_arc"):
    """ Solved the VRP with the given distance matrix, num_vehicles and depot node
    INPUT
    data : dictionary with keys "distance_matrix", "num_vehicles" and "depot"
           "depot" is the index of the starting node in the distance_matrix
    """
    if method not in ["cheapest_arc", "guided_local_search"]:
        print("Unknown method {}. Using default cheapest arc!".format(method))
        method = "cheapest_arc"
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
    if method == "cheapest_arc":
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
    elif method == "guided_local_search":
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 120
        search_parameters.log_search = True

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    return solution, manager, routing


COLOR = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "lightred",
    "white",
    "beige",
    "pink",
    "lightblue",
]


def set_up_map(data, manager, routing, solution, color=None):
    if color is None:
        color = COLOR
    mapa = folium.Map([1.43296, 103.8386047], zoom_start=16, tiles="cartodbpositron")
    for vehicle_id in range(data["num_vehicles"]):
        loc = []
        index = routing.Start(vehicle_id)
        if not np.isnan(data["df"].loc[manager.IndexToNode(index), "lat"]):
            loc.append(
                (
                    float(data["df"].loc[manager.IndexToNode(index), "lat"]),
                    float(data["df"].loc[manager.IndexToNode(index), "long"]),
                )
            )
        while not routing.IsEnd(index):
            index = solution.Value(routing.NextVar(index))
            if not np.isnan(data["df"].loc[manager.IndexToNode(index), "lat"]):
                loc.append(
                    (
                        float(data["df"].loc[manager.IndexToNode(index), "lat"]),
                        float(data["df"].loc[manager.IndexToNode(index), "long"]),
                    )
                )
        folium.Marker(loc[1], popup=str(index)).add_to(mapa)
        folium.PolyLine(loc[1:], color=color[vehicle_id], weight=2.5, opacity=1).add_to(
            mapa
        )
    return mapa
