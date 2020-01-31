# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Note
# This code produce the dataframe with the lat, lon of buildings, walking distance and time matrix.
#
# Should only be run once and the output is saved to reduce call to Google api.

# +
import pandas as pd
import json
import requests
import math
import numpy as np
from haversine import haversine

with open("config.json") as f:
    settings = json.load(f)
# -

# # Getting the Geolocation

# Reading in the release data
release_pts = pd.read_excel("Yishun Wolbachia Release Points.xlsx")
release_pts["NumRelease"] = release_pts["ReleaseLocation"].str.split(",").apply(len)
release_pts["Block"] = release_pts["Block"].astype("str")
release_pts["Block"] = release_pts["Block"].str.replace(".Bin Centre", "")
display(release_pts)


# Adding the number of release points to the df.

def count_release(df_input):
    """ Getting the number of release points at each block
    """
    output = {"Road": df_input.iloc[0, 0], "NumRelease": df_input["NumRelease"].sum()}
    return pd.Series(output)


df = release_pts.groupby("Block").apply(count_release).reset_index()
df["address"] = df["Block"] + "+" + df["Road"].str.replace(" ", "+")
df["location"] = None
display(df)

# +
url = "https://maps.googleapis.com/maps/api/geocode/json"

for each in range(num_blonum_blocks):
    if not df.loc[each, "location"]:
        params = dict(address=df.loc[each, "address"], key=settings["api_key"])
        resp = requests.get(url=url, params=params)
        data = resp.json()
        try:
            df.loc[each, "location"] = (
                str(data["results"][0]["geometry"]["location"]["lat"])
                + ","
                + str(data["results"][0]["geometry"]["location"]["lng"])
            )
        except:
            print(each)
            print(data)

df.to_pickle("data/block_with_geoloc.pickle")
# -

# # Getting the walking distance and time 

# +
# splitting the addresses into multiple string due to api limit

arr = ""
arr_total = []
count = 0
for index, row in df.iterrows():
    count += 1
    arr += str(row["address"]) + "|"
    if count >= 25:
        arr_total.append(arr)
        count = 0
        arr = ""
if count != 0:
    arr_total.append(arr)
arr_total

# +
# WILL RUN FOR 2 MINS AT MAX

num_blocks = len(df)
time_matrix = np.zeros((num_blocks, num_blocks))
time_array = []
dist_matrix = np.zeros((num_blocks, num_blocks))
dist_array = []

url_google_route = "https://maps.googleapis.com/maps/api/distancematrix/json"


for each_index in range(len(df)):
    time_array = []
    dist_array = []
    print("Inititaing " + str(each_index) + "/83 first...")
    for index, dest_input in enumerate(arr_total):
        params = dict(
            destinations=dest_input,
            origins=df.iloc[each_index]["address"],
            key=settings["api_key"],
            mode="walking",
        )
        resp = requests.get(url=url_google_route, params=params)
        data = resp.json()
        for each in range(len(data["rows"][0]["elements"])):
            if data["rows"][0]["elements"][each]["status"] == "OK":
                time_array.append(
                    data["rows"][0]["elements"][each]["duration"]["value"]
                )
                dist_array.append(
                    data["rows"][0]["elements"][each]["distance"]["value"]
                )
            else:
                print(each)
                print("Origin: {}, Destination: ".format(data["origin_addresses"][0]))
                print(data)
    print(each_index)
    time_matrix[each_index] = time_array
    dist_matrix[each_index] = dist_array
# -


np.save("data/time_matrix.npy", time_matrix)
np.save("data/dist_matrix.npy", dist_matrix)
