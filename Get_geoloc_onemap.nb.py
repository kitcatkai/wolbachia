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
# # Note
# This code produce the dataframe with the lat, lon of buildings, walking distance and time matrix.
#
# Should only be run once and the output is saved to reduce call to Google api.

# %%
import pandas as pd
import json
import requests
import math
import numpy as np
from http import HTTPStatus
from haversine import haversine


# %% [markdown]
# # Getting the Geolocation

# %%
# Reading in the release data
release_pts = pd.read_excel("NSE Release Blk Postal Code.xlsx")
release_pts['lat'] = ''
release_pts['long'] = ''


# %%
url = "https://developers.onemap.sg/commonapi/search"

def get_latlon_frm_postal(postal):
    m = {'searchVal': postal, 'returnGeom' : 'Y', 'getAddrDetails' : 'Y'}
    response = requests.request(
        "GET",
        url,
        params=m
    )
    if response.status_code == HTTPStatus.OK:
        results = response.json()['results']
        for result in results:
            if result['POSTAL'] == str(postal):
                return result['LATITUDE'], result['LONGTITUDE']
        # Cannot find match
        while response.json()['totalNumPages'] > response.json()['pageNum']:
            m['pageNum'] = response.json()['pageNum'] + 1
            print(response.json()['totalNumPages'])
            print(m['pageNum'])
            response = requests.request(
                "GET",
                url,
                params=m
            )
            results = response.json()['results']
            for result in results:
                if result['POSTAL'] == str(postal):
                    return result['LATITUDE'], result['LONGTITUDE']
        print("Could not find the matching postal code {}!".format(postal))
        return 0,0
    else:
        print("Got response {} for postal code {}".format(response.status_code, postal))
    return 0, 0


# %%
get_latlon_frm_postal(6502578)

# %%
release_pts['lat'] = 0
release_pts['long'] = 0
for idx, row in release_pts.iterrows():
    lat, long = get_latlon_frm_postal(row['Postal'])
    release_pts.loc[idx, 'lat'] = lat
    release_pts.loc[idx, 'long'] = long    

# %%
release_pts

# %%
for each in range(num_blonum_blocks):
    if not df.loc[each, "location"]:
        m = {'searchVal': , 'returnGeom' : 'Y', 'getAddrDetails', 'Y'}
        response = requests.request(
            "POST",
            url,
            json=m
        )
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

# %% [markdown]
# # Getting the walking distance and time 

# %%
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




np.save("data/time_matrix.npy", time_matrix)
np.save("data/dist_matrix.npy", dist_matrix)
