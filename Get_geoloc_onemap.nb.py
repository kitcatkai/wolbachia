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
# This code read in the excel file with a "Postal" column and add in a "lat" and "long" columns based on onemap api.  

# %%
from http import HTTPStatus

import pandas as pd
import requests

# %%
filename = "NSE Release Blk Postal Code.xlsx"
output_csv = "data/NSE_release_with_latlong.csv"

# %% [markdown]
# # Getting the Geolocation

# %%
# Reading in the release data
release_pts = pd.read_excel(filename)
release_pts["lat"] = ""
release_pts["long"] = ""


# %%
ONEMAP_SEARCH_URL = "https://developers.onemap.sg/commonapi/search"


def get_latlon_frm_postal(postal):
    m = {"searchVal": postal, "returnGeom": "Y", "getAddrDetails": "Y"}
    response = requests.request("GET", ONEMAP_SEARCH_URL, params=m)
    if response.status_code == HTTPStatus.OK:
        if response.json()["totalNumPages"] == 0:
            print("No result found for postal code {}!".format(postal))
            return 0, 0
        results = response.json()["results"]
        for result in results:
            if result["POSTAL"] == str(postal):
                return result["LATITUDE"], result["LONGTITUDE"]
        # Cannot find match
        while response.json()["totalNumPages"] > response.json()["pageNum"]:
            m["pageNum"] = response.json()["pageNum"] + 1
            response = requests.request("GET", ONEMAP_SEARCH_URL, params=m)
            results = response.json()["results"]
            for result in results:
                if result["POSTAL"] == str(postal):
                    return result["LATITUDE"], result["LONGTITUDE"]
        print("Could not find the matching postal code {}!".format(postal))
        return 0, 0
    else:
        print("Got response {} for postal code {}".format(response.status_code, postal))
    return 0, 0


# %%
release_pts["lat"] = 0
release_pts["long"] = 0
for idx, row in release_pts.iterrows():
    lat, long = get_latlon_frm_postal(row["Postal"])
    release_pts.loc[idx, "lat"] = lat
    release_pts.loc[idx, "long"] = long

# %%
release_pts.to_csv(output_csv, index=False)
