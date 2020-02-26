This code is set up primarily for VS code DevContainer, but can be run in docker.

# Running in docker

```bash
docker-compose up
docker exec -t mozzy_container pipenv run jupyter lab
```

## Data

The road network is obtained from the [MasterPlan 14 Land Use](https://data.gov.sg/dataset/master-plan-2014-land-use)

## Code

- `Get_major_road.ipynb` reads in the MasterPlan 14 Land Use and generates the road network (`data/road.json`) and also the waterbody network (`data/waterbody.json`). The road and waterbody networks are use to obtain the number of road crossing between two blocks and added as penalty.

- `Get_geoloc_walking_dist.ipynb` [**No longer using this!**] gets the latlon of buildings given as well as the walking distance and time given by google map. To run this code, you will need to provide `config.json` containing the field `api_key` with a valid google api key. 

- `Get_geoloc_onemap.ipynb` get the lat, long from the Postal code(`NSE Release Blk Postal Code.xlsx`) using onemap api and saving it to `data/NSE_release_with_latlong.csv`.

- `Get_distance_roads_crossed.ipynb` get the distance between two points and also the number of roads crossed between them. Takes in the output from `Get_geoloc_onemap.ipynb`.

- `wolbachia optimization.ipynb` contains the code that generates the route. It takes in the output from `Get_major_road.ipynb` and `Get_distance_roads_crossed.ipynb`. You can adjust the penalties for road crossing, waterbody crossing, the numbers of drivers/team and also the gathering point (currently using the centroid).
