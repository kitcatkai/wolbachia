This code is set up primarily for VS code DevContainer, but can be run in docker.

# Running in docker

```bash
docker-compose up
docker exec -t mozzy_container pipenv run jupyter lab
```

## Data

The road network is obtained from the [MasterPlan 14 Land Use](https://data.gov.sg/dataset/master-plan-2014-land-use)

## Code

`Get_major_road.ipynb` reads in the MasterPlan 14 Land Use and generates the road network (`data/road.json`). The road network is use to obtain the number of road crossing between two blocks and added as penalty.

`Get_geoloc_walking_dist.ipynb` gets the latlon of buildings given as well as the walking distance and time given by google map. To run this code, you will need to provide `config.json` containing the field `api_key` with a valid google api key.

`wolbachia optimization.ipynb` contains the code that generates the route. It takes in the output from `Get_major_road.ipynb` and `Get_geoloc_walking_dist.ipynb`.
