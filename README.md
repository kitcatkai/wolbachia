This code is set up primarily for VS code DevContainer, but can be run in docker. 

# Running in docker

```bash
docker-compose up
docker exec -t mozzy_container pipenv run jupyter lab
```

## Data

The road network is obtained from the [MasterPlan 14 Land Use](https://data.gov.sg/dataset/master-plan-2014-land-use)
