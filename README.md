# Logical Credal Network Structure Learning 

A repository for generating Logical Credal Networks (LCNs) and performing structure
learning experiments 

## Setup

Create and activate virtual environment 

```
python -m venv code_env

source code_env/bin/activate
```

Install required packages in environment

```
pip install -r requirements.txt
```


## Run API

All functionality such as generating LCNs and running experiments can
be done via an API

```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible locally at `http://localhost:8000/docs#/` 


## Run Experiments 

The endpoint `/run-all-experiments` can be used to run all experiments as shown in `workflows/rq1_experiments.py` 

Once all experiments have run, a csv file of the results can be created using 
the `/summarise-results` endpoint

The rest of the endpoints are mainly from earlier stages of the experiments and research and are not necessary to replicate results