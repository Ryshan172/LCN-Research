from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lcn_functions.model import create_lcn
from typing import Dict, Any, List
from sampler_functions.converted_sample import convert_and_sample
import pandas as pd
from sampler_functions.contingency_sampler import run_aggregate_sampler

router = APIRouter()

# Define request schema
class LCNRequest(BaseModel):
    size: int
    interval_width: float


class LCN(BaseModel):
    nodes: List[str]
    edges: List[List[str]]
    credal_sets: Dict[str, Dict[str, Dict[str, List[float]]]]
    logical_constraints: List[Dict[str, Dict[str, Any]]]


@router.post("/create-basic-lcn")
def process(request: LCNRequest):

    try:
        # Generate and return an LCN with the input parameters 
        size = request.size
        interval_width = request.interval_width

        lcn = create_lcn(size, interval_width)
        return lcn 
    
    except Exception as e:
        return {"message": f"Error creating lcn: {e}"}


@router.post("/forward-sample")
def forward_sample(request: LCN):
    try:
        # Pass the full LCN object to your sampler
        #samples_df = convert_and_sample(request.dict())  # <-- full LCN dict

        # Running aggregate sampler instead
        samples_df = run_aggregate_sampler(request.dict())
        print(samples_df.head())
        samples_df.to_csv("lcn_dataset.csv", index=False)

        samples_json = samples_df.to_dict(orient="records")
        return {"samples": samples_json}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling lcn: {e}")
