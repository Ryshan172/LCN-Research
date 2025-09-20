from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lcn_functions.model import create_lcn
from typing import Dict, Any, List
from sampler_functions.converted_sample import convert_and_sample
import pandas as pd
from sampler_functions.contingency_sampler import run_aggregate_sampler
from scoring_functions.interval_bic_score import compute_interval_bic_score
from utils.util_functions import save_json_data

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
    """
    This is the workflow for generating a tree-shaped lcn
    """

    try:
        # Generate and return an LCN with the input parameters 
        size = request.size
        interval_width = request.interval_width

        lcn = create_lcn(size, interval_width)
        
        # save lcn as a file 
        save_json_data("generated_lcn", lcn)

        return lcn 
    
    except Exception as e:
        return {"message": f"Error creating lcn: {e}"}


@router.post("/forward-sample")
def forward_sample(request: LCN):
    """
    This is the current workflow for forward sampling and scoring a generated LCN
    """

    try:
        # Set save location and file names
        dataset_path = "datasets/contingency_samples"
        contingency_save_file = "contingency_table.csv"

        # Running aggregate sampler instead of convert and sample
        samples_df = run_aggregate_sampler(request.dict())
        print(samples_df)

        # Save samples 
        samples_df.to_csv(f"{dataset_path}/{contingency_save_file}", index=False)
        
        samples_json = samples_df.to_dict(orient="records")

        # Compute BIC score
        bic_score_result = compute_interval_bic_score(samples_df)

        # Return object
        result = {
            "samples": samples_json,
            "bic_scores": bic_score_result
        }
    
        return result 
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling lcn: {e}")
