from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lcn_functions.model import create_lcn
from typing import Dict, Any, List
from sampler_functions.bn_topological import ancestral_sample_bn, bn_to_json, build_precise_bn_from_lcn
from sampler_functions.converted_sample import convert_and_sample
import pandas as pd
from sampler_functions.contingency_sampler import run_aggregate_sampler
from scoring_functions.interval_bic_score import compute_interval_bic_score
from utils.data_saving import save_experiment_to_json
from utils.data_summaries import summarise_experiments_to_csv
from utils.util_functions import save_json_data
from learn_structure import learn_structure_lcn_samples
from pgmpy.estimators import BIC as BicScore
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from workflows.med_data_workflow import run_medical_experiments
from workflows.rq1_experiments import experiment_run_controller, experiment_run_variants

router = APIRouter()

# Define request schema
class LCNRequest(BaseModel):
    size: int
    interval_width: float
    width_dist: str
    incoming_edges: int


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
        width_dist_type = request.width_dist
        in_degree = request.incoming_edges

        lcn = create_lcn(size, interval_width, width_dist_type, in_degree)
        
        # Save LCN and parameters
        output_data = {
            "lcn": lcn,
            "parameters": {
                "size": size,
                "interval_width": interval_width,
                "width_dist_type": width_dist_type,
                "in_degree": in_degree
            }
        }

        save_json_data("generated_lcn", output_data)

        # Return generated LCN
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



@router.post("/forward-sample-and-learn")
def forward_sample(request: LCN):
    """
    - Forward Sample the Generated LCN 
    - Calculate the BIC score using the LCN-BIC method
    - Calculate the BIC score using pgmpy 
    - Use the forward sampled dataset to structure learn
    - Return original edges with learned edges 
    """

    try:
        # Set save location and file names
        dataset_path = "datasets/contingency_samples"
        contingency_save_file = "contingency_table.csv"

        # Running aggregate sampler instead of convert and sample
        aggregate_df, forward_samples_df = run_aggregate_sampler(request.dict(), num_samples=100)
        print(aggregate_df)

        # Save samples 
        aggregate_df.to_csv(f"{dataset_path}/{contingency_save_file}", index=False)
        
        aggregate_json = aggregate_df.to_dict(orient="records")

        # --- Custom LCN-BIC score ---
        bic_score_result = compute_interval_bic_score(aggregate_df)

        # --- pgmpy BIC score ---
        pgmpy_bic = BicScore(forward_samples_df)
        pgmpy_model = BayesianNetwork(request.edges)
        pgmpy_bic_score = pgmpy_bic.score(pgmpy_model)

        # Structure learn from LCN samples 
        original_edges = request.edges
        learning_result = learn_structure_lcn_samples("sample_test", forward_samples_df)

        # Return object
        result = {
            "contingency": aggregate_json,
            "bic_scores": {
                "lcn_bic": bic_score_result,
                "pgmpy_bic": pgmpy_bic_score
            },
            "edges": original_edges,
            "hill_climb": learning_result["hill_climbing"],
            "sim_annealing": learning_result["sim_annealing"]
        }
    
        return result 
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling lcn: {e}")


#-------------------Bayesian Network Sampling from LCN---------------------------

@router.post("/sample-bn-from-lcn")
def sample_bn_from_lcn(request: LCN):
    """
    Sample a single precise Bayesian Network from an LCN,
    save the BN as JSON, and return sampled states.
    """
    try:
        lcn_dict = request.dict(by_alias=True)
        
        # Sample one precise BN
        model, sampled_states = build_precise_bn_from_lcn(lcn_dict)

        # Model sanity check
        model_correct = model.check_model()
        print(f"Model correct: {model_correct}")

        print("\n--- Sampled world state ---")
        print(sampled_states)

        # Convert BN to JSON
        bn_json = bn_to_json(model)

        # Save BN JSON to file
        save_json_data("sampled_bn", bn_json)

        # Run forward sampling on the bayesian network
        bn_forward_samples = ancestral_sample_bn(model, n_samples=100)

        print(bn_forward_samples)

        # Return JSON response
        return {
            "sampled_states": sampled_states,
            "bn": bn_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling BN from LCN: {e}")



@router.post("/run-sample-experiment")
def run_sample_experiment():
    try:
        experiment_res = experiment_run_controller()

        # Hardcoding RunID for now
        run_id = "run2"

        file_path = save_experiment_to_json(experiment_res, run_id)


        # Summarising results of all experiments in outputs
        df = summarise_experiments_to_csv(
            input_dir="outputs",
            output_csv="experiment_summary.csv"
        )

        return {
            "status": "success",
            "run_id": run_id,
            "result_file": file_path,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running experiments: {e}"
        )
    


@router.post("/run-all-experiments")
def run_all_experiments():
    try:
        # Run all experiments
        experiment_run_variants()

        return {
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running experiments: {e}"
        )


@router.post("/summarise-results")
def summarise_experiment_results():
    try:

        # Summarising results of all experiments in results
        df = summarise_experiments_to_csv(
            input_dir="results",
            output_csv="experiment_summary.csv"
        )

        return {
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running experiments: {e}"
        )
    

@router.post("/run-medical-application")
def run_all_experiments():
    try:
        # Run all experiments
        run_medical_experiments(
            csv_path="medical_data.csv",
            n_lcns=1,
            num_samples=300
        )

        return {
            "status": "success",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running experiments: {e}"
        )