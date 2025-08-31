from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lcn_functions.model import create_lcn

router = APIRouter()

# Define request schema
class LCNRequest(BaseModel):
    size: int
    interval_width: float


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