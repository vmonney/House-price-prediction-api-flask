"""API code for the FastAPI app."""

from pathlib import Path

import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Create api
app = FastAPI()

# Load GB Model
with Path("gb.pkl").open("rb") as f:
    model = dill.load(f)


# Type checking class through Pydantic
class ScoringItem(BaseModel):
    """Scoring item with all the features needed for prediction."""

    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs: float
    PostCode: str


@app.post("/")
async def scoring_endpoint(item: ScoringItem) -> dict:
    """Scoring endpoint for the API."""
    data = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(data)
    return {"prediction": int(yhat)}
