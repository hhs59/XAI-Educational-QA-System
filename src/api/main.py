from fastapi import FastAPI

from src.api.schemas import PredictRequest
from src.core.orchestrator import solve

app = FastAPI()


@app.post("/predict")
def predict(request: PredictRequest):
    return solve(
        question=request.question,
        premises_nl=request.premises_NL,
        premises_fol=request.premises_FOL,
    )
