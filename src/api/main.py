from fastapi import FastAPI
from src.core.orchestrator import solve

app = FastAPI()

@app.post('/predict')
def predict(request: PredictRequest):
    return solve(question=request.question,
                query_type=request.type,
                premises_nl=request.premises_NL)
