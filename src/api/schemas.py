from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    question: str
    premises_NL: list[str] | None = Field(default=None, alias="premises-NL")
    premises_FOL: list[str] | None = Field(default=None, alias="premises-FOL")
