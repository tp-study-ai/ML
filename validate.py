from typing import Optional

from pydantic import BaseModel, Field


class Submission(BaseModel):
    source_code: str
    problem_url: str
    difficulty: int = Field(ge=0, le=2, description="0 for easier, 1 for the same, 2 for harder")
    n_recs: int = Field(gt=0, le=100,
                        description="Number of recommendations must be greater than 0 and less or equal to 100")
