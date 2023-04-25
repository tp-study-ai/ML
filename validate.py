from typing import Optional

from pydantic import BaseModel, Field


class Submission(BaseModel):
    source_code: str
    problem_url: str
    rating: int
    difficulty: int = Field(ge=0, le=2, description="0 for easier, 1 for the same, 2 for harder")
    n_recs: int = Field(gt=0, le=100,
                        description="Number of recommendations must be greater than 0 and less or equal to 100")


class Problem(BaseModel):
    problem_url: str = Field(description="URL of the problem")
    rating: int = Field(default=0, ge=0, le=3500, description="Rating of the problem")
    tags: list[int] = Field(unique_items=True, description="List of tags")
    difficulty_match: Optional[int] = Field(default=0, ge=-1, le=1,
                                            description="-1 if too easy, 1 if too hard, 0 if unmarked")
    solved: Optional[bool] = Field(default=False, description="Whether the user has solved the problem")
    n_attempts: Optional[int] = Field(default=0, ge=0, description="Number of attempts to solve the problem")


class User(BaseModel):
    username: str
    story: list[Problem] = Field(description="List of problems the user has solved")
