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
    n_attempts: Optional[int] = Field(default=0, ge=0, description="Number of attempts to solve the problem")


class UserStory(BaseModel):
    solved: list[Problem] = Field(default=[], description="List of problems the user has solved")
    too_easy: list[Problem] = Field(default=[], description="List of problems the user has marked as too easy")
    too_hard: list[Problem] = Field(default=[], description="List of problems the user has marked as too hard")


class ProblemResponse(BaseModel):
    problem_url: str = Field(description="URL of the problem")
    rating: int = Field(default=0, ge=0, le=3500, description="Rating of the problem")
    tags: list[int] = Field(unique_items=True, description="List of tags")


class UserHeuristicResponse(BaseModel):
    recommended_tag: int = Field(ge=1, le=37, description="Tag by what the task is recommended")
    priority: int = Field(ge=1, le=37, description="Priority of tasks by this recommended tag")
    problems: list[ProblemResponse] = Field(..., description="List of problems with this recommended tag")


class ProgressOnTag(BaseModel):
    tag: int = Field(ge=1, le=37, description="TagID")
    done: bool = Field(description="Bool flag. If true, it represents that tag is fulfilled during the cold start")


class ColdStartResponse(BaseModel):
    problem_url: str = Field(description="Problem recommended to be solved")
    tag: int = Field(ge=1, le=37, description="TagID by what the task is recommended")
    progress: list[ProgressOnTag] = Field(unique_items=True,
                                          description="list of {ProgressOnTag for each tag we want to fill}")
    problem_tags: list[int] = Field(unique_items=True, description="List of tags of the recommended problem")
    rating: int = Field(default=0, ge=0, le=3500, description="Rating of the recommended problem")
