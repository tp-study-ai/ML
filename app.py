import ast
import logging
import sys

import faiss
import numpy
import numpy as np
import pandas as pd
import torch.nn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.encoders import jsonable_encoder

import uvicorn
from starlette.responses import JSONResponse

from src.features.preprocess import clean_code
from utils import load_bert, load_mlp, load_faiss, load_tags, load_tags2id
from validate import Submission, User, UserHeuristicResponse, ProblemResponse

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "%(asctime)s: %(levelname)s: %(name)s: %(message)s"
)
handler.setFormatter(formatter)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

app = FastAPI()

mlp = None
bert_transform = None
faiss_index = None
keys_df = None
pwt_df = None
merged_df = None
id2tags = None
np_vector = None
tags2id = None
rating_eps = 0.4


@app.get("/ml/")
def main():
    return "Welcome to StudyAI ML server!"


@app.on_event("startup")
def set_up():
    load_data_for_general_recs()
    load_data_for_context_recs()
    load_models_for_context_recs()


def load_data_for_general_recs():
    global faiss_index
    global id2tags
    global np_vector
    if faiss_index is None:
        logger.debug("Loading FAISS")
        faiss_index = load_faiss()
        logger.debug("FAISS loaded successfully")
    if id2tags is None:
        logger.info("Loading tags")
        id2tags = load_tags()
        logger.info("Tags loaded successfully")
    if np_vector is None:
        np_vector = faiss_index.reconstruct_n(0, faiss_index.ntotal)

    return faiss_index, id2tags, np_vector


def load_data_for_context_recs():
    global keys_df
    global pwt_df
    global merged_df
    global tags2id
    if keys_df is None:
        logger.info("Loading keys dataframe")
        keys_df = pd.read_csv("data/external/keys_df.csv")
        logger.info("Keys loaded successfully")
    if pwt_df is None:
        logger.info("Loading problems dataframe")
        pwt_df = pd.read_csv("data/external/final_df.csv", index_col=0)
        logger.info("Problems loaded successfully")
    if merged_df is None:
        logger.info("Merging data")
        merged_df = pd.merge(keys_df, pwt_df, how="left", on='problem_url')
        logger.info("Merged data successfully")
    if tags2id is None:
        logger.info("Loading tags to id")
        tags2id = load_tags2id()
        logger.info("Loaded tags2id successfully")
    return keys_df, merged_df, tags2id


@app.post("/ml/user_heuristic", response_model=list[UserHeuristicResponse])
async def user_heuristic(user: User,
                         general_data: tuple[faiss.Index, dict, numpy.ndarray] = Depends(load_data_for_general_recs),
                         context_data: tuple[pd.DataFrame, pd.DataFrame, dict] = Depends(load_data_for_context_recs)
                         ) -> list[UserHeuristicResponse]:
    merged_df, tags2id = context_data[1:]

    faiss_index, id2tags, np_vector = general_data

    not_approved_tags = {}
    approved_tasks = {}
    not_approved_tasks = []

    user_story = {value["eng"]: [] for value in id2tags.values()}
    for problem in user.story:
        if problem.difficulty_match < 0 or problem.solved:
            for tag_id in problem.tags:
                tasks_with_url = merged_df[merged_df["problem_url"] == problem.problem_url].index
                if len(tasks_with_url) > 0:
                    user_story[id2tags[str(tag_id)]["eng"]].append(tasks_with_url[0])
        elif problem.difficulty_match > 0 and not problem.solved:
            tasks_with_url = merged_df[merged_df["problem_url"] == problem.problem_url].index
            if len(tasks_with_url) > 0:
                not_approved_tasks.extend(list(tasks_with_url))

    for value in id2tags.values():
        cur_tag = value["eng"]
        if len(user_story[cur_tag]) != 0:
            # get 0.30 the hardest tasks
            tr = merged_df.iloc[user_story[cur_tag]].sort_values("rating")
            thresh_rating = tr.quantile(q=0.7, numeric_only=True).rating
            indexes = tr[tr.rating >= thresh_rating]["rating"].index

            # get 10 nearest tasks with current tag to mean of 0.3 hardest
            result_indexes = faiss_index.search(np.stack(tuple(np_vector[i] for i in indexes)
                                                         ).mean(axis=0, keepdims=True), k=100)[1]
            res_df = merged_df.iloc[result_indexes[0]]
            res_df = res_df[res_df.tags.str.contains(cur_tag.replace("*", "")) & (res_df.rating >= thresh_rating)][
                     :10]
        else:
            thresh_rating = 0
            res_df = merged_df[merged_df.tags.str.contains(cur_tag.replace("*", "")).fillna(False)].sort_values(
                "rating")[:10]
        for i in res_df.index:
            if i in not_approved_tasks:  # User already marked this task as "too hard"
                continue
            tags_i = ast.literal_eval(res_df["tags"][i])
            is_approve = True
            for tag in tags_i:
                if tag != cur_tag:
                    tag_df = merged_df.iloc[user_story[tag]].sort_values("rating")
                    if not merged_df["rating"][i] < tag_df[tag_df.rating > tag_df.quantile(q=0.7, numeric_only=True)
                            .rating]["rating"].median() + 300:
                        is_approve = False
                        if tag not in not_approved_tags:
                            not_approved_tags[tag] = 1
                        else:
                            not_approved_tags[tag] += 1
            if is_approve:
                if res_df.rating[i] > thresh_rating and i not in user_story[cur_tag]:
                    if tag not in approved_tasks:
                        approved_tasks[tag] = [i]
                    else:
                        approved_tasks[tag].append(i)
            else:
                not_approved_tasks.append(i)

    # return approved_tasks
    user_heuristic_response = []
    priority = 1
    for tag, _ in sorted(not_approved_tags.items(), key=lambda x: x[1]):
        if tag not in approved_tasks:
            continue
        rec_by_tag = {
            "priority": priority,
            "recommended_tag": tags2id[tag],
            "problems": []
        }
        for problem_id in approved_tasks[tag]:
            rec_by_tag["problems"].append(
                {
                    "problem_url": merged_df.problem_url.iloc[problem_id],
                    "rating": merged_df.rating.iloc[problem_id],
                    "tags": [tags2id[t] for t in eval(merged_df.tags.iloc[problem_id])]
                }
            )
        user_heuristic_response.append(rec_by_tag)
        approved_tasks.pop(tag)
        priority += 1

    for tag, problems in approved_tasks.items():
        rec_by_tag = {
            "priority": priority,
            "recommended_tag": tags2id[tag],
            "problems": []
        }
        for problem_id in approved_tasks[tag]:
            rec_by_tag["problems"].append(
                {
                    "problem_url": merged_df.problem_url.iloc[problem_id],
                    "rating": merged_df.rating.iloc[problem_id],
                    "tags": [tags2id[t] for t in eval(merged_df.tags.iloc[problem_id])]
                }
            )
        user_heuristic_response.append(rec_by_tag)

    return user_heuristic_response


def load_models_for_context_recs():
    global bert_transform
    global mlp
    global faiss_index
    if bert_transform is None:
        logger.debug(f"Loading codeBERTcpp")
        bert_transform = load_bert()
        logger.debug("BERT loaded successfully")
    if mlp is None:
        logger.debug(f"Loading MLP256")
        mlp = load_mlp()
        logger.debug("MLP loaded successfully")
    if faiss_index is None:
        logger.debug("Loading FAISS")
        faiss_index = load_faiss()
        logger.debug("FAISS loaded successfully")
    return bert_transform, mlp, faiss_index


@app.post("/ml/get_similar", response_model=list[ProblemResponse])
async def get_similar(submission: Submission,
                      models: tuple[torch.nn.Module, torch.nn.Module, faiss.Index] = Depends(
                          load_models_for_context_recs),
                      data: tuple[pd.DataFrame, pd.DataFrame, dict] = Depends(load_data_for_context_recs)) -> \
        list[ProblemResponse]:
    bert_transform, mlp, faiss_index = models

    keys_df, merged_df, tags2id = data

    sub_dict = submission.dict()
    source_code = sub_dict["source_code"]
    cleaned_code = clean_code(source_code)
    logger.info("Code has been extracted and cleaned")
    problem_url = sub_dict["problem_url"]
    difficulty = sub_dict["difficulty"]
    n_recs = sub_dict["n_recs"]
    rating = sub_dict["rating"]

    try:
        emb = mlp(bert_transform(cleaned_code[0]))
        res = faiss_index.search(emb.detach().numpy().reshape(-1, 256), k=500)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong"
        )
    logger.info("Recommendation successful. Filtering...")
    to_filter = res[1][0]
    response = []

    for i in to_filter:
        if keys_df.problem_url[i] == problem_url:
            continue
        if rating * (1 - rating_eps) <= merged_df.rating[i] <= rating + 100 and difficulty == 0 or \
                rating - 100 <= merged_df.rating[i] <= rating * (1 + rating_eps) and difficulty == 2 or \
                rating * (1 - rating_eps) <= merged_df.rating[i] <= rating * (1 + rating_eps) and difficulty == 1:
            response.append(
                {
                    "problem_url": f"{keys_df.problem_url[i]}",
                    "rating": merged_df.rating[i] if str(merged_df.rating[i]) != "nan" else 0,
                    "tags": [tags2id[tag] for tag in eval(merged_df.tags[i])]
                }
            )
            if len(response) == n_recs:
                break

    return response


@app.get("/ml/health")
def health():
    if mlp is None or bert_transform is None or faiss_index is None or keys_df is None or pwt_df is None or merged_df is None:
        raise HTTPException(
            status_code=500,
            detail="Entities undefined"
        )
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"detail": "Model is ready. Everything is OK"})
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
