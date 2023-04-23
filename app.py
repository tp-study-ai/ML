import ast
import logging
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder

import uvicorn
from starlette.responses import JSONResponse

from src.features.preprocess import clean_code
from utils import load_bert, load_mlp, load_faiss, load_tags
from validate import Submission, User

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
tags_dict = None
np_vector = None
rating_eps = 0.4


@app.get("/ml/")
def main():
    return "Welcome to StudyAI ML server!"


@app.on_event("startup")
def set_up():
    logger.info(f"Loading codeBERTcpp")
    global bert_transform
    bert_transform = load_bert()
    logger.info("BERT loaded successfully")
    logger.info(f"Loading MLP256")
    global mlp
    mlp = load_mlp()
    logger.info("MLP loaded successfully")
    logger.info("Loading FAISS")
    global faiss_index
    faiss_index = load_faiss()
    logger.info("FAISS loaded successfully")
    logger.info("Loading keys dataframe")
    global keys_df
    keys_df = pd.read_csv("data/external/keys_df.csv")
    logger.info("Keys loaded successfully")
    logger.info("Loading problems dataframe")
    global pwt_df
    pwt_df = pd.read_csv("data/external/final_df.csv", index_col=0)
    logger.info("Problems loaded successfully")
    logger.info("Merging data")
    global merged_df
    merged_df = pd.merge(keys_df, pwt_df, how="left", on='problem_url')
    logger.info("Merged data successfully")

    global tags_dict
    tags_dict = load_tags()
    logger.info("Tags loaded successfully")

    global np_vector
    np_vector = faiss_index.reconstruct_n(0, faiss_index.ntotal)


@app.post("/ml/user_heuristic")
async def user_heuristic(user: User):
    not_approved_tags = {}
    approved_tasks = {}
    not_approved_tasks = []

    user_story = {value["eng"]: [] for value in tags_dict.values()}
    for problem in user.story:
        for tag_id in problem.tags:
            tasks_with_url = merged_df[merged_df["problem_url"] == problem.problem_url].index
            if len(tasks_with_url) > 0:
                user_story[tags_dict[str(tag_id)]["eng"]].append(tasks_with_url[0])

    for value in tags_dict.values():
        cur_tag = value["eng"]
        if len(user_story[cur_tag]) != 0:
            # get 0.30 the hardest tasks
            tr = merged_df.iloc[user_story[cur_tag]].sort_values("rating")
            thresh_rating = tr.quantile(q=0.7, numeric_only=True).rating
            indexes = tr[tr.rating >= thresh_rating]["rating"].index

            # get 10 nearest tasks with current tag to mean of 0.3 hardest
            result_indexes = \
                faiss_index.search(np.stack(tuple(np_vector[i] for i in indexes)).mean(axis=0, keepdims=True),
                                   k=100)[1]
            res_df = merged_df.iloc[result_indexes[0]]
            res_df = res_df[res_df.tags.str.contains(cur_tag.replace("*", "")) & (res_df.rating >= thresh_rating)][
                     :10]
        else:
            thresh_rating = 0
            res_df = merged_df[merged_df.tags.str.contains(cur_tag.replace("*", "")).fillna(False)].sort_values(
                "rating")[:10]
        for i in res_df.index:
            tags_i = ast.literal_eval(res_df["tags"][i])
            is_approve = True
            for tag in tags_i:
                if tag != cur_tag:
                    tag_df = merged_df.iloc[user_story[tag]].sort_values("rating")
                    if not merged_df["rating"][i] < \
                           tag_df[tag_df.rating > tag_df.quantile(q=0.7, numeric_only=True).rating][
                               "rating"].median() + 300:
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

    return approved_tasks


@app.post("/ml/get_similar")
async def get_similar(submission: Submission):
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
    global rating_eps
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
                    "tags": merged_df.tags[i]
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
