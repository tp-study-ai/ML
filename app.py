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
from utils import load_bert, load_mlp, load_faiss, load_tags, load_tags2id, sample_task, get_task_quantile
from validate import Submission, UserStory, UserHeuristicResponse, ProblemResponse, ColdStartResponse

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
db_df = None
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


def load_data_for_cold_start():
    global db_df
    if db_df is None:
        logger.info("Loading db dataframe")
        db_df = pd.read_csv("data/processed/problems_small.csv", converters={"tags": ast.literal_eval,
                                                                                 "tag_ids": ast.literal_eval})
        logger.info("Db loaded successfully")
    return db_df


@app.post("/ml/cold_start", response_model=ColdStartResponse)
async def cold_start(user_story: UserStory,
                     db_df: pd.DataFrame = Depends(load_data_for_cold_start)) -> ColdStartResponse:
    tags_priority = ["implementation", "greedy", "math", "graphs", 'dp', "strings", "data structures", "sortings",
                     "brute force", "constructive algorithms"]

    solved_tasks = {i: [] for i in range(1, 38)}  # Dicts mapping each task to all of its tags
    too_hard_tasks = {i: [] for i in range(1, 38)}
    too_easy_tasks = {i: [] for i in range(1, 38)}
    n_attempts = {}

    # Fill the dicts
    for task in user_story.solved:
        found = db_df[db_df.problem_url == task.problem_url]
        if len(found) == 0:
            continue
        n_attempts[found.index[0]] = task.n_attempts
        for tag in task.tags:
            if tag not in solved_tasks.keys():
                solved_tasks[tag] = [found.index[0]]
            solved_tasks[tag].append(found.index[0])

    for task in user_story.too_hard:
        found = db_df[db_df.problem_url == task.problem_url]
        if len(found) == 0:
            continue
        n_attempts[found.index[0]] = task.n_attempts
        for tag in task.tags:
            if tag not in too_hard_tasks.keys():
                too_hard_tasks[tag] = [found.index[0]]
            too_hard_tasks[tag].append(found.index[0])

    for task in user_story.too_easy:
        found = db_df[db_df.problem_url == task.problem_url]
        if len(found) == 0:
            continue
        n_attempts[found.index[0]] = task.n_attempts
        for tag in task.tags:
            if tag not in too_easy_tasks.keys():
                too_easy_tasks[tag] = [found.index[0]]
            too_easy_tasks[tag].append(found.index[0])

    tags_progress = {tags2id[tag]: False for tag in tags_priority}

    for tag in tags_priority:
        logger.info("Checking tag: " + tag)
        tag_id = tags2id[tag]
        if len(too_hard_tasks[tag_id]) >= 3:
            # todo: save current state for this tag
            tags_progress[tag_id] = True  # TODO @levbara for all cases
            print(tag, "3 hard")
            continue
        tag_done = False
        for solved_task in solved_tasks[tag_id]:
            if n_attempts[solved_task] >= 7:
                tag_done = True
                break
        if tag_done:
            print(tag, "solved")
            # todo: save current state for this tag
            tags_progress[tag_id] = True  # TODO @levbara for all cases
            continue
        if len(too_hard_tasks[tag_id]) >= 1 and len(solved_tasks[tag_id]) != 0:
            # todo: save current state for this tag
            tags_progress[tag_id] = True  # TODO @levbara for all cases
            print(tag, "hard+solved")
            continue

        max_skipped_rating = -1
        max_solved_rating = -1
        min_hard_rating = -1
        if len(too_easy_tasks[tag_id]) != 0:
            id_of_max_skipped_task = db_df.iloc[too_easy_tasks[tag_id]].rating.idxmax()
            max_skipped_rating = get_task_quantile(id_of_max_skipped_task, tag, db_df)
        if len(solved_tasks[tag_id]) != 0:
            id_of_max_solved_task = db_df.iloc[solved_tasks[tag_id]].rating.idxmax()
            max_solved_rating = get_task_quantile(id_of_max_solved_task, tag, db_df)
        if len(too_hard_tasks[tag_id]) != 0:
            id_of_min_hard_task = db_df.iloc[too_hard_tasks[tag_id]].rating.idxmin()
            min_hard_rating = get_task_quantile(id_of_min_hard_task, tag, db_df)

        tag_df = db_df[db_df.tags.apply(lambda tags: tag in tags)].sort_values(by="rating")
        if max_solved_rating == -1 and max_skipped_rating == -1:
            # get < 0.1 rating quantile
            tag_df["tags_count"] = tag_df.tags.apply(lambda tags: len(tags))

            target_tasks = tag_df[tag_df.rating <= tag_df.quantile(q=0.1, numeric_only=True).rating].sort_values(
                by="tags_count",
                ascending=False)

            logger.info("No solved or skipped tasks for tag: " + tag)
            logger.info("Sampling task for tag: " + tag)
            
            problem_url, rating, tag_ids = sample_task(target_tasks, id2tags.keys(), solved_tasks, too_hard_tasks, too_easy_tasks)

            return {
                "finished": False,
                "problem_url": problem_url,
                "tag": tag_id,
                "progress": [{"tag": tag, "done": status} for tag, status in tags_progress.items()],
                "rating": rating,
                "problem_tags": tag_ids
            }

        elif max_solved_rating > max_skipped_rating:
            # get tasks by solved tasks
            if n_attempts[id_of_max_solved_task] < 4:
                target_rating = max_solved_rating + (1 - max_solved_rating) / 2
            else:
                target_rating = max_solved_rating + (1 - max_solved_rating) / 4

            tag_df["tags_count"] = tag_df.tags.apply(lambda tags: len(tags))

            target_tasks = tag_df[
                (tag_df.rating <= tag_df.quantile(q=min(1, target_rating + 0.02), numeric_only=True).rating) & (
                        tag_df.rating >= tag_df.quantile(q=target_rating - 0.02,
                                                         numeric_only=True).rating)].sort_values(
                by="tags_count", ascending=False)

            problem_url, rating, tag_ids = sample_task(target_tasks, id2tags.keys(), solved_tasks, too_hard_tasks,
                                                       too_easy_tasks)

            return {
                "finished": False,
                "problem_url": problem_url,
                "tag": tag_id,
                "progress": [{"tag": tag, "done": status} for tag, status in tags_progress.items()],
                "rating": rating,
                "problem_tags": tag_ids
            }

        else:
            # todo: get tasks by skipped tasks
            if min_hard_rating == -1:
                # no upper limit
                target_rating = max_skipped_rating + (1 - max_skipped_rating) / 2
            else:
                # with upper limit
                target_rating = max_skipped_rating + (min_hard_rating - max_skipped_rating) / 2

            tag_df["tags_count"] = tag_df.tags.apply(lambda tags: len(tags))

            target_tasks = tag_df[
                (tag_df.rating <= tag_df.quantile(q=min(1, target_rating + 0.02), numeric_only=True).rating) & (
                        tag_df.rating >= tag_df.quantile(q=target_rating - 0.02,
                                                         numeric_only=True).rating)].sort_values(
                by="tags_count", ascending=False)

            problem_url, rating, tag_ids = sample_task(target_tasks, id2tags.keys(), solved_tasks, too_hard_tasks,
                                                       too_easy_tasks)

            return {
                "finished": False,
                "problem_url": problem_url,
                "tag": tag_id,
                "progress": [{"tag": tag, "done": status} for tag, status in tags_progress.items()],
                "rating": rating,
                "problem_tags": tag_ids
            }

    return {
        "finished": True,
        "problem_url": "",
        "tag": 1,
        "progress": [{"tag": tag, "done": status} for tag, status in tags_progress.items()],
        "rating": 0,
        "problem_tags": []
    }


@app.post("/ml/user_heuristic", response_model=list[UserHeuristicResponse])
async def user_heuristic(user: UserStory,
                         general_data: tuple[faiss.Index, dict, numpy.ndarray] = Depends(load_data_for_general_recs),
                         context_data: tuple[pd.DataFrame, pd.DataFrame, dict] = Depends(load_data_for_context_recs)
                         ) -> list[UserHeuristicResponse]:
    merged_df, tags2id = context_data[1:]

    faiss_index, id2tags, np_vector = general_data

    not_approved_tags = {}
    approved_tasks = {}
    not_approved_tasks = []

    user_story = {value["eng"]: [] for value in id2tags.values()}

    for problem in user.solved:
        for tag_id in problem.tags:
            tasks_with_url = merged_df[merged_df["problem_url"] == problem.problem_url].index
            if len(tasks_with_url) > 0:
                user_story[id2tags[tag_id]["eng"]].append(tasks_with_url[0])

    for problem in user.too_easy:
        if problem not in user.solved:
            for tag_id in problem.tags:
                tasks_with_url = merged_df[merged_df["problem_url"] == problem.problem_url].index
                if len(tasks_with_url) > 0:
                    user_story[id2tags[tag_id]["eng"]].append(tasks_with_url[0])

    for problem in user.too_hard:
        if problem not in user.solved:
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
                if tag == cur_tag:
                    continue
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
