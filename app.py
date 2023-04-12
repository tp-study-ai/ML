import logging
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder

import uvicorn
from starlette.responses import JSONResponse

from src.features.preprocess import clean_code
from utils import load_bert, load_mlp, load_faiss
from validate import Submission

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
rating_eps = 0.4

lfm = None


@app.get("/ml/")
def main():
    return "Welcome to StudyAI ML server!"


@app.on_event("startup")
def set_up():
    # Context model
    logger.info("Setting up context model")
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

    # General model
    logger.info("Setting up general model")
    global lfm
    lfm =


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
