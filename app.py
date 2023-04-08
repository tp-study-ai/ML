import logging
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder

import uvicorn
from starlette.responses import JSONResponse

from utils import load_bert, load_mlp, load_faiss

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


@app.get("/")
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
    keys_df = pd.read_csv("data/external/codeforces-problems/keys_df.csv")


@app.post("/get_similar")
def get_similar(request):
    code = request.code
    logger.info("Code has been extracted")
    try:
        emb = mlp(bert_transform(code))
        res = faiss_index.search(emb.detach().numpy().reshape(-1, 256), k=40)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Something went wrong"
        )
    logger.info("Recommendation successful")
    return [keys_df.problem_url[i] for i in res]


@app.get("/health")
def health():
    if mlp is None or bert_transform is None or faiss_index is None or keys_df is None:
        raise HTTPException(
            status_code=500,
            detail="Entities undefined"
        )
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"detail": "Model is ready. Everything is OK"})
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
