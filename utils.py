import json
import logging

import torch
from transformers import AutoTokenizer, AutoModel
import faiss

from src.models.BertEmbeddingTransform import BertEmbeddingTransform
from src.models.MLP256 import MLP256


def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-cpp")
    BERT = AutoModel.from_pretrained("neulab/codebert-cpp", add_pooling_layer=False)
    BERT.eval()
    if torch.cuda.is_available():
        bert_transform = BertEmbeddingTransform(BERT, tokenizer, 'cuda')
    else:
        bert_transform = BertEmbeddingTransform(BERT, tokenizer, 'cpu')

    return bert_transform


def load_mlp():
    mlp = MLP256()
    if torch.cuda.is_available():
        mlp.to("cuda")
    mlp.load_state_dict(torch.load("models/MLP256_last.pth", map_location=torch.device('cpu')))
    mlp.eval()
    return mlp


def load_faiss():
    faiss_index = faiss.read_index("data/external/task_index_median_MLP256_03-04-23.bin")
    return faiss_index


def load_tags():
    with open("data/external/id2tags.json", "r") as f:
        tags_dict = json.load(f)
    tags_dict = {int(k): v for k, v in tags_dict.items()}
    return tags_dict


def load_tags2id():
    with open("data/external/tags2id.json", "r") as f:
        tags2id = json.load(f)
    return tags2id


# Useful functions for different use cases

def get_task_quantile(task_id, tag, merged_df):
    tag_df = merged_df[merged_df.tags.apply(lambda tags: tag in tags)].sort_values(by="rating")
    tag_df.reset_index(drop=False, inplace=True)
    print(tag_df)
    return tag_df[tag_df['Unnamed: 0'] == task_id].index[0] / len(tag_df)


logger = logging.getLogger()


def count_needed_tags(tags, i, tags_priority):
    cnt = 0
    for tag in tags:
        if tag in tags_priority[i + 1:]:
            cnt += 1
    return cnt


def sample_task(input_df, tag_ids, solved, too_hard, too_easy, tag_ind, tags_priority):
    input_df["needed_tags_count"] = input_df.tags.apply(lambda tags: count_needed_tags(tags, tag_ind, tags_priority))
    input_df = input_df.sort_values(
        by="needed_tags_count",
        ascending=False
    )
    for task in input_df.iloc:
        is_ok = True
        for tag in tag_ids:
            logger.info(f"Checking {task['Unnamed: 0']} for {tag}")
            if task['Unnamed: 0'] in solved[tag] or task['Unnamed: 0'] in too_hard[tag] or task['Unnamed: 0'] in \
                    too_easy[tag]:
                is_ok = False
                break
        if is_ok:
            return task['problem_url'], task['rating'], task['tag_ids']
