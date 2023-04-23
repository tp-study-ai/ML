import json

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
    return tags_dict
