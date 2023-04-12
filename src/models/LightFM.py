import os

import numpy as np
from lightfm import LightFM
import pandas as pd

DATA_DIR = "data/external/general_recs/"


def str2list(x):
    return x.strip("[]").replace("'", "").split(", ")


class LightFMRecSys:
    def __init__(self):
        self.author_urls = None
        self.problem_urls = None
        self.author_features = None
        self.problem_features = None
        self.problems = None
        self.authors = None
        self.submissions = None
        self.load_data()

    def load_data(self):
        self.submissions = pd.read_csv(os.path.join(DATA_DIR, "cf-cpp-submissions-wo-code.csv"), index_col=0)
        self.authors = pd.read_csv(os.path.join(DATA_DIR, "authors.csv"), index_col=0)
        self.authors.drop(columns="full_name", inplace=True)
        self.problems = pd.read_csv("../data/external/general_recs/final_df.csv", index_col=0,
                                    converters={"tags": str2list, "tag_ids": str2list})

    def preprocess(self):
        self.problems["filled_rating"] = self.problems.apply(
            lambda row: row.rating if not row.rating != row.rating else self.problems[
                self.problems.rating.notna() & (
                        self.problems["index"].str.slice(stop=1) == row["index"][:1])].rating.median(),
            axis=1)

        self.problems["rating_labels"] = pd.cut(self.problems.filled_rating,
                                                bins=[799, 1199, 1599, 1899, 2199, 2499, 2899, 3501],
                                                labels=["newbie", "student", "expert", "master", "international master",
                                                        "grandmaster", "legendary grandmaster"])

    def get_features(self):
        self.problem_features = list(self.problems.rating_labels.unique())
        self.problem_features.remove(np.nan)

        tags = set()

        for problem_tags in self.problems.tags:
            for tag in problem_tags:
                tags.add(tag)

        self.problem_features.extend(list(tags))

        self.author_features = list(
            set(list(self.authors.rating_text.unique())).union(set(list(self.authors.rating_max_text.unique()))))

        self.author_features.remove(np.nan)

        self.problem_urls = list(
            set(self.problems.problem_url.unique()).union(set(self.submissions.problem_url.unique())))

        self.author_urls = list(set(self.authors.author_url.unique()).union(self.submissions.author_url.unique()))


def init_model(interactions, user_features, item_features, weights):
    model = LightFM(no_components=150, learning_rate=0.05, loss="warp", random_state=2023)
    model.fit(interactions=interactions, user_features=user_features, item_features=item_features,
              sample_weight=weights,
              epochs=5, num_threads=8, verbose=True)
