import opendatasets as od
import os

if __name__ == "__main__":
    dataset = "https://www.kaggle.com/datasets/robertkhazhiev/codeforces-problems"
    od.download(dataset, data_dir="data/external")
    for file in os.listdir("data/external/codeforces-problems/"):
        os.rename(os.path.join("data/external/codeforces-problems/", file), os.path.join("data/external/", file))
    os.rmdir("data/external/codeforces-problems")
