import opendatasets as od

if __name__ == "__main__":
    dataset = "https://www.kaggle.com/datasets/robertkhazhiev/codeforces-problems"
    od.download(dataset, data_dir="data/external")
