import os

import gdown

if __name__ == "__main__":
    gdown.download_folder("https://drive.google.com/drive/folders/1J_DtTpxK939scBWaADvIOCdwUxS-UwNu", quiet=True,
                          output="data/external")
    gdown.download_folder("https://drive.google.com/drive/folders/1YSkz2QULz6sVNcPp2JHagDu3D-_51ITf", quiet=True,
                          output="data/external")
    gdown.download_folder("https://drive.google.com/drive/folders/1J_Dwtiw3jTO083wqpk_FIln049X6t-j6", quiet=True,
                          output="data/processed")
    os.mkdir("models")
    os.system("mv data/external/MLP256_last.pth models/")
