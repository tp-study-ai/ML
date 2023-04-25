import os

import gdown

if __name__ == "__main__":
    gdown.download_folder("https://drive.google.com/drive/folders/1J_DtTpxK939scBWaADvIOCdwUxS-UwNu", quiet=True,
                          output="data/external")
    gdown.download_folder("https://drive.google.com/drive/folders/1YSkz2QULz6sVNcPp2JHagDu3D-_51ITf", quiet=True,
                          output="data/external")
    os.system("mv data/external/MLP256_last.pth models/")
