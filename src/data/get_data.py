import os
import zipfile

import click as click

import gdown
from tqdm import tqdm

DEFAULT_PURPOSE = "context"
CONTEXT_GDRIVE = "https://drive.google.com/drive/folders/1J_DtTpxK939scBWaADvIOCdwUxS-UwNu"
GENERAL_GDRIVE = "https://drive.google.com/drive/folders/11wBSUb3589b-Fo4pRBbhxvXWvvf0i0-b"

GENERAL_DIR = "data/external/general_recs"


def get_data_for_context_recs(url: str):
    gdown.download_folder(url, quiet=True, output="data/external")
    os.system("mv data/external/MLP256_last.pth models/")


def get_data_for_general_recs(url: str, dest: str):
    os.makedirs(dest, exist_ok=True)
    files = gdown.download_folder(url, quiet=True, output=dest)
    if not files:
        raise RuntimeError("Unable to download files from Google Drive")
    to_unzip = [file for file in files if file.endswith(".zip")]
    for file in tqdm(to_unzip, desc="Unzipping files"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(dest)
            os.remove(file)


@click.command()
@click.option("--purpose", "-p", default=DEFAULT_PURPOSE, type=str,
              help="Purpose the data is downloaded for (context | general)")
def main(purpose: str):
    if purpose == "context":
        get_data_for_context_recs(CONTEXT_GDRIVE)
    elif purpose == "general":
        get_data_for_general_recs(GENERAL_GDRIVE, GENERAL_DIR)
    else:
        print("Incorrect downloading purpose")


if __name__ == "__main__":
    main()
