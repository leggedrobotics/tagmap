from pathlib import Path

import numpy as np
import pandas as pd
import yaml


CATEGORY_MAPPING_LINK = "https://raw.githubusercontent.com/niessner/Matterport/master/metadata/category_mapping.tsv"
CATEGORY_INDEX_MAPPING_PATH = (
    Path(__file__).parent.absolute().joinpath("category_index_mapping.yaml")
)


def get_category_index_mapping():
    df = pd.read_csv(CATEGORY_MAPPING_LINK, sep="\t")
    df.replace(np.nan, "", inplace=True)  # replace empty cells with empty string

    category_index_mapping = {}
    for _, row in df.iterrows():
        index = (
            row["index"] - 1
        )  # index actually starts from 0, but in the .tsv file it starts from 1

        mappings = {}
        for key, value in row.items():
            if key == "index":
                continue
            mappings[key] = value

        category_index_mapping[index] = mappings

    with open(CATEGORY_INDEX_MAPPING_PATH, "w") as f:
        yaml.dump(category_index_mapping, f)


def load_category_index_mapping():
    with open(CATEGORY_INDEX_MAPPING_PATH, "r") as f:
        category_index_mapping = yaml.load(f, Loader=yaml.FullLoader)
    return category_index_mapping


if __name__ == "__main__":
    get_category_index_mapping()
