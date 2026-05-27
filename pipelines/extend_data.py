import logging
import sys
import os
import json
from pathlib import Path
import logging

import pandas as pd
import numpy as np

# local imports
from src.core.config import load_config, save_config
from src.core.utils import flush_timing, skip_if_exists, timed
from src.core.io import load_df, save_df
from src.domain.plot.style import apply_plot_style, extended_palette
from src.core.log import setup_logger

setup_logger(log_file="resources/logs.txt")
apply_plot_style()
logger = logging.getLogger(__name__)


def main():
    cfg = load_config(
        config_path=Path(__file__).parent.parent / "configs",
        config_name="config",
        overrides=sys.argv[1:],
    )

    # paths
    processed_data_path = Path(cfg.path.processed_data)
    data_logs_path = Path(cfg.path.shared)

    # old code for printing out the new features to paste into the config file
    #with open(str(data_logs_path)+f"/complexity.json") as f:
        #data = json.load(f)
    #feature_df = pd.DataFrame.from_dict(data, orient="index")
    #for col in feature_df.columns:
        #print(f'  - {col}')

    # loop through the processed data directory
    for entry in os.scandir(processed_data_path):
        if entry.is_file():
            
            df = load_df(str(processed_data_path) + f"/{entry.name}")

            # prevent the script from redoing its work
            if "p5_silhouette" in df:
                logger.info(f"{cfg.data.file_name} in {entry.name} is already contains complexity features. Skipping...")
                continue
            
            # load the json file for the related dataset
            with open(str(data_logs_path)+f"/complexity.json") as f:
                data = json.load(f)
            
            # create a df from the json dictionary
            feature_df = pd.DataFrame.from_dict(data, orient="index")
            feature_df.index.name = "cluster"
            feature_df.reset_index(inplace=True)


            # the json stores data as a string, we need int
            feature_df = feature_df.astype("float64")

            # merge the feature_df with the normal df using the cluster column
            df = df.merge(feature_df, on="cluster", how="left")

            # this is something that may be changed.
            # the df has some missing values, maybe there are some samples that
            # dont have clusters associated with them.
            # easy solution is to just fill with mean values.
            df = df.fillna(df.mean(numeric_only=True))

            # override the original dataset to save the new features
            save_path = Path(str(processed_data_path) + f"/{entry.name}")
            save_df(df, str(save_path))

            
            

    

    

if __name__ == "__main__":
    main()



