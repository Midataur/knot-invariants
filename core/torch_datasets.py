from dataset_processing import get_knots
from torch.utils.data import Dataset, DataLoader
from constants_and_types import ConfigObject, Knot
from utilities import pad_list, unzip, sort_knots
from typing import NamedTuple
import urllib.request
import torch
import shutil
import gzip
import os

class TrainingState(NamedTuple):
    """
        An object to store the current training state in
        that can be passed to various functions.
    """

    epoch: int
    current_loss: float

class KnotDataWithTransforms(Dataset):
    def __init__(self, config: ConfigObject, seed_knots: list[Knot]):
        # save the data
        self.max_crossings = config.max_crossings
        self.seed_knots = seed_knots

        # derived quantities
        self.max_input_size, _ = config.get_transformer_details()
        self.empty_token = config.get_empty_token()

        # save the mixer
        self.mixer_to_use = config.mixer_to_use

        # can be used by various

    def __len__(self):
        return len(self.seed_knots)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif type(idx) is int:
            idx = [idx]

        # get the requested knots
        relevant_knots: list[Knot] = [self.seed_knots[index] for index in idx]

        pairs = []

        # get the transformed pairs
        for knot in relevant_knots:
            pair = []

            for x in range(2):
                # mix up the diagram
                transformed_code = self.mixer_to_use(knot.pd_code)

                # save the relabelled code with padding
                pair.append(pad_list(
                    transformed_code, self.max_input_size, self.empty_token
                ))
            
            pairs.append(pair)

        # unzip the pairs
        first_codes, second_codes = unzip(pairs)

        first_codes  = torch.tensor(first_codes,  dtype=int)
        second_codes = torch.tensor(second_codes, dtype=int)

        return (
            first_codes,
            second_codes
        )
    
    def save(self, location):
        torch.save(self, location)
    
# changing the keys here can break backwards compatibility, so be careful
DATASET_TYPES = {
    "knotdatawithtransforms": KnotDataWithTransforms,
}

def get_dataset_and_loader(config: ConfigObject, verbose=False, exclude_torus=True):
    if verbose:
        print(f"Creating dataset...")
    
    # get the database file if required
    db_filename = os.path.join(
        config.PATH, 
        "datasets", 
        "raw_dir", 
        config.raw_db_filename
    )

    zip_filename = f"{db_filename}.gz"

    if not os.path.exists(db_filename):
        if verbose:
            print("Couldn't find db file, downloading new one...")

        # download the file
        urllib.request.urlretrieve(
            config.data_url,
            zip_filename
        )

        # unzip the file
        with gzip.open(zip_filename, "rb") as f_in:
            with open(db_filename, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # delete the zip file
        os.remove(zip_filename)

        if verbose:
            print("Successfully downloaded new dataset.")

    # get the knots
    all_knots = list(get_knots(db_filename).values())

    # filter to only requested
    seed_knots = config.get_train_set(all_knots)

    # remove the torus knots if requested.
    # we do this because sometimes they're just copies of known knots.
    if exclude_torus:
        seed_knots = list(filter(
            lambda x: "T" not in x.knot_id,
            seed_knots
        ))

    # sort the knots for better interpretability
    seed_knots = sort_knots(seed_knots)

    # create the dataset and loader
    DataSetType = DATASET_TYPES[config.dataset_type]

    dataset = DataSetType(config, seed_knots=seed_knots)

    batchsize, n_workers = config.batchsize, config.n_workers
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers, shuffle=False)

    return dataset, dataloader