from dataset_processing import get_knots, graph_from_pd_code, Knot
from torch.utils.data import Dataset, DataLoader
from config_object import ConfigObject
import pd_transformations
import urllib.request
import random
import pd_utils
import torch
import shutil
import gzip
import os

class KnotDataWithTransforms(Dataset):
    def __init__(self, config: ConfigObject, seed_knots: list[Knot]):
        # save the data
        self.max_input_size = self.config.max_crossings*pd_utils.EDGES_PER_NODE
        self.seed_knots = seed_knots

        # save the mixer
        self.mixer = config.mixer

    def __len__(self):
        return len(self.seed_knots)

    def __getitem__(self, index):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get the requested knots
        relevant_knots: list[Knot] = [self.seed_knots[index] for index in idx]

        original_codes    = []
        transformed_codes = []

        # get the transformed pairs
        for knot in relevant_knots:
            original_codes.append(knot.pd_code)

            # mix up the diagram
            transformed_code = self.mixer(knot.pd_code)
            
            # apply a random valid symmetry
            sym_group = pd_transformations.SYMMETRY_GROUP[knot.sym_type]
            chosen_sym = random.choice(sym_group)
            transformed_code = chosen_sym(transformed_code)
        
            transformed_codes.append(transformed_code)

        original_codes    = torch.tensor(original_codes,    dtype=int)
        transformed_codes = torch.tensor(transformed_codes, dtype=int)

        return (
            original_codes,
            transformed_codes
        )
    
    def save(self, location):
        torch.save(self, location)
    
# changing the keys here can break backwards compatibility, so be careful
DATASET_TYPES = {
    "knotdatawithtransforms": KnotDataWithTransforms,
}

def get_dataset_and_loader(config: ConfigObject, verbose=False):
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
    all_knots = config.get_train_set(all_knots)

    # create the dataset and loader
    DataSetType = DATASET_TYPES[config.dataset_type]

    dataset = DataSetType(config)

    batchsize, n_workers = config.batchsize, config.n_workers
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers, shuffle=True)

    return dataset, dataloader