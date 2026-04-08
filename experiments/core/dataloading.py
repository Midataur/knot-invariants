from utilities import *
from torch.utils.data import DataLoader
from dataset_types import DATASETS
import numpy as np
import os

# subset can be "train", "val", or "test"
def create_dataset(subset, config, verbose=False):
    DataSetType = DATASETS[config["dataset_type"]]

    # sometimes this can be "verbose AND some other condition"
    should_speak = verbose

    if should_speak:
        print(f"Loading {subset} data...")
        
    # TODO: make this use config["path"] instead of .
    path = f"./datasets/{config['dataset']}/{subset}"
    
    # grab the list of files
    filenames = os.listdir(path)

    # sort files into inputs and targets
    input_files = []
    target_files = []

    for filename in filenames:
        if "input" in filename:
            input_files.append(filename)
        else:
            target_files.append(filename)

    input_files = sorted(input_files)
    target_files = sorted(target_files)

    # pair up the inputs and targets
    if len(input_files) != len(target_files):
        raise Exception(
            "Number of input files does not match number of target files"
            +f" ({len(input_files)} != {len(target_files)})"
        )
    
    file_pairs = zip(input_files, target_files)

    dataset = DataSetType(config)

    for input_filename, target_filename in file_pairs:
        input_data = np.loadtxt(f"{path}/{input_filename}", delimiter=",", dtype=int)
        target_data = np.loadtxt(f"{path}/{target_filename}", delimiter=",", dtype=int)

        dataset.append(input_data, target_data)
        
    return dataset

# subset can be "train", "val", or "test"
def get_dataset_and_loader(subset, config, verbose=False):
    if verbose:
        print(f"Creating {subset} dataset...")
    
    dataset = create_dataset(subset, config, verbose)

    batchsize, n_workers = config["batchsize"], config["n_workers"]
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers)

    return dataset, dataloader