from torch_geometric.data import InMemoryDataset, download_url, DataLoader
from core.processing import get_knots, get_graphs
import gzip
import os

class GraphDataset(InMemoryDataset):
    def __init__(self, config, transform=None):
        self.datasets_folder = f"{config['PATH']}/datasets"

        super().__init__(self.datasets_folder, transform)

        self.config = config

        self.load(self.processed_paths[0])

    def get_filepath(self, extension):
        return f"{self.config['dataset']}{extension}"

    @property
    def raw_file_names(self):
        return [self.get_filepath(".rdf")]

    @property
    def processed_file_names(self):
        return [self.get_filepath(".pt")]

    def download(self):
        # Download to `self.raw_dir`.
        file_path = download_url(self.config["url"], self.raw_dir)
        gzip.open(file_path)
        os.remove(file_path)

    def process(self):
        raw_file = self.raw_paths[0]

        # extract gauss codes from rdf
        gauss_codes = get_knots(raw_file)

        # read graphs into data_list
        data_list = get_graphs(gauss_codes)

        self.save(data_list, self.processed_paths[0])

DATASETS = {
    "graph": GraphDataset,
}

def get_dataset_and_loader(config, verbose=False):
    if verbose:
        print(f"Creating dataset...")
    
    DataSetType = DATASETS[config["dataset_type"]]

    dataset = DataSetType(config)

    batchsize, n_workers = config["batchsize"], config["n_workers"]
    dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=n_workers, shuffle=True)

    return dataset, dataloader