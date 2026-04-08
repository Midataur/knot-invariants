from torch.utils.data import Dataset
import torch

class SimpleDataset(Dataset):
    def __init__(self, config):
        self.config = config

        max_word_length = config["max_word_length"]
        
        # this comes from Thiffeault p93
        max_coord_length = 2*config["braid_count"] - 2

        self.inputs = torch.empty((0, max_word_length), dtype=int)
        self.targets = torch.empty((0, max_coord_length), dtype=int)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        sample = (
            self.inputs[index],
            self.targets[index]
        )

        return sample
    
    def save(self, location):
        torch.save(self, location)

    def process_data(self, *args, **kwargs):
        raise NotImplementedError("""
            SimpleDataset cannot be used directly. Instead, please
            create a child class that implements process_data.
        """)

    def append(self, inputs, targets, verbose=False, **kwargs):
        inputs, targets = self.process_data(inputs, targets, verbose=verbose)

        self.raw_append(inputs, targets)
    
    # assumes the data has already been processed
    def raw_append(self, inputs, targets):
        new_inputs = torch.tensor(inputs, dtype=int)
        new_targets = torch.tensor(targets, dtype=int)

        # probably overkill but ah well
        new_inputs._fix_weakref()
        new_targets._fix_weakref()

        self.inputs = torch.cat((self.inputs, new_inputs))
        self.targets = torch.cat((self.targets, new_targets))

        # probably overkill but ah well
        self.inputs._fix_weakref()
        self.targets._fix_weakref()

        # reduce size down to maximum
        if "dataset_cap" in self.config.keys():
            self.inputs = self.inputs[:self.config["dataset_cap"]]
            self.targets = self.targets[:self.config["dataset_cap"]]

class BasicRegression(SimpleDataset):
    def __init__(self, config):
        super().__init__(config)

        # retype targets
        self.targets = self.targets.float()

    def process_data(self, inputs, targets, verbose=False):
        # make sure all tokens are positive, make targets the correct type
        return (
            inputs + (self.config["braid_count"] - 1), 
            targets.astype(float)
        )


DATASETS = {
    "basicregression": BasicRegression,
}