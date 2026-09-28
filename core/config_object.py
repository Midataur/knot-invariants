from collections.abc import Callable, Sequence
from dataset_processing import Knot
from typing import NamedTuple

def identity(knots: list[Knot]):
    return knots

type PDMixer = Callable[[Sequence[int]], Sequence[int]]

# @dataclass
class ConfigObject(NamedTuple):
    model_name: str           # the name of the model
    model_type: str           # the type of the model
 
    raw_db_filename: str      # the filename of the dataset to be used.
    data_url: str             # the url to recover the raw knot data from.
    wandb_project: str        # the name of the weights and biases project.
    extra_note: str           # any extra details that might be useful to know later.
    dataset_type: str         # the type of pytorch dataset to use, as in the datasets.py file.
     
    random_seed: int          # the seed for the random number generator

    # the next few are model parameters
 
    n_embed: int              # internal embedding dimension. good starting value: 402.
    dropout: int              # dropout factor to use. i usually set this to zero.
    n_blocks: int             # number of blocks to have. higher means a deeper network.
    n_heads: int | None       # the number of attention heads to have if we're using a transformer.

    # the next few are training parameters

    learning_rate: float      # good starting value: 3*10^-4.
    batchsize: int            # common bottleneck for training speed. good starting value: 64. 
    weight_decay: float       # good starting value: 0.1.
    lr_factor: float          # the factor by which to reduce lr on plateau. usually 0.1.
    lr_patience: int          # how long to wait before declaring plateau. usually 10.
    threshold: float          # the threshold what counts as a plataeu. usually 0.01.

    # the next few are mixer parameters

    mixer: PDMixer            # the mixing function used to transform the pd codes. 
    max_crossings: int        # the maximum number of crossings allowed
    n_mix_steps: int          # the number of random reidemeister moves applied.
 
    # the next few are parameters with default values

    n_workers: int = 0        # number of workers to use for loading data to the gpus.
                              # set to 0 for "use all", +ve for a specific count.
                              # usually set to 0.

    get_train_set: Callable[[list[Knot]], list[Knot]] = identity
                              # a function that takes in a list of knots and
                              # returns a sub-list of knots that will be used
                              # for the training set. For example, it might
                              # return all knots that had a crossing count
                              # less than 15.
     
    PATH: str = ".."          # the folder path to work in. 
                              # should be ".." unless you're doing something weird.