from collections.abc import Callable, Sequence
from torch.optim import AdamW
from typing import NamedTuple

CONFIG_FILE_NAME = "config.pickle"
MODEL_FILE_NAME = "model.safetensors"

# some standard conventions
INCOMING = -1
OUTGOING = 1

STANDARD = -1
REVERSED = 1

UNDERCROSSING = -1
OVERCROSSING = 1

LEFT = -1
RIGHT = 1   

EDGES_PER_NODE = 4

WANDB_LEGAL_TYPES = (int, str, list, float, bool)

class Knot(NamedTuple):
    """
        A bunch of data related to a knot.
    """

    knot_id: str
    pd_code: list[int]
    sym_type: str

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
    extra_notes: str          # any extra details that might be useful to know later.
    dataset_type: str         # the type of pytorch dataset to use, as in the datasets.py file.
     
    random_seed: int          # the seed for the random number generator

    # the next few are model parameters
 
    n_embed: int              # internal embedding dimension. good starting value: 402.
    dropout: int              # dropout factor to use. i usually set this to zero.
    n_blocks: int             # number of blocks to have. higher means a deeper network.
    n_heads: int | None       # the number of attention heads to have if we're using a transformer.

    proj_dim: int             # the dimension to project down to for the SimCLR loss function.
                              # this shouldn't be bigger than n_embed.

    # the next few are training parameters

    optimizer: any            # the optimiser to use. expects a partial function.
    scheduler: any            # the scheduler to use. expects a partial function.
    batchsize: int            # common bottleneck for training speed. good starting value: 64. 
    logging_frequency: int    # how frequently to log expensive operations, such as the similarity matrix.

    simclr_temp: float        # the temperature used in the simclr loss function.

    # the next few are mixer parameters

    mixer_to_use: PDMixer     # the mixing function used to transform the pd codes. 
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

    def as_wandb_legal_dict(self):
        """
            Returns the named tuple as a dictionary (like :meth:`_asdict`),
            but replace all the non-wandb-legal types with the string reps.
        """

        # get the dict rep
        dict_rep = self._asdict()

        # deal with illegal types
        safe_dict = dict()

        for key, value in dict_rep.items():
            if type(value) in WANDB_LEGAL_TYPES or value is None:
                # we're chilling
                safe_dict[key] = value
            else:
                # we're not chilling
                safe_dict[key] = str(value)

        return safe_dict

    def get_max_input_size(self):
        """
            
        """

    def get_transformer_details(self):
        """
            Returns `(maximum input size required, vocabulary size required)`
            if we're using a transformer.
        """
        
        max_input_size = self.max_crossings * EDGES_PER_NODE

        # each label will appear twice in a max length code
        # the +1 is for the special empty space token
        vocab_size  = max_input_size//2 + 1
        return (max_input_size, vocab_size)

    def get_empty_token(self):
        """
            Returns the token used to pad the inputs. 
            Can't ever represent an edge label.
        """

        _, vocab_size = self.get_transformer_details()

        # this will always be the last token in the (zero-indexed) dictionary
        return vocab_size-1