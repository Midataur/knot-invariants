from pd_utils import EDGES_PER_NODE, pd_canonical_form
from constants_and_types import Knot, ConfigObject
import torch
import functools
import mixer

# set up the mixer
max_crossings = 13
n_mix_steps = 1

hamiltonian = functools.partial(
    mixer.crossing_hamiltonian_with_cutoff,
    max_crossings = max_crossings
)

def pdcf_with_kwargs(pd_code, **kwargs):
    return pd_canonical_form(pd_code)

mixer_to_use = functools.partial(
    mixer.hamiltonian_mixer,
    n_steps=n_mix_steps,
    hamiltonian=hamiltonian,
    end_step=pdcf_with_kwargs
)

TRAIN_NO_MORE_THAN = 11

# set up the train set decider
def get_train_set(knots: list[Knot]):
    """
        Enfore a maximum crossing count on seeds knots in the training set.
    """

    return list(filter(
        lambda x: len(x.pd_code)//EDGES_PER_NODE <= TRAIN_NO_MORE_THAN,
        knots
    ))

# the actual config object i'm using for this run.
# documentation of parameter meanings can be found
# in the custom_types file.
CONFIG = ConfigObject(
    model_name        = "1-move-no-RL-no-sym-LT11-5",
    model_type        = "BasicTransformer",
    raw_db_filename   = "katlas.rdf",
    data_url          = "http://katlas.org/Data/katlas.rdf.gz",
    dataset_type      = "knotdatawithtransforms",
    wandb_project     = "knot-simclr",
    random_seed       = 42,

    extra_notes       = f"Only training on knots with at most {TRAIN_NO_MORE_THAN} crossings.",

    get_train_set     = get_train_set,
    n_embed           = 900,
    n_heads           = 6,
    dropout           = 0,
    n_blocks          = 4,

    proj_dim          = 900,

    max_crossings     = max_crossings,
    n_mix_steps       = n_mix_steps,
    mixer_to_use      = mixer_to_use,

    optimizer         = torch.optim.AdamW,
    learning_rate     = 3*(10**-5), 
    logging_frequency = 30,
    batchsize         = 8192,
    weight_decay      = 0.1, 
    momentum          = 0,
    lr_factor         = 0.1, 
    lr_patience       = 30,
    threshold         = 0.01,
    simclr_temp       = 0.05,
    n_workers         = 0,
  
    PATH              = "."
)

# sanity check
assert CONFIG.n_embed % CONFIG.n_heads == 0

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)