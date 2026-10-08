from pd_utils import EDGES_PER_NODE, pd_canonical_form
from constants_and_types import Knot, ConfigObject
import torch
import functools
import mixer

# set up the mixer
TRAIN_NO_MORE_THAN = 11 # the maximum number of crossings in the og diagram

n_mix_steps = 4
max_crossings = TRAIN_NO_MORE_THAN + n_mix_steps*2

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

# set up the train set decider
def get_train_set(knots: list[Knot]):
    """
        Enforce a maximum crossing count on seeds knots in the training set.
    """

    return list(filter(
        lambda x: len(x.pd_code)//EDGES_PER_NODE <= TRAIN_NO_MORE_THAN,
        knots
    ))

optimiser = functools.partial(
    torch.optim.AdamW,
    lr=3*(10**-5), # learning rate
    weight_decay=0.01
)

MAX_EPOCHS = 14_000

scheduler = functools.partial(
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    T_0=MAX_EPOCHS, # time between restarts
)
# the actual config object i'm using for this run.
# documentation of parameter meanings can be found
# in the custom_types file.

CONFIG = ConfigObject(
    model_name        = "4-move-no-RL-no-sym-LT11-1",
    model_type        = "BasicTransformer",
    raw_db_filename   = "katlas.rdf",
    data_url          = "http://katlas.org/Data/katlas.rdf.gz",
    dataset_type      = "knotdatawithtransforms",
    wandb_project     = "knot-simclr",
    random_seed       = 42,

    extra_notes       = f"Only training on knots with at most {TRAIN_NO_MORE_THAN} crossings.",

    get_train_set     = get_train_set,
    n_embed           = 400,
    n_heads           = 20,
    dropout           = 0,
    n_blocks          = 3,

    proj_dim          = 400,

    max_crossings     = max_crossings,
    n_mix_steps       = n_mix_steps,
    mixer_to_use      = mixer_to_use,

    simclr_temp       = 0.07,
    optimizer         = optimiser,
    scheduler         = scheduler,
    logging_frequency = 100,
    batchsize         = 600,

    max_epochs        = MAX_EPOCHS,
    n_workers         = 0,
    PATH              = "."
)

# sanity check
assert CONFIG.n_embed % CONFIG.n_heads == 0

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)