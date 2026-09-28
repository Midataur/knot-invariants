from config_object import ConfigObject
from dataset_processing import Knot
from pd_utils import EDGES_PER_NODE
import functools
import mixer

# set up the mixer
max_crossings = 15
n_mix_steps = 40

hamiltonian = functools.partial(
    mixer.crossing_hamiltonian_with_cutoff,
    max_crossings = max_crossings
)

mixer = functools.partial(
    mixer.hamiltonian_mixer,
    nsteps=n_mix_steps,
    hamiltonian=hamiltonian
)

NO_MORE_THAN = 10

# set up the train set decider
def get_train_set(knots: list[Knot]):
    """
        Enfore a maximum crossing count on seeds knots in the training set.
    """

    return list(filter(
        lambda x: len(x)//EDGES_PER_NODE <= NO_MORE_THAN,
        knots
    ))

# the actual config object i'm using for this run.
# documentation of parameter meanings can be found
# in the config_object file.
CONFIG = ConfigObject(
    model_name      = "test-1",
    model_type      = "BasicTransformer",
    raw_db_filename = "katlas.rdf",
    data_url        = "http://katlas.org/Data/katlas.rdf.gz",
    dataset_type    = "knotdatawithtransforms",
    wandb_project   = "knot-simclr",
    random_seed     = 42,

    extra_notes     = f"Only training on knots with at most {NO_MORE_THAN} crossings.",

    get_train_set   = get_train_set,
    n_embed         = 402,
    n_heads         = 6,
    dropout         = 0,
    n_blocks        = 8,

    max_crossings   = max_crossings,
    n_mix_steps     = n_mix_steps,
    mixer           = mixer,

    learning_rate   = 3*(10**-5), 
    batchsize       = 8192, 
    weight_decay    = 0.001, 
    lr_factor       = 0.1, 
    lr_patience     = 10, 
    threshold       = 0.01, 
    n_workers       = 0
)

# sanity check
assert CONFIG.n_embed % CONFIG.n_heads == 0

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)