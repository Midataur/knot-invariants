CONFIG = {
    "modelname": "test-1",
    "model_type": "GCN",

    "dataset": "katlas",
    "dataset_type": "graph",

    # the url to recover the datset from
    "dataset_url": "http://katlas.org/Data/katlas.rdf.gz",

    "wandb_project": "knot-simclr", 

    "random_seed": 42,

    # dataset parameters
    "braid_count": 4,
    "max_word_length": 10,

    # embedding dimension
    # good starting value: 402
    "n_embed": 402,

    # dropout factor to use
    # i usually set to zero
    "dropout": 0,

    # number of blocks to have
    # higher means a deeper network
    "n_blocks": 8,

    # good starting value: 3*10^-4
    "learning_rate": 3*(10**-5), 

    # good starting value: 64
    "batchsize": 8192, 

    # good starting value: 0.1
    "weight_decay": 0.001, 

    # usually 0.1
    "lr_factor": 0.1, 

    # usually 10
    "lr_patience": 10, 

    # usually 0.01
    "threshold": 0.01, 

    # number of workers to use for loading data to the gpus
    # set to 0 for all, +ve for specific
    # usually 0
    "n_workers": 0, 

    # should be . or .. unless you're doing something weird
    "PATH": "..",
}

assert CONFIG["n_embed"] % CONFIG["n_heads"] == 0

if __name__ == "__main__":
    print("Loading libraries...")
    from training import train
    train(CONFIG)