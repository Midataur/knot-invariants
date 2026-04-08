CONFIG = {
    "modelname": "smallandshort-36",
    "model_type": "RegressionModel",
    "dataset": "smallandshort",
    "dataset_type": "basicregression",
    "random_seed": 42,

    # dataset parameters
    "braid_count": 4,
    "max_word_length": 10,

    # the size to cap the dataset at
    # if the dataset is already smaller than this, this does nothing
    "dataset_cap": 8_000_000,

    # embedding dimension
    # good starting value: 402
    "n_embed": 402,

    # number of attention heads:
    # good starting value: 6
    # n_embed % n_heads must be 0
    "n_heads": 6,

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