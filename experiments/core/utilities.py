from torch import argmax
from accelerate import load_checkpoint_and_dispatch
import model_types
import pickle
import os

CONFIG_FILE_NAME = "config.pickle"
MODEL_FILE_NAME = "model.safetensors"

def save_model_and_config(model, config, accelerator):
    """
        Saves a model and the related config.
    """
    # define save location
    path = config["PATH"]
    modelname = config["modelname"]
    save_directory = f"{path}/model_saves/{modelname}"

    # save the model
    accelerator.save_model(model, f"{save_directory}")

    # save the config
    with open(f"{save_directory}/{CONFIG_FILE_NAME}", "wb") as file:
        pickle.dump(config, file)


def try_loading_model(config, surgery_func=None):
    """
        Checks if a model exists and loads it if it does;
        if it doesn't, it creates a fresh one.
        
        Returns (model, config).
    """

    # define save location
    path = config["PATH"]
    modelname = config["modelname"]
    save_directory = f"{path}/model_saves/{modelname}"

    # check if the config exists and load it
    config_file_path = f"{save_directory}/{CONFIG_FILE_NAME}"

    if os.path.isfile(config_file_path):
        # redefine the config
        with open(f"{save_directory}/{CONFIG_FILE_NAME}", "rb") as file:
            config = pickle.load(file)
            print("Loaded config from file, config may be different.")
    else:
        print("Did not load config from file")

    # do surgery if we need to
    # this allows support for legacy models that had bugs
    if surgery_func is not None:
        surgery_func(config)
        print("Did some surgery")

    # create the model template
    ModelType = model_types.MODELS[config["model_type"]]

    model = ModelType(config)

    # try loading the model
    model_file_path = f"{save_directory}/{MODEL_FILE_NAME}"
    
    if os.path.isfile(model_file_path):
        model = load_checkpoint_and_dispatch(model, model_file_path)
    
    return (model, config)