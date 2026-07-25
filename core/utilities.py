from accelerate import load_checkpoint_and_dispatch
from collections import defaultdict as dd
import torch
import model_types
import pickle
import os

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

### ML RELATED ###

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

def format_for_pytorch_geo(to_format, new_shape=None, new_type=torch.float):
    """
        Formats a list into a tensor in the format pytorch geometric expects.
    """
    tensor = torch.tensor(to_format)
    
    if new_shape is not None:
        tensor = tensor.reshape(new_shape)
    
    return tensor.t().contiguous().type(new_type)

def size_signature(set_to_count):
    """
        Takes in a set of tuples.

        Gives the number of tuples of various lengths.

        Useful for debugging.
    """

    freqs = dd(int)

    for item in set_to_count:
        freqs[len(item)] += 1
    
    return sorted(freqs.items())

def color_function(start: int, end: int):
    """
        Edge coloring piecewise function.

        Swapping both crossing types is the same as
        multiplying by -1. 

        See master's notes: The Garbali-Gauss construction.
    """
    start_is_positive = start > 0
    end_is_positive = end > 0

    match (start_is_positive, end_is_positive):
        case (False, False):
            return -2
        case (False, True):
            return -1
        case (True, False):
            return 1
        case (True, True):
            return 2
    
    raise Exception(f"Invalid edge type ({start},{end}).")

def inverse_color_function(color: int):
    "The inverse of color function"

    match color:
        case -2:
            return (-1, -1)
        case -1:
            return (-1, 1)
        case 1:
            return (1, -1)
        case 2:
            return (1, 1)

    raise Exception("Invalid color given")

# takes the color (a,b) and gives you (b,a)
def reverse_edge_color(color: int):
    if abs(color) == 1:
        return -color
    
    return color

def show_list_diff(list1, list2):
    """
        Highlights differences between two lists.

        Useful for debugging.
    """

    display1 = "["
    display2 = "["

    for item1, item2 in zip(list1, list2):
        toadd1 = str(item1)
        toadd2 = str(item2)

        if item1 != item2:
            toadd1 = f"_{toadd1}_"
            toadd2 = f"_{toadd2}_"
        
        display1 += f"{toadd1}, "
        display2 += f"{toadd2}, "
    
    print(display1[:-2]+"]")
    print(display2[:-2]+"]")