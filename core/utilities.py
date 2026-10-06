from accelerate import load_checkpoint_and_dispatch
from collections import defaultdict as dd
from collections.abc import Iterable
from accelerate import Accelerator
from constants_and_types import *
import torch
import pickle
import os

### ML RELATED ###

def get_save_dir(path, model_name):
    """
        Returns the checkpoint save directory.
    """

    return f"{path}/model_saves/{model_name}_checkpoint"

def save_state_and_config(config: ConfigObject, accelerator: Accelerator):
    """
        Saves a model and the related config.
    """
    # define save location
    path = config.PATH
    model_name = config.model_name
    save_directory = get_save_dir(path, model_name)

    # save the model
    accelerator.save_state(output_dir=save_directory)

    # save the config
    with open(f"{save_directory}/{CONFIG_FILE_NAME}", "wb") as file:
        pickle.dump(config, file)


def try_loading_config(config: ConfigObject):
    """
        Checks if a checkpoint exists and loads the config it if it does.
    """

    # define save location
    path = config.PATH
    model_name = config.model_name
    save_directory = get_save_dir(path, model_name)

    # check if the config exists and load it
    config_file_path = f"{save_directory}/{CONFIG_FILE_NAME}"

    if os.path.isfile(config_file_path):
        # redefine the config
        with open(f"{save_directory}/{CONFIG_FILE_NAME}", "rb") as file:
            config = pickle.load(file)
            print("Loaded config from file, config may be different.")
    else:
        print("Did not load config from file")
    
    return config

def try_loading_state(config: ConfigObject, accelerator: Accelerator):
    """
        Checks if a checkpoint exists and loads it if it does.
    """

    # define save location
    path = config.PATH
    model_name = config.model_name
    save_directory = get_save_dir(path, model_name)

    # try loading the state
    if os.path.isfile(save_directory):
        model = accelerator.load_state(input_dir=save_directory)

def format_for_pytorch_geo(to_format, new_shape=None, new_type=torch.float):
    """
        Formats a list into a tensor in the format pytorch geometric expects.
    """
    tensor = torch.tensor(to_format)
    
    if new_shape is not None:
        tensor = tensor.reshape(new_shape)
    
    return tensor.t().contiguous().type(new_type)

class SimCLRLoss(torch.nn.Module):
    """
        An implementation of the SimCLR loss function from Chen et al (2020).

        Note: the paper calls this NT-Xent (normalised temperature scaled cross-entropy loss).
    """

    def __init__(self, temperature: float):
        super().__init__()

        # save the paramters for later
        self.temperature = temperature

    def calculate_similarities(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """
            Calculates the normalised dot product similarity matrix.
        """
        # concatenate into one matrix
        combined = torch.cat((first, second))

        # compute normalised dot-product similarity
        row_normalised = torch.nn.functional.normalize(combined) # (2B, E)
        similarities =  row_normalised @ row_normalised.transpose(0, 1) # (2B, 2B)

        return similarities

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """
            Assumes that `first` and `second` are tensors of shape `(B,E)`, where
            `B` is the batch-size and `E` is the final embedding dimension.
        """
        # compute the logits (the arguments for the exponentials)
        similarities = self.calculate_similarities(first, second)
        logits = similarities/self.temperature # (2B, 2B)

        # set the diagonal to -infty.
        # this has the same effect as the the denominator
        # indicator function in the original paper.
        logits -= torch.zeros(logits.shape).fill_diagonal_(float("inf"))

        # create a tensor describing where the other thing in the pair is.
        # this acts as the "class label" for cross entropy loss.
        num_rows = first.shape[0]
        index = torch.arange(num_rows)
        targets = torch.cat((index+num_rows, index))

        # compute the cross entropy loss
        return torch.nn.functional.cross_entropy(logits, targets)        

### NON-ML ###

def size_signature(set_to_count: set):
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

def show_list_diff(list1: list, list2: list):
    """
        Highlights differences between two lists.

        Useful for debugging.
    """

    START_COLOR = "\033[93m"
    END_COLOR = "\033[0m"

    # avoid mutations
    list1 = list(list1)
    list2 = list(list2)

    # pad the lists to be the same length
    len_diff = len(list1) - len(list2)

    if len_diff < 0:
        list1 += [" " for x in range(abs(len_diff))]
    elif len_diff > 0:
        list2 += [" " for x in range(abs(len_diff))]

    # display the lists
    display1 = "["
    display2 = "["

    for item1, item2 in zip(list1, list2):
        to_add_1 = str(item1)
        to_add_2 = str(item2)

        diff = len(to_add_1) - len(to_add_2)
        if diff < 0:
            to_add_1 += " "*abs(diff)
        elif diff > 0:
            to_add_2 += " "*abs(diff)

        if item1 != item2:
            to_add_1 = f"{START_COLOR}{to_add_1}{END_COLOR}"
            to_add_2 = f"{START_COLOR}{to_add_2}{END_COLOR}"
        
        display1 += f"{to_add_1}, "
        display2 += f"{to_add_2}, "
    
    print(display1[:-2]+"]")
    print(display2[:-2]+"]")

def cyclic_shift(to_shift: list, n: int = 1):
    """
        Takes a lift and cylicly shifts the lift by n places right.

        By default, n=1.
    """

    return [
        to_shift[(x-n)%len(to_shift)] for x in range(len(to_shift))
    ]

def unzip(iterable: Iterable, num_lists_expected: int = 2):
    """
        The inverse of the default python zip function.

        Returns a list of lists.
    """

    final_lists = [[] for x in range(num_lists_expected)]

    for item in iterable:
        for pos, x in enumerate(item):
            final_lists[pos].append(x)

    return final_lists

def pad_list(list_to_pad: list, desired_length: int, padding_element=0, strict=False):
    """
        Pads a list to be a desired length. 
        Assumes that `len(list_to_pad) <= desired_length` already.
    """

    # sanity check
    if len(list_to_pad) > desired_length:
        raise Exception(f"Expected list of at most length {desired_length} but got one of length {len(list_to_pad)}")

    remaining_length = desired_length - len(list_to_pad)

    return list_to_pad + [padding_element for x in range(remaining_length)]

def sort_knots(knots: list[Knot]):
    """
        Sorts a list of knots by crossing count,
        with `knot_id` as a tiebreaker.
    """

    return sorted(
        knots,
        key=lambda x: (len(x.pd_code), x.knot_id) # use knot id for tie breaker
    )