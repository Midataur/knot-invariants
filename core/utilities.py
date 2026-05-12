from accelerate import load_checkpoint_and_dispatch
import torch
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

# formats a list into a tensor in the format pytorch geometric expects
def format_for_pytorch_geo(to_format, new_shape=None, new_type=torch.float):
    tensor = torch.tensor(to_format)
    
    if new_shape is not None:
        tensor = tensor.reshape(new_shape)
    
    return tensor.t().contiguous().type(new_type)

def color_function(start, end):
    """
        Edge coloring piecewise function.
        Swapping both crossing types is the same as
        multiplying by -1. 
        See master's notes: The Garbali-Gauss construction.
    """
    if start < 0 and end < 0:
        return -2
    elif start < 0 and end > 0:
        return -1
    elif start > 0 and end < 0:
        return 1
    elif start > 0 and end > 0:
        return 2
    
    raise Exception(f"Invalid edge type ({start},{end}).")

def inverse_color_function(color):
    "The inverse of color function"

    if color == -2:
        return (-1, -1)
    elif color == -1:
        return (-1, 1)
    elif color == 1:
        return (1, -1)
    elif color == 2:
        return (1, 1)

    raise Exception("Invalid color given")

# takes the color (a,b) and gives you (b,a)
def reverse_edge_color(color):
    if abs(color) == 1:
        return -color
    
    return color

# the following add some graph functionality

def delete_edge(graph, edge_index):
    """
        Deletes an edge in a graph in place.
        Does not delete any attached nodes.
        Expects edge_index to be tranposed.
    """
    # delete the edge
    graph.edge_index = torch.concat([
        graph.edge_index[:edge_index],
        graph.edge_index[edge_index+1:]
    ])

    # delete the edge color
    graph.edge_attr = torch.concat([
        graph.edge_attr[:edge_index],
        graph.edge_attr[edge_index+1:]
    ])

def delete_node(graph, node_index):
    """
        Deletes a node of a graph in place.
        Does not delete any attached edges, but does relabel them if needed.
    """
    # delete the node
    graph.x = torch.concat([
        graph.x[:node_index],
        graph.x[node_index+1:]
    ])

    # we've shifted a bunch of node indices
    # so now we need to relabel the edges that touch those nodes
    mask = (graph.edge_index > node_index).to(int)
    graph.edge_index -= mask

def batch_delete(graph, node_indices=[], edge_indices=[]):
    """
        Deletes several elements at once in place.
        Does this in reverse order to make sure the indices remain valid.
    """

    # delete the nodes
    for node_index in sorted(node_indices, reverse=True):
        delete_node(graph, node_index)
    
    # delete the edges
    for edge_index in sorted(edge_indices, reverse=True):
        delete_edge(graph, edge_index)

def add_edges(graph, new_edges, new_colors):
    """
        Adds edges and their colors in place.
        Expects edge_index to be transposed.
    """
    if len(new_edges) != len(new_colors):
        raise Exception(
            f"There were {len(new_edges)} edges provided and {len(new_colors)} colors; these must match."
        )
    
    # convert if we need to
    if type(new_edges) is not torch.Tensor:
        new_edges = torch.tensor(new_edges, dtype=int)
    
    if type(new_colors) is not torch.Tensor:
        new_colors = torch.tensor(new_colors, dtype=float)

    # add the edges
    graph.edge_index = torch.concat([
        graph.edge_index,
        new_edges
    ])

    # add the new colors
    graph.edge_attr = torch.concat([
        graph.edge_attr,
        new_colors.reshape((-1,1))
    ])

def add_node(graph, sign):
    """
        Adds node in place.
        Returns the new node index.
    """
    # add the new node
    graph.x = torch.concat([
        graph.x,
        torch.tensor([[sign]])
    ])

    return graph.x.shape[0]-1