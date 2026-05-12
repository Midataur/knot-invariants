from utilities import *
import numpy as np
import torch

# the following are reidermeister moves
# see https://mathworld.wolfram.com/ReidemeisterMoves.html
# parity refers to the two possible options for each move
# eg. over twist vs under twist

# twists an untwisted edge, adds a crossing
# this follows the conventions in the masters notes
def twist(graph, edge_index: int, parity: int):
    graph = graph.clone()

    # transpose for convenience
    graph.edge_index = graph.edge_index.t()

    # get the edge data
    c1, c2 = inverse_color_function(graph.edge_attr[edge_index][0])
    source, target = graph.edge_index[edge_index]

    # delete the edge
    delete_edge(graph, edge_index)    

    # add the new node
    new_node = add_node(graph, -parity)

    # add the three edges and their colors
    new_edges = torch.tensor([
        [source, new_node],
        [new_node,  new_node],
        [new_node, target]
    ])

    new_colors = torch.tensor([
        color_function(c1, parity),
        color_function(parity, -parity),
        color_function(-parity, c2)
    ])

    add_edges(graph, new_edges, new_colors)

    # transpose back
    graph.edge_index = graph.edge_index.t()

    return graph

# untwists a twisted edge, removes a crossing
def untwist(graph, node_index):
    graph = graph.clone()

    if graph.x.shape[0] <= node_index:
        raise Exception("Node index out of bounds.")
    
    # transpose for convenience
    graph.edge_index = graph.edge_index.t()
    
    # find the attached edges
    loop = None
    incoming = None
    outgoing = None

    for pos, edge in enumerate(graph.edge_index):
        # check if the edge relates to the node
        if node_index not in edge:
            continue
        
        # figure out what kind of edge this is
        source, target = edge

        if source != node_index:
            incoming = pos
            prenode = source
            precolor = inverse_color_function(
                graph.edge_attr[pos]
            )[0]
        elif target != node_index:
            outgoing = pos
            postnode = target
            postcolor = inverse_color_function(
                graph.edge_attr[pos]
            )[1]
        else:
            loop = pos

    # check that this is actually untwistable
    if loop is None or incoming is None or outgoing is None:
        raise Exception(
            f"Node is not untwistable: loop is {loop}, incoming is {incoming}, and outgoing is {outgoing}."
        )
    
    # delete the edges and the node
    batch_delete(
        graph, 
        node_indices=[node_index], 
        edge_indices=[loop, incoming, outgoing]
    )

    # add the new edge
    add_edges(
        graph, 
        new_edges=[(prenode, postnode)],
        new_colors=[color_function(precolor, postcolor)]
    )

    # transpose back
    graph.edge_index = graph.edge_index.t()

    return graph

# swaps a twisted edge
# leaves crossing count unchanged
# we need this because you can't go below zero crossings in our formulation
def swap_twist(graph):
    if len(graph.x) > 1:
        raise Exception("Can only be used on single node graphs")

    return mirror_knot(graph)

# slides one edge over another
# adds two crossings
# this is R2
def poke(graph, edge_1, edge_2, parity):
    ...

# reverse slides on edge over another
# removes two crossings
# this is R2^{-1}
def unpoke(graph, edge_1, edge_2):
    ...

# yang-baxters
# does not change crossings
# lhs to rhs in the mathworld image
def yang_baxter(graph, edge_1, edge_2):
    ...




# the following are the 4 natural actions of Z/2Z on a knot diagram
# they take in a pytorch geometric graph and apply the transform

# the permuation matrix that swaps the rows
S2_SWAP = torch.tensor([[0,1], [1,0]])

# sends K -> -K
def reverse_knot(graph):
    new_graph = graph.clone()

    # reverse all the edges
    new_graph.edge_index = S2_SWAP @ new_graph.edge_index

    # change the colors appropriately
    new_graph.edge_attr = np.vectorize(reverse_edge_color)(new_graph.edge_attr)

    # convert back to tensor
    new_graph.edge_attr = torch.tensor(new_graph.edge_attr)

    return new_graph

def mirror_knot(graph):
    new_graph = graph.clone()

    # swap the orientations
    new_graph.x = -new_graph.x

    return new_graph

def reverse_and_mirror_knot(graph):
    return reverse_knot(mirror_knot(graph))

def identity(graph):
    return graph.clone()

VALID_SYM_TYPES = [
    "Chiral", # no symmetries
    "Fully amphicheiral", # K = -K = K* = -K*
    "Negative amphicheiral", # K = -K*
    "Positively amphicheiral", #K = K*, not actually in the database bc it's rare
    "Reversible" # K = -K
]

# for a given symmetry type, tells you the operations that generate a distinct knot
NEEDED_TRANSFORMS = {
    "Chiral": [identity, reverse_knot, mirror_knot, reverse_and_mirror_knot],
    "Fully amphicheiral": [identity],
    "Negative amphicheiral": [identity, reverse_knot], # note -K = K* for this class
    "Positively amphicheiral": [identity, reverse_knot], # note -K = -K* for this class
    "Reversible": [identity, mirror_knot] # note K* = -K* for this class
}