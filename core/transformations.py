from utilities import reverse_edge_color
import numpy as np
import torch

# the following are reidermeister moves
# see https://mathworld.wolfram.com/ReidemeisterMoves.html
# parity refers to the two possible options for each move
# eg. over twist vs under twist

# twists an untwisted edge
# adds a crossing
# this is R1
def twist(graph, edge, parity):
    ...

# untwists a twisted edge
# removes a crossing
# this is R1^{-1}
def untwist(graph, node):
    ...

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