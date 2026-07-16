from utilities import *
from graph_functions import *
import numpy as np
import torch

### the following are reidermeister moves                    ###
### see https://mathworld.wolfram.com/ReidemeisterMoves.html ###
### parity refers to the two possible options for each move  ###
### eg. over twist vs under twist                            ###

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def twist(graph, edge_index: int, parity: int):
    """
        Twists an untwisted edge, adds a crossing.

        This follows the conventions in the masters notes.
    """
    # get the edge data
    c1, c2 = inverse_color_function(graph.edge_attr[edge_index][0])
    source, target = graph.edge_index[edge_index]

    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=True
    )

    # delete the edge
    delete_edge(graph, edge_index, **graph_prep_state._asdict())

    # add the new node
    new_node = add_node(
        graph, -parity,

        **graph_prep_state._asdict()
    )

    # add the three edges and their colors
    new_edges = torch.tensor([
        [source,   new_node],
        [new_node, new_node],
        [new_node, target]
    ])

    new_colors = torch.tensor([
        color_function(c1, parity),
        color_function(parity, -parity),
        color_function(-parity, c2)
    ])

    add_edges(
        graph, new_edges, new_colors,

        **graph_prep_state._asdict()
    )

    return graph

@prep_graph(wants_edges_transposed=True, will_mutate_graph=True)
def untwist(graph, node_index):
    """
        Untwists a twisted edge, removes a crossing.
    """

    if graph.x.shape[0] <= node_index:
        raise Exception("Node index out of bounds.")
    
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
            # it's the incoming edge
            incoming = pos
            prenode = source
            precolor = inverse_color_function(
                graph.edge_attr[pos]
            )[0]
        elif target != node_index:
            # it's the outgoing edge
            outgoing = pos
            postnode = target
            postcolor = inverse_color_function(
                graph.edge_attr[pos]
            )[1]
        else:
            # it's the loop
            loop = pos

    # check that this is actually untwistable
    if loop is None or incoming is None or outgoing is None:
        raise Exception(
            f"Node is not untwistable: loop is {loop}, incoming is {incoming}, and outgoing is {outgoing}."
        )

    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=True
    )
    
    # delete the edges and the node
    batch_delete(
        graph, 
        node_indices=[node_index], 
        edge_indices=[loop, incoming, outgoing],

        **graph_prep_state._asdict()
    )

    # add the new edge
    add_edges(
        graph, 
        new_edges=[(prenode, postnode)],
        new_colors=[color_function(precolor, postcolor)],

        **graph_prep_state._asdict()
    )

    return graph

@prep_graph(wants_edges_transposed=False, will_mutate_graph=True)
def swap_twist(graph):
    """
        Swaps a twisted edge.

        Leaves crossing count unchanged.

        We need this because you can't go below zero crossings in our formulation.
    """

    if len(graph.x) > 1:
        raise Exception("Can only be used on single node graphs")

    return mirror_knot(
        graph,

        edges_start_transposed=False,
        edges_should_end_transposed=False,
        graph_has_been_cloned=True
    )

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


### the following are the 4 natural actions of Z/2Z on a knot diagram ###
### they take in a pytorch geometric graph and apply the transform    ###

# the permuation matrix that swaps the rows
S2_SWAP = torch.tensor([[0,1], [1,0]])

@prep_graph(will_mutate_graph=True, wants_edges_transposed=False)
def reverse_knot(graph):
    """
        Sends K -> -K.
    """
    # reverse all the edges
    graph.edge_index = S2_SWAP @ graph.edge_index

    # change the colors appropriately
    graph.edge_attr = np.vectorize(reverse_edge_color)(graph.edge_attr)

    # convert back to tensor
    graph.edge_attr = torch.tensor(graph.edge_attr)

    return graph

@prep_graph(will_mutate_graph=True, wants_edges_transposed=False)
def mirror_knot(graph):
    """
        Swap the orientations.
    """

    graph.x = -graph.x

    return graph

@prep_graph(will_mutate_graph=True, wants_edges_transposed=False)
def reverse_and_mirror_knot(graph):
    graph_prep_state = GraphPrepState(
        edges_start_transposed=False,
        edges_should_end_transposed=False,
        graph_has_been_cloned=True
    )

    return reverse_knot(
        mirror_knot(
            graph, **graph_prep_state._asdict()
        ),

        **graph_prep_state._asdict()
    )
 
@prep_graph(will_mutate_graph=True, wants_edges_transposed=False)
def identity(graph):
    """
        Effectively an alias for graph.clone() via the @prep_graph wrapper.
    """
    return graph

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