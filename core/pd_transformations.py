from utilities import *
from pd_functions import *

"""
    This file contains a lot of the same functions as graph_transformations.py,
    but it does them in terms of planar diagram codes instead.

    The resulting code is dramatically faster.
"""


"""
    The functions below are the reidermeister moves, but for pd codes.
"""

def pd_twist(pd_code, edge_label: int, over_under: int, node_sign: int):
    """
        Twists an untwisted edge, adds a crossing.

        This follows the conventions in the masters notes.
    """

    # copy the code
    pd_code = list(pd_code)

    # find where the edge is referenced
    incoming_pos, outgoing_pos = pd_edge_positions(pd_code, edge_label)

    # delete s from the code
    # ie. decrement all higher edges
    # and remove references to s
    pd_code[incoming_pos] = EDGE_PLACEHOLDER
    pd_code[outgoing_pos] = EDGE_PLACEHOLDER

    # get the labels of the new edges
    in_label = next_free_edge_label(pd_code)
    loop_label = in_label + 1
    out_label = loop_label + 1
    
    # add the new node group
    # see derivation in masters notes
    match over_under, node_sign:
        case (-1, -1):
            pd_code += [in_label, loop_label, loop_label, out_label]
        case (1, -1):
            pd_code += [loop_label, in_label, out_label, loop_label]
        case (-1, 1):
            pd_code += [in_label, out_label, loop_label, loop_label]
        case (1,1):
            pd_code += [loop_label, loop_label, out_label, in_label]
    
    # replace the placeholders
    # this looks wrong, but it's correct
    # in and out refer to the *new* node
    pd_code[incoming_pos] = out_label
    pd_code[outgoing_pos] = in_label

    return reindex_code(pd_code)

def pd_untwist(pd_code, node_number):
    """
        Untwists a twisted edge, removes a crossing.

        `node_number` should be zero-indexed.

        This follows the conventions in the masters notes,
        and assumes that the twisting was done by pd_twist
        (hence there are implicit conventions about edge labelling).
    """

    # copy the code
    pd_code = list(pd_code)

    # delete the node
    pd_code, node_group = delete_node(pd_code, node_number)

    # find the in and out edge labels
    # asssumes the conventions in pd_twist
    in_label = min(node_group)
    out_label = max(node_group)
    
    new_label = next_free_edge_label(pd_code)

    # connect the new edge
    new_code = []

    for item in pd_code:
        if item == in_label or item == out_label:
            new_code.append(new_label)
        else:
            new_code.append(item)

    return reindex_code(new_code)

# @prep_graph(wants_edges_transposed=False, will_mutate_graph=True)
# def graph_swap_twist(graph):
#     """
#         Swaps a twisted edge.

#         Leaves crossing count unchanged.

#         We need this because you can't go below zero crossings in our formulation.
#     """

#     if len(graph.x) > 1:
#         raise Exception("Can only be used on single node graphs")

#     return graph_mirror_knot(
#         graph,

#         edges_start_transposed=False,
#         edges_should_end_transposed=False,
#         graph_has_been_cloned=True
#     )

# # slides one edge over another
# # adds two crossings
# # this is R2
# def graph_poke(graph, edge_1, edge_2, parity):
#     ...

# # reverse slides on edge over another
# # removes two crossings
# # this is R2^{-1}
# def graph_unpoke(graph, edge_1, edge_2):
#     ...

# # yang-baxters
# # does not change crossings
# # lhs to rhs in the mathworld image
# def graph_yang_baxter(graph, edge_1, edge_2):
#     ...





"""
    The functions below are the 4 natural actions of Z/2Z on a knot diagram.

    They take in a pd code and apply the transform.
"""





def internal_swap_generator(num_nodes, pattern):
    """
        Many pd code operations involve rearranging within a node.

        This tool makes this easier.
    """

    for x in range(num_nodes):
        node = EDGES_PER_NODE*x

        for entry in pattern:
            yield node + entry

def pd_reverse_knot(pd_code):
    """
        Swaps the traversal direction. Sends K -> -K.
    """
    num_nodes = len(pd_code)//4

    # Swaps position 1 with 3 and 2 with 4.
    pattern = (2, 3, 0, 1)

    return [pd_code[pos] for pos in internal_swap_generator(num_nodes, pattern)]

def pd_mirror_knot(pd_code):
    """
        Swap the orientations. Sends K -> K*.
    """

    num_nodes = len(pd_code)//4

    # Swaps position 2 with 4.
    pattern = (0, 3, 2, 1)

    return [pd_code[pos] for pos in internal_swap_generator(num_nodes, pattern)]

def pd_reverse_and_mirror_knot(pd_code):
    """
        Swaps and mirrors. Sends K -> -K*.
    """

    return pd_reverse_knot(pd_mirror_knot(pd_code))
 
def pd_identity(pd_code):
    """
        The identity. Sends K -> K.

        Note: not literally the identity function, as it clones the list.
    """

    return list(pd_code)

# for a given symmetry type, tells you the operations that generate a distinct knot
NEEDED_PD_TRANSFORMS = {
    "Chiral": [pd_identity, pd_reverse_knot, pd_mirror_knot, pd_reverse_and_mirror_knot],
    "Fully amphicheiral": [pd_identity],
    "Negative amphicheiral": [pd_identity, pd_reverse_knot], # note -K = K* for this class
    "Positively amphicheiral": [pd_identity, pd_reverse_knot], # note -K = -K* for this class
    "Reversible": [pd_identity, pd_mirror_knot] # note K* = -K* for this class
}