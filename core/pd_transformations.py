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

PLACEHOLDER = -float("inf")

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
    pd_code[incoming_pos] = PLACEHOLDER
    pd_code[outgoing_pos] = PLACEHOLDER
    pd_code = [x-1 if x is not PLACEHOLDER and x>edge_label else x for x in pd_code]

    # get the labels of the new edges
    in_label = max(pd_code) + 1
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

    return pd_code

# @prep_graph(wants_edges_transposed=True, will_mutate_graph=True)
# def graph_untwist(graph, node_index):
#     """
#         Untwists a twisted edge, removes a crossing.
#     """

#     if graph.x.shape[0] <= node_index:
#         raise Exception("Node index out of bounds.")
    
#     # find the attached edges
#     loop = None
#     incoming = None
#     outgoing = None

#     for pos, edge in enumerate(graph.edge_index):
#         # check if the edge relates to the node
#         if node_index not in edge:
#             continue
        
#         # figure out what kind of edge this is
#         source, target = edge

#         if source != node_index:
#             # it's the incoming edge
#             incoming = pos
#             prenode = source
#             precolor = inverse_color_function(
#                 graph.edge_attr[pos]
#             )[0]
#         elif target != node_index:
#             # it's the outgoing edge
#             outgoing = pos
#             postnode = target
#             postcolor = inverse_color_function(
#                 graph.edge_attr[pos]
#             )[1]
#         else:
#             # it's the loop
#             loop = pos

#     # check that this is actually untwistable
#     if loop is None or incoming is None or outgoing is None:
#         raise Exception(
#             f"Node is not untwistable: loop is {loop}, incoming is {incoming}, and outgoing is {outgoing}."
#         )

#     graph_prep_state = GraphPrepState(
#         edges_start_transposed=True,
#         edges_should_end_transposed=True,
#         graph_has_been_cloned=True
#     )
    
#     # delete the edges and the node
#     batch_delete(
#         graph, 
#         node_indices=[node_index], 
#         edge_indices=[loop, incoming, outgoing],

#         **graph_prep_state._asdict()
#     )

#     # add the new edge
#     add_edges(
#         graph, 
#         new_edges=[(prenode, postnode)],
#         new_colors=[color_function(precolor, postcolor)],

#         **graph_prep_state._asdict()
#     )

#     return graph

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