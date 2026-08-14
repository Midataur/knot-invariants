from utilities import *
from pd_functions import *
import utilities

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

    return pd_code

def pd_untwist(pd_code, node_number):
    """
        Untwists a twisted edge, removes a crossing.

        `node_number` should be zero-indexed.

        This follows the conventions in the masters notes.
    """

    # copy the code
    pd_code = list(pd_code)

    # delete the node
    pd_code, node_group = delete_node(pd_code, node_number)

    # find the in and out edge labels by checking four cases
    # specifically, check where the loop edge is in the code
    # format: internal index of  (i, l, o)
    # index of l is chosen such that (l+1)%4 is the other l
    OPTIONS = [
        (0, 1, 3), # case 1 in pd_twist
        (1, 3, 2), # case 2 in pd_twist
        (0, 2, 1), # case 3 in pd_twist
        (3, 0, 2), # case 4 in pd_twist
    ]

    # check against the cases
    for i, l, o in OPTIONS:
        potential_other_l = (l+1)%EDGES_PER_NODE
        
        if node_group[l] == node_group[potential_other_l]:
            in_label, out_label = node_group[i], node_group[o]
            break
    
    new_label = next_free_edge_label(pd_code)

    # connect the new edge
    new_code = []

    for item in pd_code:
        if item == in_label or item == out_label:
            new_code.append(new_label)
        else:
            new_code.append(item)

    return new_code

def pd_swap_twist(pd_code):
    """
        Swaps a twisted edge.

        Leaves crossing count unchanged.

        We need this because you can't go below zero crossings in our formulation.
    """

    if len(pd_code) > EDGES_PER_NODE:
        raise Exception("Can only be used on single node codes.")

    return pd_mirror_knot(pd_code)

def pd_poke(pd_code, edge_1_pos, edge_2_pos, parity):
    """
        Slides one edge over another. Adds two crossings.

        This follows the conventions in the masters notes.

        `edge_1_pos` and `edge_2_pos` are positions in the pd_code.

        `parity` is +1 if the left edge goes over, -1 otherwise.
        "The left edge" generally refers to edge 1 if edges are similarly
        oriented and the genuine left edge if they're oppositely oriented.
    """
    
    # get edge labels
    edge_1_label = pd_code[edge_1_pos]
    edge_2_label = pd_code[edge_2_pos]

    # get the relative orientation
    # check both sides
    for side in (LEFT, RIGHT):
        face = get_ordered_face(pd_code, edge_1_pos, side, force_edge_relative=True)

        if edge_2_label in face.keys():
            relative_orientation = face[edge_2_label]
            detected_side = side
            break

    # handle cases in the order of the notes
    # make sure edge 1 is on the left
    tau = None
    match relative_orientation, detected_side:
        case utilities.REVERSED, utilities.LEFT:
            # whoops, the edges should be the other way around
            return pd_poke(pd_code, edge_2_pos, edge_1_pos, parity)
        case utilities.STANDARD, utilities.RIGHT:
            tau =  1
        case utilities.STANDARD, utilities.LEFT:
            tau = -1

    # get new edge labels
    # see picture in masters notes
    i1, i2, p1, p2, o1, o2 = next_free_edge_label(
        pd_code, amount=6
    )

    # get the two new nodes

    # there's a bunch of possible combinations.
    # they don't seem to simplify too nicely.
    # hence, big piecewise function (see notes).
    # there's no nicer way to do this than drawing the picture 
    # (at least, that i know of).

    # here we assume d = +1, we fix it later
    match relative_orientation, tau:
        case  1,  _: # case 1
            node1 = [p2, p1, o2, o1]
            node2 = [i2, p1, p2, i1]
        case -1,  1: # case 3
            node1 = [i2, o1, p2, p1]
            node2 = [p2, i1, o2, p1]
        case -1, -1: # case 5
            node1 = [p2, p1, o2, i1]
            node2 = [i2, p1, p2, o1]

    # (maybe) swap the crossings
    # derivation in the masters notes
    if parity == -1:
        node1 = cyclic_shift(node1, -relative_orientation*parity)
        node2 = cyclic_shift(node2,  relative_orientation*parity)

    # get the connection points
    edge_1_in, edge_1_out = pd_edge_positions(pd_code, edge_1_label)
    edge_2_in, edge_2_out = pd_edge_positions(pd_code, edge_2_label)

    # add the new nodes and stitch them in
    new_code = list(pd_code) + node1 + node2
    new_code[edge_1_in] = o1
    new_code[edge_1_out] = i1
    new_code[edge_2_in] = o2
    new_code[edge_2_out] = i2

    return new_code

# # reverse slides one edge over another
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