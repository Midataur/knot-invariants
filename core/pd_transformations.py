from utilities import *
from pd_utils import *
import utilities
import itertools

"""
    This file contains a lot of the same functions as graph_transformations.py,
    but it does them in terms of planar diagram codes instead.

    The resulting code is dramatically faster.
"""


"""
    The functions below are the Reidemeister moves, but for pd codes.
"""

def twist(pd_code: list[int], edge_label: int, over_under: int, node_sign: int):
    """
        Twists an untwisted edge, adds a crossing.

        This follows the conventions in the masters notes.
    """

    # copy the code
    pd_code = list(pd_code)

    # find where the edge is referenced
    outgoing_pos, incoming_pos = get_edge_positions_in_code(pd_code, edge_label)

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

def untwist(pd_code: list[int], node_number: int):
    """
        Untwists a twisted edge, removes a crossing.

        `node_number` should be the zero-indexed number of the node that will get removed.

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

def swap_twist(pd_code: list[int]):
    """
        Swaps a twisted edge.

        Leaves crossing count unchanged.

        We need this because you can't go below zero crossings in our formulation.
    """

    if len(pd_code) > EDGES_PER_NODE:
        raise Exception("Can only be used on single node codes.")

    return mirror_knot(pd_code)

def poke(pd_code: list[int], edge_1_pos: int, edge_2_pos: int, parity: int):
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
    found = False
    for side in (LEFT, RIGHT):
        face = get_ordered_face(pd_code, edge_1_pos, side, force_edge_relative=True)

        if edge_2_label in face.keys():
            relative_orientation = face[edge_2_label]
            detected_side = side
            found = True
            break

    if not found:
        raise Exception("Did not find edge2 in either edge1 face.")

    # handle cases in the order of the notes
    # make sure edge 1 is on the left
    tau = None
    match relative_orientation, detected_side:
        case utilities.REVERSED, utilities.LEFT:
            # whoops, the edges should be the other way around
            return poke(pd_code, edge_2_pos, edge_1_pos, parity)
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
    edge_1_out, edge_1_in = get_edge_positions_in_code(pd_code, edge_1_label)
    edge_2_out, edge_2_in = get_edge_positions_in_code(pd_code, edge_2_label)

    # add the new nodes and stitch them in
    new_code = list(pd_code) + node1 + node2
    new_code[edge_1_in] = o1
    new_code[edge_1_out] = i1
    new_code[edge_2_in] = o2
    new_code[edge_2_out] = i2

    return new_code

def unpoke(pd_code: list[int], node_1_number: int, node_2_number: int):
    """
        Takes two strands that cross without intertwining and seperates them.
        Assumes that there is a genuine poke going on, may break otherwise.

        This follows the conventions in the masters notes.
    """

    # ensure node1 is earlier than node2
    if node_1_number > node_2_number:
        return unpoke(pd_code, node_2_number, node_1_number)

    # get the two node indices
    node_1_index = EDGES_PER_NODE*node_1_number
    node_2_index = EDGES_PER_NODE*node_2_number

    node1 = pd_code[node_1_index:node_1_index+EDGES_PER_NODE]
    node2 = pd_code[node_2_index:node_2_index+EDGES_PER_NODE]

    # get the directions of each edge
    directions = calculate_orientations(pd_code, return_directions=True)

    # figure out what the connecting edges are
    # these are the two that are shared between the nodes
    shared_edges = set(node1).intersection(set(node2))

    # get two new edge labels
    string1_label, string2_label = next_free_edge_label(pd_code, 2)

    # look at the first edge in each node
    # this will be the start of a string or a connecting edge
    # if it's a start, the end of that string will be third in the other node
    if node1[0] not in shared_edges:
        string1 = (node1[0], node2[2])
    else:
        string1 = (node2[0], node1[2])
    
    # get the other string
    all_edges = set(node1+node2)
    seen_already = set(string1).union(shared_edges)
    string2 = tuple(all_edges.difference(seen_already))

    # delete the nodes
    new_code = (
          pd_code[:node_1_index]
        + pd_code[node_1_index+EDGES_PER_NODE:node_2_index]
        + pd_code[node_2_index+EDGES_PER_NODE:]
    )

    # stitch the broken edges
    for pos in range(len(new_code)):
        if new_code[pos] in string1:
            new_code[pos] = string1_label
        elif new_code[pos] in string2:
            new_code[pos] = string2_label
    
    return new_code
    
def yang_baxter(pd_code: list[int], triangle: tuple[int, int, int]):
    """
        The third Reidemeister move.
        Does not change the crossing count.
        This move is its own inverse.

        `triangle` should be a list of three integers,
        the labels of the three edges in the triangle.
    """

    # duplicate list to avoid mutations
    pd_code = list(pd_code)

    ### INFORMATION STAGE ###

    yb_info = yb_information(pd_code, triangle)

    if yb_info is None:
        raise Exception("Invalid YB move.")
    
    intersections, strings = yb_info
    
    ### MODIFICATION STAGE

    for intersection, intersection_strings in zip(
        intersections,
        itertools.combinations(strings, 2)
    ):
        # get the new crossing
        crossing, start_index = yb_construct_crossing(intersection, intersection_strings)
        
        # overwrite the old crossing
        for pos, x in enumerate(crossing):
            pd_code[start_index+pos] = x
        
    # done!
    return pd_code




"""
    The functions below are the 4 natural actions of Z/2Z on a knot diagram.

    They take in a pd code and apply the transform.
"""



def internal_swap_generator(num_nodes: int, pattern: tuple[int, ...]):
    """
        Many pd code operations involve rearranging within a node.

        This tool makes this easier.
    """

    for x in range(num_nodes):
        node = EDGES_PER_NODE*x

        for entry in pattern:
            yield node + entry

def reverse_knot(pd_code: list[int]):
    """
        Swaps the traversal direction. Sends K -> -K.
    """
    num_nodes = len(pd_code)//4

    # Swaps position 1 with 3 and 2 with 4.
    pattern = (2, 3, 0, 1)

    return [pd_code[pos] for pos in internal_swap_generator(num_nodes, pattern)]

def mirror_knot(pd_code: list[int]):
    """
        Swap the orientations. Sends K -> K*.
    """

    num_nodes = len(pd_code)//4

    # Swaps position 2 with 4.
    pattern = (0, 3, 2, 1)

    return [pd_code[pos] for pos in internal_swap_generator(num_nodes, pattern)]

def reverse_and_mirror_knot(pd_code: list[int]):
    """
        Swaps and mirrors. Sends K -> -K*.
    """

    return reverse_knot(mirror_knot(pd_code))
 
def pd_identity(pd_code: list[int]):
    """
        The identity. Sends K -> K.

        Note: not literally the identity function, as it clones the list.
    """

    return list(pd_code)

# for a given symmetry type, tells you the operations that form the symmetry group
SYMMETRY_GROUP = {
    "Chiral": [pd_identity],
    "Fully amphicheiral": [pd_identity, reverse_knot, mirror_knot, reverse_and_mirror_knot],
    "Negative amphicheiral": [pd_identity, reverse_and_mirror_knot], # note -K = K* for this class
    "Positively amphicheiral": [pd_identity, mirror_knot], # note -K = -K* for this class
    "Reversible": [pd_identity, reverse_knot] # note K* = -K* for this class
}

# for a given symmetry type, tells you the operations needed to create all variants
NEEDED_PD_TRANSFORMS = {
    "Chiral": [pd_identity, reverse_knot, mirror_knot, reverse_and_mirror_knot],
    "Fully amphicheiral": [pd_identity],
    "Negative amphicheiral": [pd_identity, reverse_knot], # note -K = K* for this class
    "Positively amphicheiral": [pd_identity, reverse_knot], # note -K = -K* for this class
    "Reversible": [pd_identity, mirror_knot] # note K* = -K* for this class
}