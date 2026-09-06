from collections import defaultdict as dd
from utilities import *

SHIELDS_MAX_ITERATIONS = 100_000
EDGE_PLACEHOLDER = -float("inf")

def faces_from_pd_code(pd_code):
    """
        Gets the faces in a planar diagram code.

        Assumes the code is a list of integers, like those
        outputted by processing.process_PD. 
    """

    faces = set()

    other_occurance_table = get_pd_other_occurrance_table(pd_code)

    # each edge will be in exactly two faces
    # if we keep track of this, we can save a lot of computation
    times_seen = dd(int)

    # add the left and right face for each edge
    for start_pos in range(len(pd_code)):
        for direction in (LEFT, RIGHT):
            face = []
            cur_pos = start_pos

            quit_early = False
            while True:
                # get current edge label
                cur_edge = pd_code[cur_pos]

                # see if we've seen it before
                if times_seen[cur_edge] >=2:
                    quit_early = True
                    break

                # add the current edge
                face.append(cur_edge)

                # turn in the appropriate direction
                cur_node = cur_pos//EDGES_PER_NODE
                pos_in_node = cur_pos%EDGES_PER_NODE
                next_side = (pos_in_node + direction)%EDGES_PER_NODE + cur_node*EDGES_PER_NODE
                
                # go to the other occurance
                cur_pos = other_occurance_table[next_side]

                if cur_pos == start_pos:
                    break
        
            # add face if appropriate
            if not quit_early:
                faces.add(tuple(sorted(face)))

                # update times seen
                for edge in face:
                    times_seen[edge] += 1
    
    return faces

def get_pd_other_occurrance_table(code):
    """
        Given a PD code, precomputes the "other occurrance" lookup table.

        This saves time in the main Shields algorithm

        Assumes that the code is zero indexed.
    """

    lookup = [None for x in code]
    
    # keeps track of the first time we saw a symbol
    first_time = [None for x in range(max(code)+1)]

    for pos, x in enumerate(code):
        if first_time[x] == None:
            # seen this symbol for the first time,
            # don't know where the other one is yet
            first_time[x] = pos
        else:
            # second time found, update the lookup
            lookup[first_time[x]] = pos
            lookup[pos] = first_time[x]
    
    # sanity check
    if None in lookup:
        raise Exception(f"Malformed PD code detected: {code}")

    return lookup

def calculate_orientations(code, other_occurrance_table=None, return_directions=False):
    """
        Takes in a PD code and calculates the orientation of each node.

        This is called the Shields algorithm in my masters notes.
        The algorithm is explained in more detail there.

        If return_direction is set to true, returns the directions 
        list instead of the orientations (sub)list.
        
        The directions list tells you whether a given position in the code
        represents an incoming or an outgoing edge.
    """

    # initialise the directions array
    directions = [None for x in code]

    n_nodes = len(code)//EDGES_PER_NODE

    for x in range(n_nodes):
        directions[EDGES_PER_NODE*x] = INCOMING
        directions[EDGES_PER_NODE*x+2] = OUTGOING
    
    # calculate the other occurance table if it was not provided
    if other_occurrance_table == None:
        other_occurrance_table = get_pd_other_occurrance_table(code)

    # calculate the unknown orientations
    iterations = 0
    while None in directions:
        iterations += 1

        for x in range(n_nodes):
            # get indexes of the over symbols
            # using slightly different notation to the notes
            odd_index_1 = EDGES_PER_NODE*x+1
            odd_index_2 = EDGES_PER_NODE*x+3

            # gets the (possibly) known direction of the other end of the edge
            odd_1_other = directions[other_occurrance_table[odd_index_1]]
            odd_2_other = directions[other_occurrance_table[odd_index_2]]

            # update the directions if possible
            if odd_1_other is not None:
                directions[odd_index_1] = -odd_1_other

            if odd_2_other is not None:
                directions[odd_index_2] = -odd_2_other

            if directions[odd_index_2] is not None:
                directions[odd_index_1] = -directions[odd_index_2]

            if directions[odd_index_1] is not None:
                directions[odd_index_2] = -directions[odd_index_1]

        if iterations > SHIELDS_MAX_ITERATIONS:
            raise Exception(f"Exceeded max iterations.\ndirections was {directions}.\ncode was {code}.")

    # check if we want to return the raw directions instead
    if return_directions:
        return directions
    
    # extract the orientations
    orientations = [directions[EDGES_PER_NODE*x+1] for x in range(n_nodes)]
    return orientations

def pd_edge_positions(pd_code, edge_label):
    """
        Takes in a pd code and an edge_label.

        Returns (o, i), where 
        `o` is the index where the edge is outgoing and 
        `i` is the index in the code where the edge is incoming.

        Alternately, this is (start, end).
    """

    # get the directions of each position in the code
    directions = calculate_orientations(pd_code, return_directions=True)

    # get the ones corresponding to the desired edge
    # maybe there's a more pythonic solution here? idk
    for index, label in enumerate(pd_code):
        if label == edge_label:
            if directions[index] == INCOMING:
                incoming_pos = index
            else:
                outgoing_pos = index
    
    return outgoing_pos, incoming_pos

def next_free_edge_label(pd_code, amount=1):
    """
        Takes in a pd_code and gives the next free edge label.

        This is just a more readable alias for max + 1.

        If amount = n > 1, gives the list of next n available.
    """

    next_avail = max(pd_code) + 1

    if amount == 1:
        return next_avail
    
    return [next_avail + x for x in range(amount)]

def reindex_code(pd_code):
    """
        Takes in a pd_code and a deleted label.

        Reindexes the edge labels to be zero-indexed and consecutive.
    """

    # list --> set --> list removes duplicates
    current_labels = sorted(list(set(pd_code)))

    return [current_labels.index(x) for x in pd_code]

def delete_node(pd_code, node_number):
    """
        Deletes a node group from the code and reindexes the edges.

        Returns the new pd code and the deleted node group.
    """

    # get the node group to be deleted
    node_group_index = EDGES_PER_NODE * node_number
    node_group = pd_code[node_group_index:node_group_index+EDGES_PER_NODE]

    # remove the node from the code
    pd_code = pd_code[:node_group_index] + pd_code[node_group_index+EDGES_PER_NODE:]

    return pd_code, node_group

def to_canonical_form(pd_code):
    """
        Relabels a pd code to be "first come
        first served", ie. edges are labeled in
        the order the occur in the code.
    """

    # get the occurance order
    occurance_order = []

    for label in pd_code:
        if label not in occurance_order:
            occurance_order.append(label)

    # reorder the code
    return [occurance_order.index(x) for x in pd_code]

def get_ordered_face(pd_code, start_pos, direction, force_edge_relative=False):
    """
        Takes a pd code, an starting position, and a face direction (LEFT/RIGHT).

        Returns the ordered face + the relative direction of
        each edge to the first edge as a dictionary.

        By default this function assumes that "left" means the edge to
        the left if the specified starting node is above the starting edge.
        If `force_edge_relative` is set to `True`, then the convention is
        to have the edge pointing upwards (in the traversal direction sense).
    """

    # get some helper lists
    other_occurance_table = get_pd_other_occurrance_table(pd_code)
    directions = calculate_orientations(
        pd_code, 
        return_directions=True,
        other_occurrance_table=other_occurance_table
    )

    # set up starting config
    face = dict()
    cur_pos = start_pos
    start_edge_direction = directions[start_pos]

    if start_edge_direction == REVERSED and force_edge_relative:
        # need to directions
        return get_ordered_face(
            pd_code, 
            other_occurance_table[start_pos],
            direction,
            force_edge_relative
        )

    relative_traversal_direction = STANDARD

    while True:
        # get current edge label
        cur_edge = pd_code[cur_pos]

        # add the current edge + the relative direction
        face[cur_edge] = relative_traversal_direction

        # turn in the appropriate direction
        cur_node = cur_pos//EDGES_PER_NODE
        pos_in_node = cur_pos%EDGES_PER_NODE
        next_side = (pos_in_node + direction)%EDGES_PER_NODE + cur_node*EDGES_PER_NODE
        
        # go to the other occurance
        cur_pos = other_occurance_table[next_side]

        # get the new traversal direction.

        # if the next edge is leaving the node, then we're
        # following it in the normal direction. otherwise,
        # we're following it in the opposite direction.
        
        # then, we multiply by start_edge_direction to get relative direction
        direction_at_node = directions[next_side] # inc/out
        relative_traversal_direction = start_edge_direction*direction_at_node

        if cur_pos == start_pos:
            break

    return face

def crossing_type_from_index(index):
    """
        Returns if the index represents an under or
        an over crossing in a pd code.

        Note: this doesn't depend on the code, only on
        the value of the index mod 2.
    """

    match index % 2:
        case 0:
            return UNDERCROSSING
        case 1:
            return OVERCROSSING
    
    raise Exception("This state should be unreachable")
        
def opposite_index(index):
    """
        Takes in an index, returns the index that repesents
        the opposite side of the crossing. This will be the index
        of the pre/post edge.
    """

    node_number = index//EDGES_PER_NODE
    internal_index = index%EDGES_PER_NODE

    opposite_internal = (internal_index + EDGES_PER_NODE//2)%EDGES_PER_NODE

    return node_number*EDGES_PER_NODE + opposite_internal

def yb_construct_crossing(intersection, strings):
    """
        Takes in an intersection, performs a Yang-Baxter move,
        then gives the new crossing.

        Returns `(crossing, start_index)`.
    """

    # unpack values
    intersection_order, node_sign, node_number = intersection
    string1, string2 = strings

    # perform the yang-baxter (reverse the intersection order)
    intersection_order = intersection_order[::-1]

    # get the relevant edges
    # the first edge is always the same
    edge1 = string1[intersection_order[0]]
    
    # similarly, the 3rd edge is always the same
    edge3 = string1[intersection_order[0]+1]

    # the other two depend on the sign
    match node_sign:
        case 1:
            # string 2 is running left to right
            edge2 = string2[intersection_order[1]+1]
            edge4 = string2[intersection_order[1]]
        case -1:
            # string 2 is running right to left
            edge2 = string2[intersection_order[1]]
            edge4 = string2[intersection_order[1]+1]
    
    crossing = (edge1, edge2, edge3, edge4)
    start_index = node_number*EDGES_PER_NODE

    return crossing, start_index