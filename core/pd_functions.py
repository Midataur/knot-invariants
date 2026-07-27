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
        raise Exception("Malformed PD code detected")

    return lookup

def calculate_orientations(code, other_occurrance_table=None, return_directions=False):
    """
        Takes in a PD code and calculates the orientation of each node.

        This is called the Shields algorithm in my masters notes.
        The algorithm is explained in more detail there.

        If return_raw is set to true, returns the directions 
        list instead of the orientations (sub)list.
        TODO: split this functionality into its own function.
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

        Returns (i, o), where:
            
        i is the index in the code where the edge is incoming.
        
        o is the index where the edge is outgoing.
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
    
    return incoming_pos, outgoing_pos

def next_free_edge_label(pd_code):
    """
        Takes in a pd_code and gives the next free edge label.

        This is just a more readable alias for max + 1.
    """

    return max(pd_code) + 1

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