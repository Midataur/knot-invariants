from collections import defaultdict as dd
from utilities import *
import itertools

SHIELDS_MAX_ITERATIONS = 100_000
EDGE_PLACEHOLDER = -float("inf")

def faces_from_pd_code(pd_code: list[int]):
    """
        Gets the faces in a planar diagram code.
    """

    faces = set()

    other_occurance_table = get_other_occurrance_table(pd_code)

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

def get_other_occurrance_table(pd_code: list[int]):
    """
        Given a PD code, precomputes the "other occurrance" lookup table.

        This saves time in the main Shields algorithm

        Assumes that the code is zero indexed.
    """

    lookup = [None for x in pd_code]
    
    # keeps track of the first time we saw a symbol
    first_time = [None for x in range(max(pd_code)+1)]

    for pos, x in enumerate(pd_code):
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
        raise Exception(f"Malformed PD code detected: {pd_code}")

    return lookup

def calculate_orientations(
        pd_code: list[int], 
        other_occurrance_table: list = None, 
        return_directions: bool = False
):
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
    directions = [None for x in pd_code]

    n_nodes = len(pd_code)//EDGES_PER_NODE

    for x in range(n_nodes):
        directions[EDGES_PER_NODE*x] = INCOMING
        directions[EDGES_PER_NODE*x+2] = OUTGOING
    
    # calculate the other occurance table if it was not provided
    if other_occurrance_table == None:
        other_occurrance_table = get_other_occurrance_table(pd_code)

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
            raise Exception(f"Exceeded max iterations.\ndirections was {directions}.\ncode was {pd_code}.")

    # check if we want to return the raw directions instead
    if return_directions:
        return directions
    
    # extract the orientations
    orientations = [directions[EDGES_PER_NODE*x+1] for x in range(n_nodes)]
    return orientations

def get_edge_positions_in_code(pd_code: list[int], edge_label: int):
    """
        Takes in a `pd_code` and an `edge_label`.

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

def next_free_edge_label(pd_code: list[int], amount: int = 1):
    """
        Takes in a pd_code and gives the next free edge label.

        This is just a more readable alias for max + 1.

        If amount = n > 1, gives the list of next n available.
    """

    next_avail = max(pd_code) + 1

    if amount == 1:
        return next_avail
    
    return [next_avail + x for x in range(amount)]

def reindex_code(pd_code: list[int]):
    """
        Takes in a pd_code and a deleted label.

        Reindexes the edge labels to be zero-indexed and consecutive.
    """

    # list --> set --> list removes duplicates
    current_labels = sorted(list(set(pd_code)))

    return [current_labels.index(x) for x in pd_code]

def delete_node(pd_code: list[int], node_number: int):
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

def pd_canonical_form(pd_code: list[int]):
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

def get_ordered_face(
        pd_code: list[int], 
        start_pos: int, 
        direction: int, 
        force_edge_relative: bool = False
):
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
    other_occurance_table = get_other_occurrance_table(pd_code)
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

def crossing_type_from_index(index: int):
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
        
def opposite_index(index: int):
    """
        Takes in an index, returns the index that repesents
        the opposite side of the crossing. This will be the index
        of the pre/post edge.
    """

    node_number = index//EDGES_PER_NODE
    internal_index = index%EDGES_PER_NODE

    opposite_internal = (internal_index + EDGES_PER_NODE//2)%EDGES_PER_NODE

    return node_number*EDGES_PER_NODE + opposite_internal

def yb_construct_crossing(intersection: tuple[int, int, int], strings: tuple[int, int]):
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

def get_node(pd_code: list[int], node_number: int):
    """
        Takes in a `pd_code` and a node_number and returns the node.
    """

    start = node_number*EDGES_PER_NODE
    end = start + EDGES_PER_NODE

    return pd_code[start:end]

def pd_can_unpoke(pd_code: list[int], node_1_num: int, node_2_num: int) -> bool:
    """
        Takes in a `pd_code` and two node numbers and tells you if
        they can be removed with an unpoke move. This code assume that
        theorem 7.8 in my notes is correct.

        TODO: write a better proof for theorem 7.8
    """

    node1 = get_node(pd_code, node_1_num)
    node2 = get_node(pd_code, node_2_num)    

    # condition 1 in theorem 7.8
    shared_edges = set(node1).intersection(set(node2))
    connection_condition = len(shared_edges) == 2

    # condition 2 in theorem 7.8
    adjacency_per_node = []
    for node in (node1, node2):
        positions_in_node = [node.index(x) for x in shared_edges]

        # the length of a node is exactly 4, and adjacency is cyclic
        # hence, they are adjacent if the positions differ by an odd number

        adjacent_in_node = abs(max(positions_in_node)-min(positions_in_node)) % 2 == 1
        adjacency_per_node.append(adjacent_in_node)

    adjacency_condition = all(adjacency_per_node)
    
    # condition 3 in theorem 7.8
    orientations = calculate_orientations(pd_code)
    sign_condition = orientations[node_1_num] != orientations[node_2_num]

    return connection_condition and adjacency_condition and sign_condition

def yb_information(pd_code: list[int], triangle: tuple[int, int, int]):
    """
        Finds all the relevant information for a YB move,
        If the triangle is a valid yb-triangle, returns
        a tuple `(intersections, strings)`. Otherwise, returns None.

        `triangle` should be a list of three integers,
        the labels of the three edges in the triangle.
    """

    # sanity check
    if len(triangle) != 3:
        print("length fail", triangle)
        return None

    # get indices where the edge occurs
    edge_positions = [get_edge_positions_in_code(pd_code, edge) for edge in triangle]

    # figure out height levels
    # one will be strictly under (-2)
    # one will be in the middle (0)
    # and one will be strictly over (2)
    # the specific values don't matter, only the order
    height_levels = [
        crossing_type_from_index(source) + crossing_type_from_index(target)
        for source, target in edge_positions
    ]

    # sanity check
    if sorted(height_levels) != [-2, 0, 2]:
        print("height fail", height_levels)
        return None

    # sort everything by height
    height_levels, triangle, edge_positions = unzip(sorted(zip(
        height_levels, triangle, edge_positions
    )), num_lists_expected=3)

    # everything is now in the order U, M, O
    # (under, middle, over)

    # assemble the full strings
    # ie. the pre-edge, the edge, and the post-edge
    strings = []

    for label, edge in zip(triangle, edge_positions):
        # unpack values
        start, end = edge

        # get front, middle, and back of the string
        middle = label

        front_index = opposite_index(start)
        front = pd_code[front_index]

        back_index = opposite_index(end)
        back = pd_code[back_index]

        strings.append((front, middle, back))

    orientations = calculate_orientations(pd_code)

    # get the intersection data
    # order will be [U ∩ M, U ∩ O, M ∩ O]
    intersections = []
    for edge1, edge2 in itertools.combinations(edge_positions, 2):
        source1, target1 = edge1
        source2, target2 = edge2

        # find what the node number of the intersection is
        # and what the intersection order is

        string1_first   = source1//EDGES_PER_NODE
        string1_second  = target1//EDGES_PER_NODE
        string2_first   = source2//EDGES_PER_NODE
        string2_second  = target2//EDGES_PER_NODE

        # case bashing time!
        # note: we're zero indexing bc it's nicer in code
        if string1_first == string2_first:
            node_number = string1_first
            intersection_order = (0, 0)

        elif string1_first == string2_second:
            node_number = string1_first
            intersection_order = (0, 1)

        elif string1_second == string2_first:
            node_number = string1_second
            intersection_order = (1, 0)

        elif string1_second == string2_second:
            node_number = string1_second
            intersection_order = (1, 1)

        else:
            print(f"""Warning: this state should be unreachable. Values observed: {
                source1, source2, target1, target2
            }""")
        
            return None
    
        node_sign = orientations[node_number]

        intersections.append((intersection_order, node_sign, node_number))
    
    return (intersections, strings)