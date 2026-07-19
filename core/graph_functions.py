from typing import NamedTuple
from functools import wraps
import torch

# some standard conventions
INCOMING = -1
OUTGOING = 1

STANDARD = -1
REVERSED = 1

UNDERCROSSING = -1
OVERCROSSING = 1

LEFT = -1
RIGHT = 1   

VALENCY = 4

def color_function(start: int, end: int):
    """
        Edge coloring piecewise function.

        Swapping both crossing types is the same as
        multiplying by -1. 

        See master's notes: The Garbali-Gauss construction.
    """
    start_is_positive = start > 0
    end_is_positive = end > 0

    match (start_is_positive, end_is_positive):
        case (False, False):
            return -2
        case (False, True):
            return -1
        case (True, False):
            return 1
        case (True, True):
            return 2
    
    raise Exception(f"Invalid edge type ({start},{end}).")

def inverse_color_function(color: int):
    "The inverse of color function"

    match color:
        case -2:
            return (-1, -1)
        case -1:
            return (-1, 1)
        case 1:
            return (1, -1)
        case 2:
            return (1, 1)

    raise Exception("Invalid color given")

# takes the color (a,b) and gives you (b,a)
def reverse_edge_color(color: int):
    if abs(color) == 1:
        return -color
    
    return color

class GraphPrepState(NamedTuple):
    """
        Stores information relevant to the prep_graph decorator.

        A dict works fine, this is just for convenience.
    """

    edges_start_transposed: bool
    edges_should_end_transposed: bool
    graph_has_been_cloned: bool

DEFAULT_STATE = GraphPrepState(
    edges_start_transposed=False,
    edges_should_end_transposed=False,
    graph_has_been_cloned=False
)

def prep_graph(
        *, # forces kwargs for these
        wants_edges_transposed: bool,
        will_mutate_graph: bool
    ):
    """
        Many functions take in a graph. 
        This wrapper makes sure the graph is in a desireable 
        format without repeating too much work.

        More specifically: the wrapper will clone the graph if needed,
        and transpose the edge tensor if needed.
    """

    def wrapper(func):

        @wraps(func)
        def wrapped_function(
            graph, 

            *args, 

            edges_start_transposed: bool,
            edges_should_end_transposed: bool,
            graph_has_been_cloned: bool,

            **kwargs
        ):
            need_to_update_face_cache = False

            # do we need to clone the graph?
            if will_mutate_graph and not graph_has_been_cloned:
                need_to_update_face_cache = True
                graph = graph.clone()
            
            # do we need to transpose the edges?
            edges_are_transposed = edges_start_transposed

            if edges_are_transposed != wants_edges_transposed:
                graph.edge_index = graph.edge_index.t()
                edges_are_transposed = wants_edges_transposed
            
            result = func(graph, *args, **kwargs)

            # do we need to update the face cache?
            if need_to_update_face_cache and func is not update_face_cache:
                graph = update_face_cache(
                    graph,
                    edges_start_transposed=edges_are_transposed,
                    edges_should_end_transposed=edges_are_transposed,
                    graph_has_been_cloned=True
                )

            # do we need to untranspose the edges?
            if edges_are_transposed != edges_should_end_transposed:
                graph.edge_index = graph.edge_index.t()

            return result

        return wrapped_function
    
    return wrapper

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def delete_edge(graph, edge_index):
    """
        Deletes an edge in a graph in place.

        Does not delete any attached nodes.
    """
    # delete the edge
    graph.edge_index = torch.concat([
        graph.edge_index[:edge_index],
        graph.edge_index[edge_index+1:]
    ])

    # delete the edge color
    graph.edge_attr = torch.concat([
        graph.edge_attr[:edge_index],
        graph.edge_attr[edge_index+1:]
    ])

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def delete_node(graph, node_index):
    """
        Deletes a node of a graph in place.

        Does not delete any attached edges, but does relabel them if needed.

        Note: this function has wants_edges_transposed=True 
        because of the contexts this is usually used in. The function
        itself doesn't care either way.
    """
    # delete the node
    graph.x = torch.concat([
        graph.x[:node_index],
        graph.x[node_index+1:]
    ])

    # we've shifted a bunch of node indices
    # so now we need to relabel the edges that touch those nodes
    mask = (graph.edge_index > node_index).to(int)
    graph.edge_index -= mask

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def batch_delete(graph, node_indices=[], edge_indices=[]):
    """
        Deletes several elements at once in place.

        Does this in reverse order to make sure the indices remain valid.
    """
    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=True
    )

    # delete the nodes
    for node_index in sorted(node_indices, reverse=True):
        delete_node(
            graph, node_index, **graph_prep_state._asdict()
        )
    
    # delete the edges
    for edge_index in sorted(edge_indices, reverse=True):
        delete_edge(
            graph, edge_index, **graph_prep_state._asdict()
        )

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def add_edges(graph, new_edges, new_colors):
    """
        Adds edges and their colors in place.
    """
    if len(new_edges) != len(new_colors):
        raise Exception(
            f"There were {len(new_edges)} edges provided and {len(new_colors)} colors; these must match."
        )
    
    # convert if we need to
    if type(new_edges) is not torch.Tensor:
        new_edges = torch.tensor(new_edges, dtype=int)
    
    if type(new_colors) is not torch.Tensor:
        new_colors = torch.tensor(new_colors, dtype=float)

    # add the edges
    graph.edge_index = torch.concat([
        graph.edge_index,
        new_edges
    ])

    # add the new colors
    graph.edge_attr = torch.concat([
        graph.edge_attr,
        new_colors.reshape((-1,1))
    ])

@prep_graph(will_mutate_graph=True, wants_edges_transposed=False)
def add_node(graph, sign):
    """
        Adds node in place.

        Returns the new node index.
    """
    # add the new node
    graph.x = torch.concat([
        graph.x,
        torch.tensor([[sign]])
    ])

    return graph.x.shape[0]-1

def get_left(crossing_type: int, direction: int, node_sign: int):
    """
        Tells you if the incoming or outgoing edge is "left".

        -1 = incoming.

        +1 = outgoing.

        See april 19th section of masters notes. 
        This is equation 22 (at time of writing).
    """
    
    return -crossing_type*direction*node_sign

@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_next_node_index(graph, edge_index: int, direction: int):
    """
        Takes in an index of a directed edge.

        Returns the index of either the source node or the target node; 
        source if direction = +1, target if direction = -1.
    """

    # get the edge
    source, target = graph.edge_index[edge_index]

    if direction == STANDARD:
        return target
    else:
        return source

@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_face_next_edge(
    graph, 
    face_side: int,
    crossing_type: int, 
    incoming_outgoing: int, 
    pivot_node_index: int
):
    """
        Returns the next edge when traversing a face.
        Also returns the traversal direction of that edge.

        face_side is -1 for left, +1 for right.
    """

    next_node_sign = graph.x[pivot_node_index]

    # dd is -1 if the incoming edge is the desired edge
    # and +1 if the outgoing is
    left = get_left(crossing_type, incoming_outgoing, next_node_sign)
    desired_direction = -face_side*left

    graph_prep_state = GraphPrepState(
        graph_has_been_cloned=False,
        edges_start_transposed=True,
        edges_should_end_transposed=True
    )
    
    # get edges connected to node
    candidates = get_adjacent_edges(
        graph, 
        pivot_node_index,

        **graph_prep_state._asdict()
    )

    # find the correct next edge
    for edge_index in candidates:
        # get edge
        source, target = graph.edge_index[edge_index]

        # get edge crossing type
        source_type, target_type = inverse_color_function(
            graph.edge_attr[edge_index]
        )

        # looking for: 
        # edge type at node is opposite of crossing type
        # AND 
        # direction is wanted_direction
        if (
            desired_direction == INCOMING and 
            target_type == -crossing_type and 
            target == pivot_node_index
        ):
            traversal_direction = REVERSED
            return edge_index, traversal_direction
        
        elif (
            desired_direction == OUTGOING and 
            source_type == -crossing_type and
            source == pivot_node_index
        ):
            traversal_direction = STANDARD
            return edge_index, traversal_direction
    
    raise Exception("No next edge found, something has gone wrong.")
    
@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_adjacent_edges(graph, node_id):
    """
        Finds all edges that attach to a node.

        Returns a list of edge ids.
    """

    adjacent_edges = []

    for edge_index, edge in enumerate(graph.edge_index):
        # check if either end is the correct node
        if node_id in edge:
            adjacent_edges.append(edge_index)
    
    return adjacent_edges

@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_face(graph, start_edge, face_side):
    """
        Gets the face an edge belongs to.
        start_edge = index of the edge to start on.
        Set face_side = +1 for left face, -1 for right.
    """
    # track the edge we're looking at
    cur_edge_index = start_edge

    # keeps track of which way we're looking at the current edge
    # we always start by considering it to be incoming
    traversal_direction = STANDARD

    # keeps track of the edges in the current face
    face = []

    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=False
    )

    while True:
        # add current edge to face
        face.append(cur_edge_index)

        # get next node data
        next_node_index = get_next_node_index(
            graph=graph,
            edge_index=cur_edge_index,
            direction=traversal_direction,
            
            **graph_prep_state._asdict()
        )

        # get edge crossing type
        source_type, target_type = inverse_color_function(
            graph.edge_attr[cur_edge_index]
        )

        edge_crossing_type = target_type if traversal_direction == STANDARD else source_type

        # incoming or outgoing?
        # see april 19th section of masters notes
        cur_edge_index, traversal_direction = get_face_next_edge(
            graph=graph,
            face_side=face_side,
            crossing_type=edge_crossing_type,
            incoming_outgoing=traversal_direction,
            pivot_node_index=next_node_index,

            **graph_prep_state._asdict()
        )

        # halt condition
        if cur_edge_index in face:
            break

    return tuple(sorted(face))

@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_faces(graph):
    """Finds all the faces in a graph"""

    faces = set()

    num_edges = graph.edge_index.shape[0]

    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=False
    )

    # each edge will be in exactly two faces
    # if we keep track of this, we can save a lot of computation
    times_seen = [0 for edge in graph.edge_index]

    for edge_index in range(num_edges):
        # check if we can save some time
        if times_seen[edge_index] >= 2:
            continue

        for side in [LEFT, RIGHT]:
            new_face = get_face(
                graph=graph, 
                start_edge=edge_index, 
                face_side=side, 
                **graph_prep_state._asdict()
            )

            # get left face
            faces.add(new_face)

            for edge in new_face:
                times_seen[edge] += 1
    
    return faces

@prep_graph(will_mutate_graph=True, wants_edges_transposed=True)
def update_face_cache(graph):
    """
        Recalculates the face cache.

        TODO: If this is too slow, do a smarter update.
    """
    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=True
    )

    graph.faces = get_faces(graph, **graph_prep_state._asdict())

    return graph

@prep_graph(will_mutate_graph=False, wants_edges_transposed=True)
def get_pd_code_from_graph(graph):
    """
        Takes in a graph, gives the planar diagram code.
    """

    graph_prep_state = GraphPrepState(
        edges_start_transposed=True,
        edges_should_end_transposed=True,
        graph_has_been_cloned=False
    )

    pd_code = ""

    # build up the code one node at a time
    for node_id in range(graph.x.shape[0]):
        # get the connecting edges
        adjacent_edges = get_adjacent_edges(graph, node_id, **graph_prep_state._asdict())

        starting_edge = None

        # find the incoming under
        for edge_id in adjacent_edges:
            target = graph.edge_index[edge_id][1]

            # get edge crossing type
            target_type = inverse_color_function(graph.edge_attr[edge_id])[1]

            # looking for: 
            # edge type at node is UNDERCROSSING
            # AND 
            # target is the node
            if target_type == UNDERCROSSING and target == node_id:
                starting_edge = edge_id
                break

        assert starting_edge is not None
        
        # get counter-clockwise order of the edges
        # inefficient, but it works
        order = [starting_edge]
        cur_edge = starting_edge
        cur_direction = STANDARD
        cur_crossing_type = UNDERCROSSING

        for x in range(VALENCY-1):
            # get next edge id
            cur_edge, cur_direction = get_face_next_edge(
                graph,
                face_side=RIGHT,
                crossing_type=cur_crossing_type,
                incoming_outgoing=cur_direction,
                pivot_node_index=node_id,

                **graph_prep_state._asdict()
            )

            # swap the direction (we're staying at the same node)
            cur_direction *= -1

            # add next edge to the order
            order.append(cur_edge)

            # update crossing type
            # we will always alternate over and under
            cur_crossing_type *= -1
    
        # add to the code
        pd_code += f"X{list(order)};"

    # strip the last ;
    return pd_code[:-1]