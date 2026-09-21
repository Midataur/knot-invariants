from pd_utils import *
from pd_transformations import *
from utilities import *
from typing import NamedTuple
from collections.abc import Callable
import random
import functools

### UTILITIES ###

class MoveData(NamedTuple):
    """
        Stores the data of a legal Reidemeister move.

        ``transformation_func`` is a partial function that takes in only ``pd_code``,
        because the other arguments have been predetermined. The second value in the tuple is
        the number of crossings there will be after applying the move.
    """

    transformation_func: Callable
    num_crossings_after_application: int
    move_type: str

def get_valid_edge_moves(pd_code: list[int], edge_pos: int) -> list[MoveData]:
    """
        Takes in a pd_code and an edge label and returns all Reidemeister moves
        that can be performed involving that edge.

        Returns a list of moves. Specifically, the returned list is a list of move_data types.
    """

    moves = []
    edge_label = pd_code[edge_pos]

    num_nodes = len(pd_code)//EDGES_PER_NODE

    # get the faces
    faces = [get_ordered_face(pd_code, edge_pos, direction) for direction in (LEFT, RIGHT)]

    # add the twist moves
    for option1 in (UNDERCROSSING, OVERCROSSING):
        for option2 in (-1, 1):
            # create the partial function
            partial_move = functools.partial(
                twist, 
                edge_label=edge_label, 
                over_under=option1,
                node_sign=option2
            )

            # add the move
            moves.append(
                MoveData(
                    transformation_func=partial_move, 
                    num_crossings_after_application=num_nodes+1,
                    move_type="twist"
                ) 
            )

    # add the untwist moves
    for face in faces:
        if len(face) == 1: # check that it's actually untwistable
            node_number = edge_pos//EDGES_PER_NODE

            partial_move = functools.partial(
                untwist,
                node_number=node_number
            )

            moves.append(
                MoveData(
                    transformation_func=partial_move, 
                    num_crossings_after_application=num_nodes-1,
                    move_type="untwist"
                )
            )
    
    # add the poke moves
    for face in faces:
        for second_edge_label in face.keys():
            second_edge_pos = pd_code.index(second_edge_label)

            for parity in (-1, 1):
                partial_move = functools.partial(
                    poke,
                    edge_1_pos=edge_pos,
                    edge_2_pos=second_edge_pos,
                    parity=parity
                )

                moves.append(
                    MoveData(
                        transformation_func=partial_move,
                        num_crossings_after_application=num_nodes+2,
                        move_type="poke"
                    )
                )

    # add the unpoke moves
    source, target = get_edge_positions_in_code(pd_code, edge_label)

    start_node = source//EDGES_PER_NODE
    end_node = target//EDGES_PER_NODE

    if pd_can_unpoke(pd_code, start_node, end_node):
        partial_move = functools.partial(
            unpoke,
            node_1_number=start_node,
            node_2_number=end_node
        )

        moves.append(
            MoveData(
                transformation_func=partial_move,
                num_crossings_after_application=num_nodes-2,
                move_type="unpoke"
            )
        )

    # add the yang baxter moves
    for face in faces:
        potential_triangle = list(face.keys())

        # returns a non-None value if the triangle is valid
        if yb_information(pd_code, potential_triangle) is not None:
            partial_move = functools.partial(
                yang_baxter,
                triangle=potential_triangle,
            )

            moves.append(
                MoveData(
                    transformation_func=partial_move,
                    num_crossings_after_application=num_nodes,
                    move_type="yang-baxter"
                )
            )

    # make sure we don't go below zero crossings
    moves = list(filter(
        lambda x: x.num_crossings_after_application > 0,
        moves
    ))

    return moves

### HAMILTONIANS AND TEMPERATURE CURVES ###

# a hamiltonian takes in a pd_code and a move (of type MoveData) and returns a number
# a temp curve takes in a current_step value and some parameters and returns a positive number

def crossing_hamiltonian(pd_code: list[int]):
    """
        Returns the number of crossings in a code.
    """

    return len(pd_code)//EDGES_PER_NODE

def simple_linear_curve(current_step: int, max_temp: float, max_val: int):
    return (max_val - current_step)*max_temp

def constant_curve(current_step: int, temperature: int):
    return temperature

### ACTUAL PROCESSES ###

def apply_random_symmetry(pd_code: list[int], symmetry_type: str):
    """Takes a pd code and a symmetry group and randomly applies something in that group."""

    random_symmetry = random.choice(SYMMETRY_GROUP[symmetry_type])

    return random_symmetry(pd_code)

def hamiltonian_mixer(
        pd_code: list[int], 
        n_steps: int, 
        hamiltonian: Callable[[MoveData], float] = crossing_hamiltonian, 
        temperature_curve: Callable[[int], float] | None = None,
        return_full_history: bool = False
    ):
    """
        Takes a planar diagram code and applies `n_steps` random Reidemeister moves to it.

        The current approach uses a simple rejection-sampling approach that
        assumes the diagrams are sampled from a Boltzmann distribution. See the masters
        notes for more details.
    """

    steps_left = n_steps
    current_code = pd_code

    if return_full_history:
        history = []

    # set temperature curve to default
    if temperature_curve is None:
        temperature_curve = functools.partial(constant_curve, temperature=1)

    while steps_left > 0:
        # pick a random edge
        edge_pos = random.randrange(len(pd_code))

        # find the valid moves we can do involving this edge
        valid_moves = get_valid_edge_moves(pd_code, edge_pos)

    return current_code