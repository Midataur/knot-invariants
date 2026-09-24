from pd_utils import *
from pd_transformations import *
from utilities import *
from typing import NamedTuple
from collections.abc import Callable
import random
import functools
import math

INFINITY = float("inf")

### UTILITIES ###

class MoveData(NamedTuple):
    """
        Stores the data of a legal Reidemeister move.

        ``transformation_func`` is a partial function that takes in only ``pd_code``,
        because the other arguments have been predetermined. The second value in the tuple is
        the number of crossings there will be after applying the move.
    """

    transformation_func: Callable[[list[int]], list[int]]
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

            # disallow poking edges with themselves, since this breaks things.
            # these moves are just two twists anyway.
            if edge_label != second_edge_label:
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

        # check that this is genuinely a triangle,
        # ie. has three sides and three corners

        # check sides
        if len(potential_triangle) != 3:
            continue
        
        # check corners
        corners = set()

        for label in potential_triangle:
            positions = get_edge_positions_in_code(pd_code, label)

            for pos in positions:
                corners.add(
                    tuple(get_node(pd_code, pos//EDGES_PER_NODE))
                )
        
        if len(corners) != 3:
            continue

        # alright let's try
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

def crossing_hamiltonian(pd_code: list[int], move: MoveData | None = None):
    """
        Returns the number of crossings in a code.
        If a move is given, returns `num_crossings_after_application`.
        Otherwise, returns the number of crossings in the node.
    """

    if move is not None:
        return move.num_crossings_after_application

    return len(pd_code)//EDGES_PER_NODE

def crossing_hamiltonian_with_cutoff(pd_code: list[int], max_crossings: int = INFINITY, move: MoveData | None = None):
    """
        Like `crossing_hamiltonian` but doesn't allow going above `max_crossings`.
    """

    if move is not None:
        crossings = move.num_crossings_after_application
    else:
        crossings = len(pd_code)//EDGES_PER_NODE
    
    return crossings if crossings <= max_crossings else INFINITY

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

        hamiltonian: Callable[[list[int], MoveData | None], float] = crossing_hamiltonian, 
        temperature_curve: Callable[[int], float] | None = None,

        return_full_history: bool = False,
        strict = False
    ):
    """
        Takes a planar diagram code and applies `n_steps` random Reidemeister moves to it.

        The current approach uses a simple rejection-sampling approach that
        assumes the diagrams are sampled from a Boltzmann distribution. See the masters
        notes for more details.

        If `strict` is set to `True` then the code will be checked at each step to make sure
        it's still a valid pd code. This is expensive, so it is recommended to only enable this
        for debugging purposes.
    """

    steps_left = n_steps
    current_code = pd_code

    history = [current_code]

    # set temperature curve to default
    if temperature_curve is None:
        temperature_curve = functools.partial(constant_curve, temperature=1)

    while steps_left > 0:
        # pick a random edge
        selected_edge_pos = random.randrange(len(current_code))

        # find the valid moves we can do involving this edge
        valid_moves = get_valid_edge_moves(current_code, selected_edge_pos)

        # compute current state
        current_energy = hamiltonian(current_code, move=None)
        current_step = n_steps-steps_left
        temperature = temperature_curve(current_step)

        # pick a random valid move
        selected_move = random.choice(valid_moves)

        # compute probability of acceptance
        selected_energy = hamiltonian(current_code, move=selected_move)

        ratio = math.exp(-(selected_energy-current_energy)/temperature)
        acceptance_prob = min(ratio, 1)

        # do move with calculated probability
        if random.random() <= acceptance_prob:
            # print("before moving", current_code)

            if strict:
                old_code = list(current_code)

            current_code = selected_move.transformation_func(current_code)
            steps_left -= 1

            # print("moves left", steps_left)
            # print("acceptance prob", acceptance_prob)
            # print(selected_move)
            # print()

            if return_full_history:
                history.append(current_code)
        
            # sanity check, if requested
            if strict and not pd_code_is_valid(current_code, verbose=True):
                print("Code before moving:", old_code)
                print("Selected move:", selected_move)
                raise Exception(f"Error found in PD code: {current_code}")

    if return_full_history:
        return current_code, history
    
    return current_code