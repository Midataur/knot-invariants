from utilities import format_for_pytorch_geo, color_function
from collections import defaultdict as dd
from transformations import *
from tqdm import tqdm
import torch_geometric as tg
import torch

GC_IDENTIFIER = "invariant:Gauss_Code"
PD_IDENTIFIER = "invariant:PD_Presentation"
SYM_IDENTIFIER = "invariant:Symmetry_Type"
RDF_BREAKPOINT = "> \""
LINK_IDENTIFIER = "{"

UNKNOT_ID = "0_1"
UNKNOT_CODE = "X<sub>1122</sub>" # listed as having no PD code but we want it to have one
UNKNOT_SYM_TYPE = "Fully amphicheiral" # listed as having no sym-type but it should be fully symmetric

PD_CODE = "PD_code"
SYM_TYPE = "sym_type"
IS_LINK = "is_link"

SHIELDS_MAX_ITERATIONS = 100_000

# if no sim type is provided we assume that the knot is fully symmetric.
# this is the conservative assumption and guarantees that we won't accidentally
# have two equivalent knots listed as different.
DEFAULT_SIM_TYPE = "Fully amphicheiral"

VALID_SYM_TYPES = [
    "Chiral", # no symmetries
    "Fully amphicheiral", # K = -K = K* = -K*
    "Negative amphicheiral", # K = -K*
    #"Positively amphicheiral", K = K*, not actually in the database bc it's rare
    "Reversible" # K = -K
]

# processes a PD presentation from katlas into a nicer form
def process_PD(raw):
    nodes = raw.split(" ")
    nodes = [x.replace("</sub>","").replace("X<sub>","") for x in nodes]

    PD_code = []

    for node in nodes:
        if "," in node:
            new_node = node.split(",")
        else:
            new_node = list(node)
        
        PD_code += [int(x) for x in new_node]

    return PD_code

# takes a line from the rdf file and processes it
def extract_line_info(line, mode="PD"):
    # split into components
    line = line.strip()
    components = line.split(RDF_BREAKPOINT)

    # arcane string manipulation time!!!

    # extract the knot id
    knot_id = components[0].split("> ")[0].split(":")[-1]
    
    # strip irrelevant characters
    info = components[-1].replace('" .', "")

    if mode == "PD":
        # specially handle the unknot
        if knot_id == UNKNOT_ID:
            # we can't have an empty code for technical reasons
            # so instead we use the simplest non-empty code
            info = UNKNOT_CODE

        info = process_PD(info)
    
    elif mode == "sym":
        # specially handle the unknot
        if knot_id == UNKNOT_ID:
            # for some reason it's listed as having the wrong type
            info = UNKNOT_SYM_TYPE

    return (knot_id, info)

# extracts gauss codes from the katlas dataset
def get_knots(raw_filename):
    knots = dd(dict)

    # filter lines
    with open(raw_filename, "r") as source_file:
        for line in tqdm(source_file.readlines(), desc="Extracting knots from db..."):
            # extract planar diagram presentation
            if PD_IDENTIFIER in line:
                knot_id, code = extract_line_info(line, mode="PD")

                knots[knot_id][PD_CODE] = code

            # extract symmetry type
            elif SYM_IDENTIFIER in line:
                knot_id, sym_type = extract_line_info(line, mode="sym")

                knots[knot_id][SYM_TYPE] = sym_type
            
            # check if it's a link
            # this information can be extracted from the gauss code
            elif GC_IDENTIFIER in line:
                # we're only using the knot id
                # this is not a sym line so the sym_type will mean nothing
                knot_id, _ = extract_line_info(line, mode="sym")
                
                knots[knot_id][IS_LINK] = LINK_IDENTIFIER in line
    
    # throw away links
    real_knots = {}

    for knot_id, knot in knots.items():
        if not knot[IS_LINK]:
            # it's a real knot, add to the list
            real_knots[knot_id] = knot

            # we don't need this key anymore
            del real_knots[knot_id][IS_LINK]

            # some knots have weird broken symmetry types, fix this
            if SYM_TYPE not in knot.keys() or knot[SYM_TYPE] not in VALID_SYM_TYPES:
                knot[SYM_TYPE] = DEFAULT_SIM_TYPE

    return real_knots

# given a PD code, precomputes the "other occurrance" lookup table
# this saves time in the main Shields algorithm
def get_other_occurrance_table(code):
    lookup = [None for x in code]
    
    # keeps track of the first time we saw a symbol
    first_time = [None for x in range(max(code))]

    for pos, x in enumerate(code):
        # the minus 1 is to account for the difference in indexing
        # 0 vs 1 indexed
        if first_time[x-1] == None:
            # seen this symbol for the first time,
            # don't know where the other one is yet
            first_time[x-1] = pos
        else:
            # second time found, update the lookup
            lookup[first_time[x-1]] = pos
            lookup[pos] = first_time[x-1]
    
    # sanity check
    if None in lookup:
        raise Exception("Malformed PD code detected")

    return lookup

# takes in a PD code and calculates the orientation of each node
# this is called the Shields algorithm in my masters notes
# the algorithm is explained in more detail there
def calculate_orientations(code, other_occurrance_table=None):
    # initialise the directions array
    directions = [None for x in code]

    n_nodes = len(code)//4

    for x in range(n_nodes):
        directions[4*x] = -1
        directions[4*x+2] = 1
    
    # calculate the other occurance table if it was not provided
    if other_occurrance_table == None:
        other_occurrance_table = get_other_occurrance_table(code)

    # calculate the unknown orientations
    iterations = 0
    while None in directions:
        iterations += 1

        for x in range(n_nodes):
            # get indexes of the over symbols
            # using slightly different notation to the notes
            odd_index_1 = 4*x+1
            odd_index_2 = 4*x+3

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

    # extract the orientations
    orientations = [directions[4*x+1] for x in range(n_nodes)]
    return orientations

# takes in the processes data from the RDF file and converts them to Garbali graphs
def get_graphs(knots):
    graphs = []

    # read all the PD codes
    for knot_id, knot in tqdm(knots.items(), desc="Constructing graphs..."):
        code = knot[PD_CODE]
        sym_type = knot[SYM_TYPE]

        edges = []
        edge_colors = []

        # calculate the other occurrance table
        other_occurrance_table = get_other_occurrance_table(code)
        orientations = calculate_orientations(code, other_occurrance_table)

        # build up the graph
        # see master's notes: the PD to Garbali algorithm
        # we're using a slight modification where we use the Shields algo for orientations
        current = 0

        for x in range(max(code)):
            # find the opposite edge
            current_node = current//4
            opposite = 4*current_node + (current+2)%4

            # find where the opposite edge connects to
            next_one = other_occurrance_table[opposite]
            next_node = next_one//4

            # work out what the edge color should be
            # even positions are under, odd are over
            source_crossing_type = (-1)**(opposite % 2)
            target_crossing_type = (-1)**(next_one % 2)

            # save the data
            edges.append((current_node, next_node))
            edge_colors.append(
                color_function(source_crossing_type, target_crossing_type)
            )

            # move to the next one
            current = next_one

        # convert to tensors
        edges       = format_for_pytorch_geo(edges,                          new_type=torch.long)
        nodes       = format_for_pytorch_geo(orientations, new_shape=(1,-1), new_type=torch.float)
        edge_colors = format_for_pytorch_geo(edge_colors,  new_shape=(1,-1), new_type=torch.float)

        # instantiate the graph
        graph = tg.data.Data(
            edge_index=edges,
            edge_attr=edge_colors,
            x=nodes,
            knot_id=knot_id
        )

        # check there's no mistakes
        graph.validate(raise_on_error=True)

        # generate non-equivalent graphs depenind on symmetry type
        variants = []

        transformations_to_do = NEEDED_TRANSFORMS[sym_type]
        
        if len(transformations_to_do) == 1:
            # it's just the identity
            variants = [graph]
        else:
            for pos, transform in enumerate(transformations_to_do):
                new_graph = transform(graph)
                new_graph.knot_id = f"{graph.knot_id} v{pos+1}"
                assert(new_graph is not graph)
                variants.append(new_graph)
        
        # save the graphs
        graphs += variants
    
    return graphs