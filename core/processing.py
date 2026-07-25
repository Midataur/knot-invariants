from graph_functions import color_function
from collections import defaultdict as dd
from tqdm import tqdm
from pd_functions import *
from pd_transformations import *
from utilities import *
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

def process_PD(raw):
    """
        Processes a PD presentation from katlas into a nicer form.

        Specifically, the output is just the list of integers.
    """

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

def extract_line_info(line, mode="PD"):
    """
        Takes a line from the rdf file and processes it.
    """

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

                # make the pd code zero indexed
                zero_indexed_code = [x-1 for x in code]

                knots[knot_id][PD_CODE] = zero_indexed_code

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

def graph_from_pd_code(pd_code):
    """Turns a planar diagram code into the corresponding graph."""

    edges = []
    edge_colors = []

    # calculate the other occurrance table
    other_occurrance_table = get_pd_other_occurrance_table(pd_code)
    orientations = calculate_orientations(pd_code, other_occurrance_table)

    # build up the graph
    # see master's notes: the PD to Garbali algorithm
    # we're using a slight modification where we use the Shields algo for orientations
    current_pos = 0

    # we want the edge labels to match the pd code
    # we use this list to make sure that happens
    pd_code_edge_labels = []

    for x in range(max(pd_code)+1):
        # find the opposite edge
        current_node = current_pos//EDGES_PER_NODE
        opposite = current_node*EDGES_PER_NODE + (current_pos+2)%EDGES_PER_NODE

        # find where the opposite edge connects to
        next_pos = other_occurrance_table[opposite]
        next_node = next_pos//EDGES_PER_NODE

        # work out what the edge color should be
        # even positions are under (-1), odd are over (+1)
        source_crossing_type = (-1)**(opposite % 2 + 1)
        target_crossing_type = (-1)**(next_pos % 2 + 1)

        # find the pd code label
        pd_code_edge_labels.append(pd_code[next_pos])

        # add the edge
        edges.append((current_node, next_node))
        edge_colors.append(
            color_function(source_crossing_type, target_crossing_type)
        )

        # move to the next one
        current_pos = next_pos
    
    # sort the edges to match the edge label order
    sorted_edges = []
    sorted_edge_colors = []

    for label, edge, color in sorted(zip(
        pd_code_edge_labels, edges, edge_colors, 
        strict=True
    )):
        sorted_edges.append(edge)
        sorted_edge_colors.append(color)

    # convert to tensors
    nodes              = format_for_pytorch_geo(orientations,        new_shape=(1,-1), new_type=torch.float)
    sorted_edges       = format_for_pytorch_geo(sorted_edges,                          new_type=torch.long )
    sorted_edge_colors = format_for_pytorch_geo(sorted_edge_colors,  new_shape=(1,-1), new_type=torch.float)

    # instantiate the graph
    graph = tg.data.Data(
        edge_index=sorted_edges,
        edge_attr=sorted_edge_colors,
        x=nodes,
        pd_code=pd_code,
        faces=faces_from_pd_code(pd_code)
    )

    # check there's no mistakes
    graph.validate(raise_on_error=True)
    
    return graph

def get_graphs(knots):
    """
        Takes in the processed data from the RDF file and converts them to Garbali graphs.
    """

    graphs = []

    # read all the PD codes
    for knot_id, knot in tqdm(knots.items(), desc="Constructing graphs..."):
        code = knot[PD_CODE]
        sym_type = knot[SYM_TYPE]

        # generate non-equivalent codes depending on symmetry type
        variants = []

        transformations_to_do = NEEDED_PD_TRANSFORMS[sym_type]
        
        # compute the transformed codes
        for transform in transformations_to_do:
            variants.append(transform(code))

        # compute the associated graphs
        for number, code in enumerate(variants):
            graph = graph_from_pd_code(code)
            graph.knot_id = f"{knot_id} v{number+1}"

            # save the graph
            graphs.append(graph)

    return graphs