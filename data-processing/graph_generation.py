from tqdm import tqdm
import torch_geometric as tg
import torch

BREAKPOINT = ":"

input_filename = "katlas_gauss.rdf"

graphs = []


def color_function(start, end):
    """
        Edge coloring piecewise function.
        Swapping both crossing types is the same as
        multiplying by -1. 
        See master's notes: The Garbali-Gauss construction.
    """
    if start < 0 and end < 0:
        return -2
    elif start < 0 and end > 0:
        return -1
    elif start > 0 and end < 0:
        return 1
    else:
        return 2

# read all the gauss codes
with open(input_filename, "r") as file:
    for line in tqdm(file.readlines()):
        # extract the code
        gauss_code = [int(x) for x in line.strip().split(BREAKPOINT)[-1].split(",")]

        # add periodic condition
        gauss_code.append(gauss_code[0])

        edges = []
        edge_colors = []

        # build up the graph
        # see master's notes: the Garbali-Gauss construction
        for pos, node in enumerate(gauss_code[:-1]):
            # get the next node
            next_node = gauss_code[pos+1]

            # add directed edge
            # the minus 1 converts to python indexing
            edges.append((abs(node)-1, abs(next_node)-1))

            # color the edge
            edge_colors.append(color_function(node, next_node))

        # convert to tensors
        edges = torch.tensor(edges).t().contiguous().type(torch.long)
        edge_colors = torch.tensor(edge_colors).reshape((1,-1)).t().contiguous().type(torch.float)

        # instantiate the graph
        graph = tg.data.Data(
            edge_index=edges,
            edge_attr=edge_colors,
            num_nodes=max(gauss_code)
        )

        graph.validate(raise_on_error=True)

        graphs.append(graph)

print(graphs[1].edge_attr)