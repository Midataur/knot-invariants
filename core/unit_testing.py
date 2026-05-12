# a bunch of unit tests to make sure that everything is correct
import torch
import torch.testing
import unittest
import time
import processing
import transformations

# tests the transformation code
class TestTransformations(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # get graphs
        raw_filename = "../datasets/raw_dir/katlas.rdf"

        self.knots = processing.get_knots(raw_filename)
        self.graphs = processing.get_graphs(self.knots)
        self.trefoil = self.graphs[0]

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print("Test %s done in %.3f seconds" % (self.id(), t))

    def test_processing_types(self):
        """Checks that everything in the graphs that should be a tensor is."""

        for graph in self.graphs:
            self.assertIsInstance(graph.x, torch.Tensor, msg=f"id is {graph.knot_id}")
            self.assertIsInstance(graph.edge_index, torch.Tensor, msg=f"id is {graph.knot_id}")
            self.assertIsInstance(graph.edge_attr, torch.Tensor, msg=f"id is {graph.knot_id}")

    def test_twist(self):
        """Try twisting the first edge with parity 1"""
        post_twist = transformations.twist(
            self.trefoil,
            0,
            1
        )

        # check the nodes
        correct_x = torch.tensor([
            -1, 
            -1, 
            -1,
            -1 # added new node
        ]).reshape((-1,1)).float()

        torch.testing.assert_close(post_twist.x, correct_x)

        # check the edges
        correct_edge_index = torch.tensor([
            #[0, 2], deleted the first edge
            [2, 1],
            [1, 0],
            [0, 2],
            [2, 1],
            [1, 0],

            [0, 3], # added new edges
            [3, 3],
            [3, 2]
        ]).t()

        torch.testing.assert_close(post_twist.edge_index, correct_edge_index)

        # check the edge colors
        correct_colors = torch.tensor([
            #1, removed the first edge
            -1,
            1,
            -1,
            1,
            -1,

            2, # added new ones
            1,
            -2
        ]).reshape((-1, 1)).float()

        torch.testing.assert_close(post_twist.edge_attr, correct_colors)

        post_twist.validate()
    
    def test_untwist(self):
        """Tries twisting and untwisting every knot."""
        for graph in self.graphs:
            og_edges = sorted(graph.edge_index.t().tolist())

            undone = transformations.untwist(
                transformations.twist(graph, 0, 1), 
                len(graph.x) # the index of the added node
            )

            new_edges = sorted(undone.edge_index.t().tolist())

            self.assertListEqual(og_edges, new_edges, msg=f"id is {graph.knot_id}")

            undone.validate()

if __name__ == "__main__":
    unittest.main()