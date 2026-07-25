# a bunch of unit tests to make sure that everything is correct
from pd_functions import faces_from_pd_code
from graph_functions import *
from graph_transformations import *
from pd_functions import *
from pd_transformations import *
import torch
import torch.testing
import unittest
import time
import processing
import graph_transformations as graph_transformations

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
        """
            Try twisting the first edge using graph and pd.

            Makes sure they give the same answer.
        """

        for graph in self.graphs:
            for option1 in (UNDERCROSSING, OVERCROSSING):
                for option2 in (-1, 1):
                    via_graph_twist = graph_twist(graph, 0, option1, option2, **SUPPRESSED_DEFAULT._asdict())
                    via_pd_twist = pd_twist(graph.pd_code, 0, option1, option2)

                    # compare the codes
                    code_from_graph_twist = get_pd_code_from_graph(via_graph_twist, **SUPPRESSED_DEFAULT._asdict())

                    self.assertListEqual(
                        code_from_graph_twist, via_pd_twist, 
                        msg=f"Broke on {graph.knot_id} with settings {option1} {option2}"
                    )
    
    def test_pd_untwist(self):
        """Tries twisting and untwisting every knot using pd transformations."""

        # for graph in self.graphs:
        #     for option1 in (UNDERCROSSING, OVERCROSSING):
        #         for option2 in (-1, 1):
        #             twisted_pd = pd_twist(graph.pd_code)
                    

        raise NotImplementedError("PD UNTWIST NOT IMPLEMENTED")
    
    def test_faces(self):
        """
            Calculates the faces of all the base knots in two different ways,
            then checks to see that they both give the same answer.
        """

        for graph in self.graphs:
            # calculate from graph
            graph_faces = graph_get_faces(graph, **SUPPRESSED_DEFAULT._asdict())

            # calculate from pd code
            pd_code = graph.pd_code
            pd_faces = faces_from_pd_code(pd_code)

            self.assertSetEqual(
                graph_faces, pd_faces,
                msg=f"failed on {graph.knot_id}"
            )

            self.assertSetEqual(
                graph.faces, pd_faces,
                msg=f"failed on {graph.knot_id}"
            )

    def test_reverse(self):
        """
            Reverses each base knot via graph and pd methods.

            Checks that these both give the same pd code.
        """

        for graph in self.graphs:
            # reverse the graph with graph methods
            reversed_graph = graph_reverse_knot(graph, **SUPPRESSED_DEFAULT._asdict())
            pd_from_graph = get_pd_code_from_graph(reversed_graph, **SUPPRESSED_DEFAULT._asdict())

            # reverse using pd methods
            pd_code = graph.pd_code
            reversed_pd_code = pd_reverse_knot(pd_code)

            self.assertListEqual(pd_from_graph, reversed_pd_code, msg=f"Broke on {graph.knot_id}.")

    def test_mirror(self):
        """
            Mirrors each base knot via graph and pd methods.

            Checks that these both give the same pd code.
        """

        for graph in self.graphs:
            # reverse the graph with graph methods
            mirrored_graph = graph_mirror_knot(graph, **SUPPRESSED_DEFAULT._asdict())
            pd_from_graph = get_pd_code_from_graph(mirrored_graph, **SUPPRESSED_DEFAULT._asdict())

            # reverse using pd methods
            pd_code = graph.pd_code
            mirrored_pd_code = pd_mirror_knot(pd_code)

            self.assertListEqual(pd_from_graph, mirrored_pd_code, msg=f"Broke on {graph.knot_id}.")


if __name__ == "__main__":
    unittest.main()