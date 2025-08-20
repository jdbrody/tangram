import unittest
import numpy as np
from relation import Polygon, compute_relation, Relation

class TestPolygonRelations(unittest.TestCase):
    def setUp(self):
        # Define some simple polygons
        self.square1 = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        self.square2 = Polygon([(20, 0), (40, 0), (40, 20), (20, 20)])  # Tangential to square1
        self.square3 = Polygon([(300, 300), (500, 300), (500, 500), (300, 500)])  # Disconnected
        self.square4 = Polygon([(20, 0), (40, 0), (40,10), (20, 10)])  # TOVI
        self.square5 = Polygon([(20, -10), (40, -10), (40, 10), (20, 10)  ])  # TO
        self.triangle = Polygon([(40, 0), (20, 10), (50, 20)])  # VEC
        
    def test_tangential_equal(self):
        relation, score = compute_relation(self.square1, self.square2)
        self.assertEqual(relation, Relation.TEQ, "Failed to classify tangential equality")
    
    def test_disconnected(self):
        relation, score = compute_relation(self.square1, self.square3)
        self.assertEqual(relation, Relation.DC, "Failed to classify disconnected polygons")
    
    def test_tangential_overlapping(self):
        relation, score = compute_relation(self.square1, self.square5)
        self.assertEqual(relation, Relation.TO, "Failed to classify tangential overlapping")
    
    def test_tangential_one_vertex_included(self):
        relation, score = compute_relation(self.square1, self.square4)
        self.assertEqual(relation, Relation.TOVI, "Failed to classify tangential one vertex included")
    
    def test_vertex_connection(self):
        relation, score = compute_relation(self.square1, self.triangle)
        self.assertEqual(relation, Relation.VEC, "Failed to classify vertex edge connection")

if __name__ == "__main__":
    unittest.main()
