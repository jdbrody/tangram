import numpy as np
import itertools
from enum import Enum, auto
from numpy.linalg import norm
from PIL import Image, ImageDraw, ImageColor
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import json
import math


from util import reln_sexp

_DC_THRESHOLD = 10.0
_TOUCH_THRESHOLD = 1.0   # Distance needs to be under this to possibly be touching
_COS_PARALELL_THRESH = 0.86  # cos(30) -- vectors with cosine below cannot be parallel
_COS_PERP_THRESH = 0.866  # cos(30) -- vectors with cosine above cannot be perpendicular
_PWEIGHT = 100.0
_LWEIGHT = 10.0

_debug = False
class Relation(Enum):
    """Enum for representing the relation between two entities."""
    TEQ = auto(),    # tangential equal
    VC = auto(),     # vertex connection
    TOVI = auto(),   # tangential one vertex included
    TOVII = auto(),  # tangential over vertex included, inverse
    TO = auto(),     # tangential overlapping
    TEI = auto(),    # tangential edge included
    TEII = auto(),   # tangential edge included, inverse
    VEC = auto(),    # vertex edge connection
    DC = auto()     # disconnected

"""
More primitive edge relations:
-- Same length
-- Longer
-- Parallel
-- Vertex to edge
"""


def compute_cos(edge1, edge2):
    dir1 = edge1[1] - edge1[0]
    dir2 = edge2[1] - edge2[0]
    cos = np.dot(dir1, dir2)/ (norm(dir1) * norm(dir2))
    return cos

def parallel_cost(edge1, edge2):
    # Modulo _COS_PARALELL_THRESH returns _PWEIGHT if edges are pependicular; 0 if edges are parallel.
    cos = compute_cos(edge1, edge2)
    if np.abs(cos) < _COS_PARALELL_THRESH:
        return np.inf
    else:
        return _PWEIGHT*(1 - np.abs(cos))
    
def perpendicular_cost(edge1, edge2):
    # Modulo _COS_PERP_THRESH returns _PWEIGHT if edges are parallel; 0 if edges are perpendicular
    cos = compute_cos(edge1, edge2)
    if np.abs(cos) > _COS_PERP_THRESH:
        return np.inf
    else:
        # if _debug:
        #     print(f" perp_cost; cosine is: {cos}")
        return _PWEIGHT*(np.abs(cos))    

def same_length_cost(edge1, edge2):
    return abs(norm(edge1) - norm(edge2))

def edge_norm(edge):
    return norm(edge[1] - edge[0])


def longer_cost(edge1, edge2):
    # To what degree is it false that edge1 is longer than edge2
    # I.e., the cost associated with the assertion that edge1 is longer
    edge2_longer = norm(edge2) - norm(edge1)
    return max(0, edge2_longer)

def longer_cost2(edge1, edge2):
    # Assert that edge2 is longer than edge1
    l1 = norm(edge1[1] - edge1[0])
    l2 = norm(edge2[1] - edge2[0])
    ratio = l1 / l2
    if ratio < .75:
        return 0
    else:
        return _LWEIGHT*np.exp(ratio - .75)


def vertex_to_edge_cost(point, edge):
    # From
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment.
    # More common solutions extend the line segment defined by the
    # edge to an infinite line; this solution finds the minimal distance to a
    # point on the line segment, not the line.
    x3, y3 = point
    x1, y1 = edge[0]
    x2, y2 = edge[1]

    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx*dx + dy*dy)**.5

    return dist

def order_vertices(vertices):
    """
    Reorders a list of 2D vertices to be in clockwise order, with the
    vertex having the largest internal angle placed in the second position.

    The function performs three main steps:
    1.  Orders the vertices into a clockwise sequence using the centroid method.
    2.  Calculates the internal angle for each vertex in the polygon.
    3.  Rotates the ordered list so that the vertex with the largest internal
        angle is at index 1.

    Args:
        vertices (List[List[float]]): A list of vertices, where each vertex
                                      is a list or tuple of two floats [x, y].

    Returns:
        List[List[float]]: The reordered list of vertices.
    """
    # An internal angle requires at least 3 vertices to be defined.
    if len(vertices) < 3:
        return vertices

    # Step 1: Get an initial clockwise ordering of vertices.
    # This uses the angle-to-centroid method from the original function.
    centroid = np.mean(vertices, axis=0)
    angles_from_centroid = [math.atan2(v[1] - centroid[1], v[0] - centroid[0]) for v in vertices]
    #clockwise_vertices = [v for _, v in sorted(zip(angles_from_centroid, vertices), key=lambda p: p[0], reverse=True)]
    clockwise_vertices = [v for _, v in sorted(zip(angles_from_centroid, vertices), key=lambda p: p[0], reverse=False)]

    num_vertices = len(clockwise_vertices)
    
    # Step 2: Calculate the interior angle at each vertex.
    internal_angles = []
    for i in range(num_vertices):
        # Get the current vertex (P) and its neighbors (A and B)
        p = np.array(clockwise_vertices[i])
        a = np.array(clockwise_vertices[i - 1]) # Previous vertex
        b = np.array(clockwise_vertices[(i + 1) % num_vertices]) # Next vertex

        # Create vectors from P to its neighbors A and B
        vec_pa = a - p
        vec_pb = b - p

        # Calculate the angle between the two vectors using the dot product.
        # angle = arccos((v1 . v2) / (|v1| * |v2|))
        dot_product = np.dot(vec_pa, vec_pb)
        mag_pa = np.linalg.norm(vec_pa)
        mag_pb = np.linalg.norm(vec_pb)
        
        # Avoid division by zero for coincident points
        if mag_pa == 0 or mag_pb == 0:
            angle = 0.0
        else:
            # Clip to handle potential floating-point inaccuracies
            cos_theta = np.clip(dot_product / (mag_pa * mag_pb), -1.0, 1.0)
            angle = math.acos(cos_theta)
        
        internal_angles.append(angle)

    # Step 3: Find the index of the vertex with the largest internal angle.
    max_angle_idx = np.argmax(internal_angles)

    # Step 4: Rotate the list to place that vertex at index 1.
    # The new starting element should be the vertex right before the one with
    # the largest angle in the clockwise sequence.
    start_idx = (max_angle_idx - 1 + num_vertices) % num_vertices
    
    final_order = clockwise_vertices[start_idx:] + clockwise_vertices[:start_idx]

    return final_order




    
class Polygon():
    def __init__(self, vertices):
        self._vertices = order_vertices([np.array(v) for v in vertices])
        self._labels = [chr( ord('a') + i) for i in range(len(self._vertices))]
        self._edges = []
        self._vertex_labels = {str(v): l for (l,v) in zip(self._labels, self._vertices)}
        for i in range(len(self._vertices)):
            self._edges.append( [self._vertices[i], self._vertices[ (i+1) % len(vertices)]])

    def vertices(self):
        return self._vertices

    def vertices_with_labels(self):
        return [(l,v) for (l,v) in zip(self._labels, self._vertices)]

    def edges(self):
        return self._edges

    def get_edge_name(self, e):
        #return f"({self._vertex_labels[str(e[0])]}-{self._vertex_labels[str(e[1])]})"
        return f"{self._vertex_labels[str(e[0])]}{self._vertex_labels[str(e[1])]}"

    def get_edge_name_pair(self, e):
        return (f"{self._vertex_labels[str(e[0])]}", f"{self._vertex_labels[str(e[1])]}")
    
    def get_vertex_name(self, v):
        return self._vertex_labels[str(v)]
    


def relative_edge_directions(edge1, edge2):                               
    base_point = edge1[0]   # Base directions on the first vertex of the first edge
    directed_edge1, directed_edge2  = [e -  base_point for e in edge1], [e - base_point for e in edge2]
    dir1, dir2 = directed_edge1[1], directed_edge2[1]
    if np.dot(dir1, dir2) < 0:
        dir2 = directed_edge1[0]
    return dir1, dir2
        
def compute_relation(polygon1, polygon2):
    mins, arg_mins = np.inf, None
    min_relation = None
    scores = {}
    for relation in Relation:
        #print(f"Computing relation {relation} between {polygon1} and {polygon2}")
        
        score, edge_pair = compute_relation_score(polygon1, polygon2, relation)
        scores[relation] = (score, edge_pair)
        if (score < mins) or (min_relation == Relation.VC and score == mins):
            mins = score
            min_relation = relation

        # if _debug:
        #     print(scores)

    if mins > _DC_THRESHOLD:
        mins = _DC_THRESHOLD
        min_relation = Relation.DC

    if (min_relation == Relation.VEC or min_relation == Relation.VC ) and scores[Relation.TEQ][0] < _DC_THRESHOLD:
        mins = scores[Relation.TEQ][0]
        min_relation = Relation.TEQ
        

    if _debug and min_relation != Relation.DC:
        print(f"\t\t\t\t Relation: {min_relation} edges: {scores[min_relation]}")
        
    return min_relation, scores[min_relation][0], scores[min_relation][1]




def teq_edge_cost(edge1, edge2):
    pcost = parallel_cost(edge1, edge2)
    vertex_cost = min( norm(edge1[0] - edge2[0]) + norm(edge1[1] - edge2[1]),
                       norm(edge1[0] - edge2[1]) + norm(edge1[1] - edge2[0]))
    alpha = 1.0
    beta = 4.0

    if _debug:
        #print(f"TEQ cost is {alpha*pcost + beta*vertex_cost}         pcost: {pcost} vertex_cost: {vertex_cost}")
        
        return alpha*pcost + beta*vertex_cost, (f"\t\tTEQ edge cost: {alpha*pcost + beta*vertex_cost} pcost: {pcost} vertex_cost: {vertex_cost}")
    return alpha*pcost + beta*vertex_cost



def tovi_edge_cost(edge1, edge2):
    # TOVI means that there is are vertices v1 from edge1 and v2 from
    # edge2 which are close to each other and the edges (v1, u1) and
    # (v2, u2) are parallel and in the same direction; i.e. the cosine is close to 1
    # Also, edge2 is longer than edge1

    vertex_distances = [ (norm(edge1[i] - edge2[j]), i, j) for i in range(2) for j in range(2)]
    vertex_cost, i, j  = min(vertex_distances)
    other = lambda i: 1-i

    v1 = edge1[i]
    v2 = edge2[j]
    u1 = edge1[other(i)]
    u2 = edge2[other(j)]

    dir1 = u1 - v1
    dir2 = u2 - v2
    cos_cost = _PWEIGHT * (1 - np.dot(dir1, dir2)/ (norm(dir1) * norm(dir2)))
    lcost = longer_cost2( edge1, edge2 )  # cost of edge1 being longer than edge2


    alpha = 2.0
    beta = 4.0
    gamma = 1.0
    if _debug:
        return alpha*cos_cost + beta*vertex_cost + gamma*lcost, f"\t\tTOVI edge cost: {alpha*cos_cost + beta*vertex_cost + gamma*lcost} vertex_cost: {vertex_cost} cos_cost: {cos_cost}  lcost: {lcost}"
    return alpha*cos_cost + beta*vertex_cost + gamma*lcost

def tovii_edge_cost(edge1, edge2):
    return tovi_edge_cost(edge2, edge1)

def to_edge_cost(edge1, edge2):
    # Find the furthest vertices on different edges, say v1, v2
    # Let u1, u2 be the other vertices.  Then TO corresponds to:
    #   a) u1 = v1 + ||u1 - v1|| * (v2 - v1)/||v2 - v1||
    #   b) u2 = v1 + ||u2 - v1|| * (v2 - v1)/||v2 - v1||
    #   c) ||v1 - u2|| << ||v1 - u1|| or ||v1 - u2|| / ||v1 - u1|| << 1
    vertex_distances = [ (norm(edge1[i] - edge2[j]), i, j) for i in range(2) for j in range(2)]
    max_distance, i, j = max(vertex_distances)
    other = lambda i: 1-i

    v1 = edge1[i]
    v2 = edge2[j]
    u1 = edge1[other(i)]
    u2 = edge2[other(j)]

    dir = (v2 - v1) / norm(v2 - v1)  # direction from v1 to v2
    vcost1 = norm(u1 - (v1 + norm(u1-v1) * dir))
    vcost2 = norm(u2 - (v1 + norm(u2-v1) * dir))
    lcost = longer_cost2( [v1,u2], [v1, u1] )  # cost of edge1 being longer than [v1,u2]
    ccost = parallel_cost(  [v1,u2], [v1, u1])

    alpha = 4.0
    beta = 4.0
    gamma = 1.0
    delta = 1.0

    if _debug:
        return alpha*vcost1 + beta*vcost2 + gamma*lcost + delta*ccost,  f"\t\tTO edge cost: {alpha*vcost1 + beta*vcost2 + gamma*lcost} vcost1: {vcost1} vcost2: {vcost2} lcost: {lcost} ccost: {ccost}. \t\t (v1, v2):  ({v1, v2}). (u1, u2): ({u1, u2}).   max_distance: {max_distance}"
    return alpha*vcost1 + beta*vcost2 + gamma*lcost + delta*ccost
    


def tei_edge_cost(edge1, edge2):
    # Let edge1 = [v1, v2] and edge2 = [u1, u2].
    # Then TEI corresponds to:
    #   a) u1 = v1 + ||u1 - v1|| * (v2 - v1)/||v2 - v1||
    #   b) u2 = v1 + ||u2 - v1|| * (v2 - v1)/||v2 - v1||
    #   c) ||u1 - u2|| << ||v1 - v2||

    v1, v2 = edge1
    u1, u2 = edge2
    dir = (v2 - v1) / norm(v2 - v1)  # direction from v1 to v2
    vcost1 = norm(u1 - (v1 + norm(u1-v1) * dir))
    vcost2 = norm(u2 - (v1 + norm(u2-v1) * dir))
    lcost = longer_cost2( edge2, edge1)

    alpha = 4.0
    beta = 4.0
    gamma = 1.0

    if _debug:
        return alpha*vcost1 + beta*vcost2 + gamma*lcost, f"\t\tTEI edge cost: {alpha*vcost1 + beta*vcost2 + gamma*lcost} vcost1: {vcost1} vcost2: {vcost2} lcost: {lcost}.   dir: {dir}    u1, u2, v1, v2:   {u1, u2, v1, v2}"
    return alpha*vcost1 + beta*vcost2 + gamma*lcost
    

def vc_edge_cost(edge1, edge2):
    vc_cost = np.min([ np.linalg.norm(v1 -v2) for (v1,v2) in itertools.product(edge1, edge2)])
    # we want to make sure that the edges AREN'T parallel, otherwise
    # this could be a tangency relation
    #pcost = _PWEIGHT - parallel_cost(edge1, edge2)
    pcost = perpendicular_cost(edge1, edge2)

    alpha = 4.0
    beta  = 5 / _PWEIGHT

    if _debug:
        return alpha*vc_cost + beta*pcost, f"\t\tVC edge cost: {alpha*vc_cost + beta*pcost} vc_cost: {vc_cost} pcost: {pcost}"
    return alpha*vc_cost + beta*pcost

def vec_edge_cost(edge1, edge2):
    """
    Cost of of thinking of a VEC relation between one vertex of an edge and the other edge.
    For extra information, return the actual vertex and edge.
    """
    all_vec_costs = [ (vertex_to_edge_cost(u, edge2), u, edge2, edge1) for u in edge1] + [ (vertex_to_edge_cost(v, edge1), v, edge1, edge2) for v in edge2]
    vec_cost, min_vec, min_edge, full_edge = min(all_vec_costs, key=lambda x: x[0])
    if vec_cost > _TOUCH_THRESHOLD:
        vec_cost = np.inf

    # we want to make sure that the edges AREN'T parallel, otherwise
    # this could be a tangency relation.
    
    # note that the min_vertex will be on two edges; one of which
    # could be parallel while the other is perpendicular -- need to
    # make sure we handle this situation.  
    
    pcost = perpendicular_cost(min_edge, full_edge)

    # For vec, we also want to make sure that the nearest spot on the edge isn't a vertex to avoid vc.
    # First, calculate the min cosine from the vertex to the edge
    # Then, if the min cosine is close to 1, we want to add a cost
    # to the vertex cost
    # I think this isn't the best approach -- let's just say the vertex can't be within 10% of the edge length of either end.

    if False:
        dir1 = min_vec - min_edge[0]
        dir2 = min_edge[1] - min_edge[0]
        cos1 = np.dot(dir1, dir2)/ (norm(dir1) * norm(dir2))
        dir3 = min_vec - min_edge[1]
        dir4 = min_edge[0] - min_edge[1]
        cos2 = np.dot(dir3, dir4)/ (norm(dir3) * norm(dir4))
        cos = 1 - np.abs(np.max([cos1, cos2]))
        cos_cost = 0 if cos < 0.5 else cos
    edge_len = edge_norm(min_edge)
    end_dist = np.min( [np.linalg.norm(min_vec - min_edge[i]) for i in range(2)] )
    if end_dist < 0.1 * edge_len:
        end_cost = np.inf
    else:
        end_cost = 1/ end_dist  #this should probably be somehow scaled with edge_len.  

    alpha = 4.0
    beta  = 5 / _PWEIGHT
    gamma = 10.0

    if _debug:
        print(f"VEC edge cost: {alpha*vec_cost + beta*pcost + gamma*end_cost} vec_cost: {vec_cost} pcost: {pcost} end_cost: {end_cost}. \t\t  min_vec, min_edge: {min_vec, min_edge} \t\t end_dist: {end_dist} edge_len: {edge_len} end_cost: {end_cost}")
        return alpha*vec_cost + beta*pcost + gamma*end_cost, f"\t\tVEC edge cost: {alpha*vec_cost + beta*pcost + gamma*end_cost} vec_cost: {vec_cost} pcost: {pcost} end_cost: {end_cost}. \t\t  min_vec, min_edge: {min_vec, min_edge}"
    return (alpha*vec_cost + beta*pcost + gamma*end_cost, (min_vec, min_edge)) a
    

def teii_edge_cost(edge1, edge2):
    return tei_edge_cost(edge2, edge2)

def compute_generic_score(polygon1, polygon2, edge_cost_function):
    min_cost = np.inf
    min_edge1, min_edge2 = None, None
    min_dbug = None
    for edge1 in polygon1.edges():
        for edge2 in polygon2.edges():
            if _debug:
                cost, dbug = edge_cost_function(edge1, edge2)
            else:
                cost, extra_info = edge_cost_function(edge1, edge2) 
            if cost < min_cost:
                min_cost = cost
                min_edge1, min_edge2 = edge1, edge2
                if _debug:
                    min_dbug = dbug
    if _debug:
        print(min_dbug)

    return min_cost, (min_edge1, min_edge2, extra_info)
    
def compute_relation_score(polygon1, polygon2, relation):
    cost_functions = {
        Relation.TEQ: teq_edge_cost,
        Relation.VC: vc_edge_cost,
        Relation.TOVI: tovi_edge_cost,
        Relation.TOVII: tovii_edge_cost,
        Relation.TO: to_edge_cost,
        Relation.TEI: tei_edge_cost,
        Relation.TEII: teii_edge_cost,
        Relation.VEC: vec_edge_cost,
        #Relation.DC: dc_edge_cost
    }
    if relation != Relation.DC:
        costf = cost_functions[relation]
        return compute_generic_score(polygon1, polygon2, costf)
    else:
        return np.inf, None

def compute_teq_score(polygon1, polygon2):
    """ Tangential equal corresponds to one edge of the first polygon aligning exactly with one edge of the second polygon."""
    return compute_generic_score(polygon1, polygon2, teq_edge_cost)

def compute_to_score(polygon1, polygon2):
    """Tangential overlapping corresponds to one edge of the first
    polygon aligning exactly with one edge in direction but with a
    translation or different lengths."""
    return compute_generic_score(polygon1, polygon2, to_edge_cost)






def compute_vc_score(polygon1, polygon2):
    """ Vertex connection corresponds to one vertex of the first polygon aligning exactly with one vertex of the second polygon."""
    score = np.inf
    min_v1, min_v2 = None, None
    for vertex1 in polygon1.vertices():
        for vertex2 in polygon2.vertices():
            if np.linalg.norm(vertex1 - vertex2) < score:
                score = np.linalg.norm(vertex1 - vertex2)
                min_v1, min_v2 = vertex1, vertex2
    return score, (min_v1, min_v2)

def compute_tangential_score(polygon1, polygon2):
    """ Tangential overlapping corresponds to a normalized edge of the first polygon aligning exactly with a normalized edge of the second polygon."""
    score = np.inf
    min_edge1, min_edge2 = None, None
    for edge1 in polygon1.edges():
        for edge2 in polygon2:
            norm_edge1 = edge1 / np.linalg.norm(edge1)
            norm_edge2 = edge2 / np.linalg.norm(edge2)
            if np.dot(edge1, edge2) > 0:
                if np.linalg.norm(norm_edge1 - norm_edge2) < score:
                    score = np.linalg.norm(norm_edge1 - norm_edge2)
                    min_edge1, min_edge2 = edge1, edge2
            else:
                if np.linalg.norm(norm_edge1 + norm_edge2) < score:
                    score = np.linalg.norm(norm_edge1 + norm_edge2)
                    min_edge1, min_edge2 = edge1, edge2
    return score, (min_edge1, min_edge2)

def compute_tovi_score(polygon1, polygon2):
    """ The difference between tovi and tovii seems to be whether the aligned edge is longer on polygon1 or polygon2.
    So for tovi we only check pairs of edges where the edge of polygon1 is longer than the edge of polygon2.
    """
    score = np.inf
    min_edge1, min_edge2 = None, None
    for edge1 in polygon1.edges():
        for edge2 in polygon2.edges():
            d1, d2 = relative_edge_directions(edge1, edge2)
            directional_sim = 1 - ( np.dot(d1, d2) / ( norm(d1) * norm(d2) ) )
            single_vertex_sim = np.min( [norm(v1 - v2) for v1 in edge1 for v2 in edge2])  # distance between closest vertices
            longest_edge_len = max( edge_norm(edge1), edge_norm(edge2) )
            shortest_edge_len = min( edge_norm(edge1), edge_norm(edge2) )
            edge_len_sim = shortest_edge_len / longest_edge_len
            sim = directional_sim + single_vertex_sim + edge_len_sim
            if sim < score:
                score = sim
                min_edge1, min_edge2 = edge1, edge2
    return score, (min_edge1, min_edge2)
    
    

def compute_tovii_score(polygon1, polygon2):
    """ The difference between tovi and tovii seems to be whether the aligned edge is longer on polygon1 or polygon2.
    So for tovi we only check pairs of edges where the edge of polygon1 is shorter than the edge of polygon2.
    """
    return compute_tovi_score(polygon2, polygon1)


def compute_to_score(polygon1, polygon2):
    score = np.inf
    min_edge1, min_edge2 = None, None
    for edge1 in polygon1.edges():
        for edge2 in polygon2.edges():
            # Vertex distance is the sum of the distances between
            # endpoints of the two edges.  We take the minimum over
            # each of the possible orientations of the edges.
            vd1 = np.linalg.norm(edge1[0] - edge2[0]) + np.linalg.norm(edge1[1] - edge2[1])
            vd2 = np.linalg.norm(edge1[0] - edge2[1]) + np.linalg.norm(edge1[1] - edge2[0])
            vertex_distance = min(vd1, vd2)
            loss = np.abs(np.dot(edge1, edge2)) + vertex_distance
            if loss < score:
                score = loss
                min_edge1, min_edge2 = edge1, edge2
    return score, (min_edge1, min_edge2)
    
def compute_tei_score(polygon1, polygon2):
    score = np.inf
    min_edge1, min_edge2 = None, None
    for edge1 in polygon1.edges():
        for edge2 in polygon2.edges():
            # Vertex distance is the sum of the distances between
            # endpoints of the two edges.  We take the minimum over
            # each of the possible orientations of the edges.
            vd1 = np.linalg.norm(edge1[0] - edge2[0]) + np.linalg.norm(edge1[1] - edge2[1])
            vd2 = np.linalg.norm(edge1[0] - edge2[1]) + np.linalg.norm(edge1[1] - edge2[0])
            vertex_distance = min(vd1, vd2)
            loss = np.abs(np.dot(edge1, edge2)) + vertex_distance
            if loss < score:
                score = loss
                min_edge1, min_edge2 = edge1, edge2
    return score, (min_edge1, min_edge2)    

def compute_teii_score(polygon1, polygon2):
    return compute_tei_score(polygon2, polygon1)

def compute_vec_score(polygon1, polygon2):
    return np.inf

def compute_dc_score(polygon1, polygon2):
    return np.inf


relation_functions = {
    Relation.TEQ: compute_teq_score,
    Relation.VC: compute_vc_score,
    Relation.TOVI: compute_tovi_score,
    Relation.TOVII: compute_tovii_score,
    Relation.TO: compute_to_score,
    Relation.TEI: compute_tei_score,
    Relation.TEII: compute_teii_score,
    Relation.VEC: compute_vec_score,
    Relation.DC: compute_dc_score
}


def test1():
    red = Polygon([(923, 24), (901, 57), (929, 77)])
    orange = Polygon([(984, 94), (948, 125), (990, 154)])
    green = Polygon([(1063, 930), (1058, 995), (1094, 995)])
    yellow = Polygon([(935, 83), (915, 102), (942, 117), (960, 97)])
    purple =  Polygon([(821, 0), (821, 2), (829, 2), (829, 0)])
    blue =  Polygon([(946, 45), (957, 59), (958, 52)])

    r0, s0 = compute_relation(yellow, orange)

    
    polygons = [red, orange, green, yellow, purple, blue]
    for polygon1, polygon2 in itertools.combinations(polygons, 2):

        relation, score = compute_relation(polygon1, polygon2)
        print(f"{relation} {score}")


def parse_robot_jsons_7_9(json_files, img_dir):
    """
    Parse a list of s??_robot.json files into the same polygons dict format.

    Args:
        json_files (List[str]): Paths to your s??_robot.json files.
        img_dir (str): Directory where screenshots live (named like s06_robot_frame_4-26.jpg).

    Returns:
        Dict[str, Dict]: 
            {
                "s06_robot_frame_4-26.jpg": {
                    "polygons": [
                        {"color": "Red",    "polygon": <Polygon(...)>},
                        {"color": "Blue",   "polygon": <Polygon(...)>},
                        ...
                    ],
                    "image": <PIL.Image.Image at ...>
                },
                ...
            }
    """
    polygons = {}

    all_img_files = set(os.listdir(img_dir))
    for json_path in json_files:
        base = os.path.splitext(os.path.basename(json_path))[0]  # e.g. "s06_robot"
        with open(json_path, "r") as f:
            D = json.load(f)

        for frame in D.get("frames_data", []):
            ts_ms = frame.get("frame_timestamp_ms")
            if ts_ms is None:
                continue

            # convert to minutes and zero-padded seconds
            total_s = ts_ms / 1000.0
            m = int(total_s // 60)
            s = int(total_s % 60)
            time_suffix = f"{m}-{str(s).zfill(2)}"

            img_name = f"{base}_frame_{time_suffix}.jpg"
            entry = {"polygons": []}


            if img_name not in all_img_files:
                continue

            entry["timestamp"] = ts_ms
            img_path = os.path.join(img_dir, img_name)
            try:
                entry["image"] = Image.open(img_path)
            except FileNotFoundError:
                #print(f"⚠️  Image {img_name} not found in {img_dir}")
                continue

            for piece in frame.get("pieces", []):
                verts = piece.get("vertices")
                if isinstance(verts, list) and len(verts) >= 3:
                    try:
                        poly = Polygon(verts)
                        entry["polygons"].append({
                            "color": piece.get("class_name").split('_')[0].capitalize(),
                            "polygon": poly
                        })
                    except Exception as e:
                        print(f"⚠️  Could not make Polygon for {img_name}: {e}")



            polygons[img_name] = entry

    return polygons

def parse_robot_jsons5_10(json_files, img_dir):
    """
    Parse a list of s??_robot.json files into the same polygons dict format.

    Args:
        json_files (List[str]): Paths to your s??_robot.json files.
        img_dir (str): Directory where screenshots live (named like s06_robot_frame_4-26.jpg).

    Returns:
        Dict[str, Dict]: 
            {
              "s06_robot_frame_4-26.jpg": {
                 "polygons": [
                   {"color": "Red",   "polygon": <Polygon(...)>},
                   {"color": "Blue",  "polygon": <Polygon(...)>},
                    ...
                 ],
                 "image": <PIL.Image.Image at ...>
              },
              ...
            }
    """
    import json
    import os
    from PIL import Image
    polygons = {}

    all_img_files = set(os.listdir(img_dir))
    for json_path in json_files:
        base = os.path.splitext(os.path.basename(json_path))[0]  # e.g. "s06_robot"
        with open(json_path, "r") as f:
            D = json.load(f)

        for frame in D.get("frames_data", []):
            ts_ms = frame.get("frame_timestamp_ms")
            if ts_ms is None:
                continue

            # convert to minutes and zero-padded seconds
            total_s = ts_ms / 1000.0
            m = int(total_s // 60)
            s = int(total_s % 60)
            time_suffix = f"{m}-{str(s).zfill(2)}"

            img_name = f"{base}_frame_{time_suffix}.jpg"
            entry = {"polygons": []}


            if img_name not in all_img_files:
                continue

            entry["timestamp"] = ts_ms
            img_path = os.path.join(img_dir, img_name)
            try:
                entry["image"] = Image.open(img_path)
            except FileNotFoundError:
                #print(f"⚠️  Image {img_name} not found in {img_dir}")
                continue            

            for piece in frame.get("pieces", []):
                verts = piece.get("vertices")
                if isinstance(verts, list) and len(verts) >= 3:
                    try:
                        poly = Polygon(verts)
                        entry["polygons"].append({
                            "color": piece.get("color_name"),
                            "polygon": poly
                        })
                    except Exception as e:
                        print(f"⚠️  Could not make Polygon for {img_name}: {e}")



            polygons[img_name] = entry

    return polygons
        
def test_get_json_data(may_10_style = False,
                       july_9_style = False):
    import glob
    if may_10_style:
        img_dir ="screenshots2"
        json_files = glob.glob("5-10/s*robot.json")
        return parse_robot_jsons(json_files, img_dir)
    elif july_9_style:
        img_dir ="screenshots2"
        json_files = glob.glob("7-9/s*robot.json")
        return parse_robot_jsons_7_9(json_files, img_dir)    
        
    import json, os
    with open("output_vertices.json", "r") as f:
        data = json.load(f)
    img_dir = "screenshots2"
    polygons = {}
    for img in data['images']:
        img_name = img['image_name']
        polygons[img_name] = {}

        polygons[img_name]['polygons'] = []
        for color in img['vertices']:
            if type(img['vertices'][color]) == type([]):
                polygons[img_name]['polygons'].append(
                    {'color': color, 'polygon': Polygon(img['vertices'][color])})
        try:
            polygons[img_name]['image'] = Image.open(f"{img_dir}/{img_name}")
        except FileNotFoundError:
            print(f"Image {img_name} not found in {img_dir}")
            continue
    return polygons
    import os


def save_polygon_renders(polygons_dict, output_dir, background_color="white"):
    """
    For each entry in polygons_dict, draw all of its polygons
    (filled in their color_name) on a blank canvas and save
    the result to output_dir under the same filename.

    Args:
        polygons_dict (dict): 
            {
              "img_name.jpg": {
                 "polygons": [
                     {"color": "Red", "polygon": <shapely.geometry.Polygon>},
                     ...
                 ],
                 "image": <PIL.Image.Image>    # optional, but preferred
              },
              ...
            }
        output_dir (str): where to save the rendered images
        background_color (str or tuple): canvas background;
            any PIL-compatible color spec (default “white”)
    """
    from PIL import Image, ImageDraw, ImageColor
    import os
    os.makedirs(output_dir, exist_ok=True)

    for img_name, entry in polygons_dict.items():
        # Determine canvas size
        canvas_w, canvas_h = None, None
        if False and entry.get("image") is not None:   # there seems to be a scale discrepancy
            canvas_w, canvas_h = entry["image"].size
        else:
            # fallback to the max extents of all polygons
            xs, ys = [], []
            for p in entry["polygons"]:
                xs += [x for x, y in p["polygon"].vertices()]
                ys += [y for x, y in p["polygon"].vertices()]
            if len(xs) == 0:
                continue
            canvas_w = int(max(xs) + 1)
            canvas_h = int(max(ys) + 1)

        # Create blank canvas
        canvas = Image.new("RGBA", (canvas_w, canvas_h), background_color)
        draw = ImageDraw.Draw(canvas)

        # Draw each polygon
        for piece in entry["polygons"]:
            col_name = piece.get("color", "black")
            try:
                fill_color = ImageColor.getrgb(col_name)
            except ValueError:
                # fallback: lowercase or basic mapping
                fill_color = ImageColor.getrgb(col_name.lower())

            r, g, b, = fill_color
            fill_color = (r, g, b, 128)  # semi-transparent
            coords = [tuple(v) for v in list(piece["polygon"].vertices())]
            draw.polygon(coords, fill=fill_color, outline="black")

        # Save
        out_path = os.path.join(output_dir, img_name.replace(".jpg", ".png"))
        canvas.save(out_path)
        print(f"Saved render → {out_path}")




def save_polygon_renders_matplotlib(
    polygons_dict,
    output_dir,
    background_color="white",
    save_as_svg=True,
    svg_scale_factor=5.0,
    png_dpi=300,
    post_render=False,
    pr_strings=None
):
    """
    Renders polygons to either a high-resolution PNG or a scaled SVG, with
    optional post-render annotations.

    Args:
        polygons_dict (dict): The dictionary of polygons to render.
        output_dir (str): Where to save the renders.
        background_color (str): Canvas background color.
        save_as_svg (bool): If True, saves as SVG. If False, saves as PNG.
        svg_scale_factor (float): Multiplies the default SVG size for convenience.
        png_dpi (int): The DPI to use when saving as a PNG.
        post_render (bool): If True, adds text and vertex labels to the render.
        pr_strings (list): A list of strings to write on the image if post_render is True.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    BASE_DPI = 100 

    for img_name, entry in polygons_dict.items():
        xs, ys = [], []
        # This assumes piece["polygon"] has a .vertices() method returning [(x,y), ...]
        for piece in entry.get("polygons", []):
            for x, y in piece["polygon"].vertices():
                xs.append(x)
                ys.append(y)
        if not xs:
            continue
        canvas_w = int(max(xs) + 1)
        canvas_h = int(max(ys) + 1)

        scale = svg_scale_factor if save_as_svg else 1.0
        fig_w = (canvas_w / BASE_DPI) * scale
        fig_h = (canvas_h / BASE_DPI) * scale
        
        fig, ax = plt.subplots(
            figsize=(fig_w, fig_h),
            dpi=BASE_DPI,
            facecolor=background_color
        )
        ax.set_facecolor(background_color)

        # Draw polygons
        for piece in entry["polygons"]:
            col_name = piece.get("color", "black")
            alpha_val = piece.get("alpha", 0.5)
            rgb = ImageColor.getrgb(col_name.lower())
            rgba_facecolor = (*(c/255 for c in rgb), alpha_val)
            verts = list(piece["polygon"].vertices())
            patch = MplPolygon(
                verts, closed=True, facecolor=rgba_facecolor, edgecolor="none"
            )
            ax.add_patch(patch)

        ax.set_xlim(0, canvas_w)
        ax.set_ylim(canvas_h, 0) # Invert y-axis so (0,0) is top-left
        ax.axis("off")

        # --- NEW: POST-RENDER ANNOTATIONS ---
        if post_render:
            # 1. Write the strings from pr_strings near the top
            if pr_strings and isinstance(pr_strings, list):
                text_to_draw = "\n".join(pr_strings)
                ax.text(
                    canvas_w / 2, 10, text_to_draw,
                    ha='center', va='top', fontsize=10, color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2)
                )

            # 2. Label the vertices of each polygon
            for piece in entry.get("polygons", []):
                polygon_obj = piece["polygon"]
                # This assumes polygon_obj has a .get_vertex_name(vertex) method
                try:
                    for vertex in polygon_obj.vertices():
                        label = polygon_obj.get_vertex_name(vertex)
                        if label: # Only draw if a label exists
                            vx, vy = vertex
                            ax.text(vx + 2, vy, label, fontsize=7, color='blue', ha='left', va='center')
                except AttributeError:
                    print(f"Warning: The provided polygon object does not have the 'get_vertex_name' method. Skipping vertex labeling.")
                    break # Avoid repeating this warning for every vertex

        # Final Save Logic
        if save_as_svg:
            out_name = os.path.splitext(img_name)[0] + ".svg"
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(
                out_path, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0
            )
            print(f"Saved vector render → {out_path}")
        else: # Save as PNG
            out_name = os.path.splitext(img_name)[0] + ".png"
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(
                out_path, dpi=png_dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0
            )
            print(f"Saved PNG render → {out_path}")
            
        plt.close(fig)


def test2(may_10=False, july_9=False, img_dbug = False):
    import os
    global _debug
    polygons = test_get_json_data(may_10_style=may_10,
                                  july_9_style=july_9)        
    save_polygon_renders_matplotlib(polygons, "/tmp/rel_dbug_mpl", background_color="white",
                                    post_render=True, pr_strings=[] )
    # Now get the relations
    for img in polygons:
        print(f"Next image is {img}")
        if False:
            nxt = input("Press enter to continue (or 't <enter>' to toggle debugging) ...")
            if 't' in nxt:
                _debug = not _debug

        annotations = []
        
        for polygon1, polygon2 in itertools.combinations(polygons[img]['polygons'], 2):
            p1, p2 = polygon1['polygon'], polygon2['polygon']
            if _debug:
                print(f"Comparing {polygon1['color']} and {polygon2['color']}")
            relation, cost, ep = compute_relation(p1, p2)
            if ep is not None and len(ep) == 2:
                rs = reln_sexp(relation.name, p1, p2, polygon1['color'], polygon2['color'], ep)
                print(f"Relation between {polygon1['color']} and {polygon2['color']} is {relation} with cost {cost} realized at {ep}, also known as {p1.get_edge_name(ep[0]), p2.get_edge_name(ep[1])}.\n Sexpr:  {rs}.")
                annotations.append(rs)
        print("="*80 + "\n\n\n")
        save_polygon_renders_matplotlib({img: polygons[img]}, "/tmp/rel_dbug_ann",
                                         background_color="white",
                                         post_render=True,
                                         pr_strings = annotations)
            
        if img_dbug:
            output_folder = "/tmp/relation_dbug"
            os.makedirs(output_folder, exist_ok=True)
            draw = ImageDraw.Draw(polygons[img]['image'], mode='RGB')
            for polygon in polygons[img]['polygons']:
                verts2 = [(int(v[0]), int(v[1])) for v in polygon['polygon']._vertices]
                draw.polygon(verts2, fill=(255,255,255), outline=polygon['color'])
            draw.text((0,0), f"{len(polygons[img]['polygons'])} polys", fill='black')
            draw.text((0,20), f"{relation}", fill='black')
            #polygons[img]['image'].show()
            new_path = os.path.join(output_folder, img)
            polygons[img]['image'].save(new_path)

        

if __name__ == "__main__":
    #test1()
    #test2(may_10=True, img_dbug=True)
    test2(july_9=True, img_dbug=True)
