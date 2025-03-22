from enum import Enum, auto
from scipy.spatial import Voronoi
import numpy as np
from abc import ABC, abstractmethod

def links_from_voronoi(points: np.ndarray, max_dist: float=None):
    '''
    Generate links IDs from the Voronoi diagram.
    
    Parameters
    ----------
    points:
        Array with the positions. Its shape should be (N, D) 
        where N is the number of points and D the dimension of points.
        
        Example:
        >>> pos[i, 0] -> x position of the i-th point  
        >>> pos[i, 1] -> y position of the i-th point
    
    max_dist:
        Maximum length that a link can have. If `None`, there is no limit
        to the link's length.

    Returns
    -------
    links_ids:
        IDs of the points that form a link. Its shape is (M, 2), where M is the 
        number of links. Each link is a connection between two
        points in `points`.

        >>> links_ids[i, 0] -> ID of the point in `points` that starts the i-th link 
        >>> links_ids[i, 1] -> ID of the point in `points` that ends the i-th link 

        Therefore, the vector representing the i-th link is

        >>> id1, id2 = links_ids[i]
        >>> link_i = pos[id2] - pos[id1]
    '''
    vor = Voronoi(points)
    points_adj = vor.ridge_points
    edges = np.sort(points_adj, axis=-1)
    links_ids = np.array(sorted((a,b) for a,b in edges.astype(int)))
    
    if max_dist is not None:
        edge_dist = points[links_ids[:,0]] - points[links_ids[:,1]]
        edge_dist = np.sqrt(np.square(edge_dist).sum(axis=1))
        
        mask = edge_dist < max_dist
        links_ids = links_ids[mask]

    return links_ids

def links_set_to_array(s: set):
    '''
    Convert a set of links to a NumPy array.

    Parameters
    ----------
    s: 
        A set of tuples, where each tuple contains two elements representing a link.

    Returns
    -------
    ndarray: 
        A 2D-array with shape (n, 2) where n is the number of links in the set.
        If the set is empty, returns an empty array with shape (0, 2).
    '''
    if len(s)==0:
        return np.zeros((0,2), dtype=np.int64)
    return np.array([[a,b] for a,b in sorted(s)], dtype=np.int64)

def links_appeared_disappeared(link_ids_0, link_ids_1):
    '''
    Compute the links that appeared, disappeared and stay conserved, 
    between two sets of links.

    OBS: It is assumed that the two set of links came from the same
    set of points, i.e, two links are the same if they have the same ID's.
    '''
    s0 = set((a,b) for a,b in np.sort(link_ids_0, axis=-1))
    s1 = set((a,b) for a,b in np.sort(link_ids_1, axis=-1))
    pairsa = links_set_to_array(s1-s0)
    pairsd = links_set_to_array(s0-s1)
    pairsc = links_set_to_array(s0&s1)
    return pairsa, pairsd, pairsc

def generate_link_uids(link_ids: np.ndarray, points_uids: np.ndarray) -> np.ndarray:
    """
    Generate unique IDs for links based on the unique IDs of the points that compose the links.

    Parameters
    ----------
    link_ids: np.ndarray
        Array of indices defining links with shape (B,2), where B is the number of links.
    
    point_uids: np.ndarray
        Array of unique identifiers for the points with shape (P,), where P is the number of points.

    Returns
    -------
    np.ndarray
        Array of unique identifiers for the links with shape (B,).
    """
    # Extract the unique IDs for the points in each link
    link_uids = np.sort(points_uids[link_ids], axis=1)
    
    # Combine the sorted unique IDs into a single unique identifier for each link
    link_ids = np.array([(uid1, uid2) for uid1, uid2 in link_uids])
    
    return link_ids

def links_intersect(links_ids_1: np.ndarray, links_ids_2: np.ndarray, uids_1: np.ndarray, uids_2: np.ndarray):
    '''
    Given links in two frames, and the respective array with unique positions ID's,
    return the links that are present at both frames.

    Parameters
    ----------
    links_ids_1, links_ids_2:
        Arrays with points indices that form a link. Its shape is (M, 2),
        where M is the number os links. They do not need to have the same M.

    uids_1, uids_2:
        1-D array with positions unique indices. For instance, the unique ID's for the
        ith link in the first frame is 
        
        >>> id1, id2 = link_ids_1[i]
        >>> uids_1[id1] # Unique ID for the first link point
        >>> uids_1[id2] # Unique ID for the last link point

    Returns
    -------
    links_intersect_1, links_intersect_2:
        Links that are present at both frames.
    '''
    links_1_uids = generate_link_uids(links_ids_1, uids_1)
    links_2_uids = generate_link_uids(links_ids_2, uids_2)
    _, indices, _ = np.intersect1d(links_1_uids, links_2_uids, return_indices=True)
    return links_ids_1[indices]

def links_intersect_same_points(links_ids_1: np.ndarray, links_ids_2: np.ndarray):
    '''
    Given links in two frames, assuming they were created from the same points at both frames,
    i.e., two links are the same if they have the same ID's, returns the links that are present at both frames.
    '''
    links_sorted_1 = np.sort(links_ids_1, axis=1)
    links_sorted_2 = np.sort(links_ids_2, axis=1)

    ids_1 = np.empty(links_ids_1.shape[0], dtype=object)
    ids_2 = np.empty(links_ids_2.shape[0], dtype=object)

    for i, (id1, id2) in enumerate(links_sorted_1):
        ids_1[i] = (id1, id2)
    
    for i, (id1, id2) in enumerate(links_sorted_2):
        ids_2[i] = (id1, id2)

    _, indices, _ = np.intersect1d(ids_1, ids_2, return_indices=True)
    return links_ids_1[indices]


class LinkCfg(ABC):
    @abstractmethod
    def link_func(self, points: np.ndarray) -> np.ndarray:
        pass

class VoronoiLink(LinkCfg):
    def __init__(self, max_dist: float=None):
        self.max_dist = max_dist

    def link_func(self, points):
        return links_from_voronoi(points, self.max_dist)
