import numpy as np
from numba import guvectorize

from texture.grids import RegularGrid
from texture import grids_mathieu
from texture.links import *

def mine_to_mathieu_grid(grid: RegularGrid):
    return grids_mathieu.RegularGrid(
            offsets=[-s/2 for s in grid.size],
            steps=grid.cell_size,
            nsteps=grid.shape,
        )

#==
# Frame data
#==
def data_in_both_frames(data_1: np.ndarray, data_2: np.ndarray, uids_1: np.ndarray, uids_2: np.ndarray):
    '''
    Given datas from two frames (`data_1` and `data_2`) and its unique indices 
    (`uids_1` and `uids_2`), return the data that is present in both frames in order.

    Parameters
    ----------
    data_1, data_2:
        Array with shape (number of data points, shape of a data point) containing the frame data.
    
    uids_1, uids_2:
        1-D array with datas unique indices, e.g, the unique index of data_1[i] is uids_1[i].

    Return
    ------
    data_intersect_1, data_intersect_2:
        Data points that are present at both frames in order, i.e, the data point `data_intersect_1[i]`
        has the same unique index as `data_intersect_2[i]`.
    '''
    _, indices_1, indices_2 = np.intersect1d(uids_1, uids_2, return_indices=True)
    return data_1[indices_1], data_2[indices_2]


#==
# Post process
#==
def grid_data_mean(data_sum: np.ndarray, count: np.ndarray, remove_zero_entries=True):
    '''
    Given the sum of some data in a grid (`data`) and the amount of data points in each grid
    element (`count`), returns the mean per grid element.
    
    Parameters
    ----------
    data_sum:
        Array with shape (N_r, N_c, D), where
        
        N_r: Number of rows in the grid 
        N_c: Number of columns in the grid 
        D: Shape of a data point
    
    count:
        The count of data points per grid element. Array of integers with shape (N_r, N_c).

    remove_zero_entries:
        If True, set all zero entries in count to one.

    Return
    ------
    data_mean:
        Mean per grid element. 
    '''
    # count_non_zero = np.maximum(1, count)
    # return data_sum / count_non_zero[..., None]
    if remove_zero_entries:
        count_non_zero = np.maximum(1, count)
    else:
        count_non_zero = count

    n = len(data_sum.shape[2:])
    count_non_zero_expanded = count_non_zero
    for _ in range(n):
        count_non_zero_expanded = count_non_zero_expanded[..., None]
    return data_sum / count_non_zero_expanded


#==
# Utils
#==
def B_from_C(C: np.ndarray):
    "Convert from a C (or sum_C) matrix to a B (resp sum_B) matrix (upper triangle)"
    B = (C + np.swapaxes(C, -1, -2))
    i,j = np.triu_indices(C.shape[-1])
    return B[...,i,j]

def square_from_triangular(M: np.ndarray):
    "Convert a triangular representation to square symmetric"
    #dimension of space from the number of upper triangular coefficients
    d = int((np.sqrt(1 + 8 * M.shape[-1]) - 1)/2)
    m = np.zeros(M.shape[:-1]+(d,d), dtype=M.dtype)
    i,j = np.triu_indices(d)
    #fill upper triangle and diagonal
    m[...,i,j] = M
    #fill lower triangle and diagonal
    m[...,j,i] = M
    return m

# @guvectorize(['(float32[:,:], float32[:,:], float32[:,:])', '(float64[:,:], float64[:,:], float64[:,:])'], '(n,n),(n,n)->(n,n)', nopython=True, target='parallel')
# def leastsq(A,B, res):
#     """Compute element by element the least square solution to Ax=B"""
#     res[:] = np.linalg.lstsq(A,B, rcond=1e-3)[0]
def leastsq(A,B):
    """Compute element by element the least square solution to Ax=B"""
    return np.linalg.lstsq(A,B, rcond=1e-3)[0]

def bin_count(points: np.ndarray, links_ids: np.ndarray, grid: RegularGrid, points_per_link=3):
    '''
    Count how many links there is in each grid element.
    '''
    link_start = points[links_ids[:,0]]
    link_end = points[links_ids[:,1]]

    t = parameter_range_across_link(points_per_link)[:, None, None]
    texture_points = (1-t)*link_start + t*link_end

    coords = grid.coords(texture_points)

    return grid.count(
        coords,
        remove_out_of_bounds=True,
    ).sum(axis=0)


#==
# Discrete Calculators
#==
def parameter_range_across_link(points_per_bond, one_point_pos=0.5):
    if points_per_bond > 1:
        t = np.linspace(0, 1, points_per_bond)
    else:
        t = np.array([one_point_pos])
    return t

def bin_texture_sum(points: np.ndarray, links_ids: np.ndarray, grid: RegularGrid, points_per_link=3):
    '''
    Bin texture tensor on a grid.
    
    Parameters
    ----------
    points: 
        Array of coordinates with shape (P,D) where:
        
        P: Number of points.
        D: Number of spacial dimension. 
    
    links_ids: 
        Array of indices defining links with shape (B,2) where:

        B: Number of links.
    
        OBS: The link vector is:

        >>> id1, id2 = links[i]      
        >>> link_vector = points[id2] - points[id1] 

    grid: 
        A D-dimensional Grid instance that performs the binning. 
        
    points_per_link:
        The number of points per link: 
        
        1: only the link middle.
        2: only the ends.
        3: ends and middle.
        n: n points evenly spaced along the link.
    
    Returns
    -------
    texture_sum: 
        Sum of the texture matrices (in the triangular form) on each grid element.
        For two dimensional data, it's shape is (*grid.shape, 3).

        >>> texture_sum[i, j] = Texture binned to the grid element at i-th row and j-th column.

        remembering that the grid element at the bottom left corner corresponds to the index (0, 0).
    
        To reconstruct the matrix, one needs to do the fallowing:

        texture_matrix = | texture_sum[i, j][0]  texture_sum[i, j][1] |   
                         | texture_sum[i, j][1]  texture_sum[i, j][2] | 

    count: 
        Number of matrices binned in each grid element. Each points_per_link counts for 1.
        It's shape is the same as the grid.
    '''
    if points.shape[1] != grid.num_dims:
        raise ValueError(f"Points dimension ({points.shape[1]}) and grid dimension ({grid.num_dims}) don't match!")
    if points_per_link < 1:
        raise ValueError(f"points_per_bond should be > 1, but it is {points_per_link}")
    if not isinstance(points_per_link, int):
        raise TypeError(f"points_per_bond should be an integer, but it is {type(points_per_link).__name__}")

    link_start = points[links_ids[:,0]]
    link_end = points[links_ids[:,1]]
    links = link_end - link_start
    
    # link matrix (symmetric, but compute everything, that is a ratio of (D-1)/(2*D) too many coefficients: 1/4, 1/3, etc)
    textures = links[:,None,:] * links[:,:, None]
    
    # since m is symmetric keep only the upper triangle
    tri_inds = np.triu_indices(points.shape[1])
    textures = textures[:, tri_inds[0], tri_inds[1]]
    
    # Mathieu
    # grid_m = grids_mathieu.RegularGrid(
    #     offsets=[-s/2 for s in grid.size],
    #     steps=grid.cell_size,
    #     nsteps=grid.shape,
    # )
    # sumw = np.zeros(grid_m.shape+(link_matrix.shape[1],))
    # count_m = np.zeros(grid_m.shape, np.int64)
    # for x in np.linspace(0,1,points_per_bond):
    #     s, c = grid_m.count_sum_discreet((1-x)*link_start + x*link_end, link_matrix)
    #     sumw += s
    #     count_m += c
    
    t = parameter_range_across_link(points_per_link)[:, None, None]

    texture_points = (1-t)*link_start + t*link_end

    coords = grid.coords(texture_points)

    sum_textures = grid.sum_by_cell(
        values=np.array([textures] * points_per_link),
        coords=coords,
        zero_value=np.zeros_like(textures, shape=textures.shape[1:]),
        remove_out_of_bounds=True,
    ).sum(axis=0)
    
    count = grid.count(
        coords,
        remove_out_of_bounds=True,
    ).sum(axis=0)
    
    return sum_textures, count

def bin_geometrical_changes_sum(points_0: np.ndarray, points_1: np.ndarray, links_ids: np.ndarray, dt: float,
    grid: RegularGrid, points_per_link=3):
    """
    Bin geometrical changes of the texture tensor between two times on a grid. 
    It is based on links which exist at both times.
    
    Parameters
    ----------
    points_0, points_1: 
        (P,D) arrays of coordinates
    
    links_ids: 
        (B,2) array of indices defining bounded pairs of particles at both frames.
    
    dt:
        Time interval between frames 1 and 0.
    
    grid: 
        D-dimensional Grid instance that performs the binning
    
    points_per_link:
        The number of points per link: 
        
        1: only the link middle.
        2: only the ends.
        3: ends and middle.
        n: n points evenly spaced along the link.
    
    Returns
    -------
    sum_C: 
        Sum of the C matrices on each grid element. From C, one obtains `B = C + transpose(C)`, 
        there is function for that, check `B_from_C()`. Provided the texture M, we can also obtain V and $\Omega$.
    
    count: 
        Number of matrices binned in each grid element. Each link points that remains on the same grid element 
        between both frames counts for 1. Its shape is the same as the grid.
        
        Caution: Intensive matrix C is obtained by dividing sumC of the present function by the 
        count of `bin_texture_sum()` (averaged between both frames) using the same `points_per_link` parameter.
    """
    if points_0.shape[1] != grid.num_dims:
        raise ValueError(f"points_0 dimension ({points_0.shape[1]}) and grid dimension ({grid.num_dims}) don't match!")
    if points_1.shape[1] != grid.num_dims:
        raise ValueError(f"points_1 dimension ({points_1.shape[1]}) and grid dimension ({grid.num_dims}) don't match!")
    if points_0.shape[0] != points_1.shape[0]:
        raise ValueError(f"Number of points in points_0 ({points_0.shape[0]}) and points_1 ({points_1.shape[0]}) don't match!")

    if points_per_link < 1:
        raise ValueError(f"points_per_bond should be > 1, but it is {points_per_link}")
    if not isinstance(points_per_link, int):
        raise TypeError(f"points_per_bond should be an integer, but it is {type(points_per_link).__name__}")
    
    link_start_0 = points_0[links_ids[:,0]]
    link_end_0 = points_0[links_ids[:,1]]
    links_0 = link_end_0 - link_start_0
    
    link_start_1 = points_1[links_ids[:,0]]
    link_end_1 = points_1[links_ids[:,1]]
    links_1 = link_end_1 - link_start_1
    
    # Average link and difference
    links = (links_0 + links_1) * 0.5
    delta_links = (links_1 - links_0) / dt
    
    # Extensive version of the asymmetric tensor $C = \ell \otimes \Delta\ell$ see equation C.7
    C = links[:, None, :] * delta_links[:, :, None]
    

    # Mathieu
    # since C is not symmetric we have to keep all coefficients
    # bin on each end of each bond and on the middle point
    # only points that stay in the same bin will be counted
    # grid_m = my_to_mathieu_grid(grid)
    # sumw = np.zeros(grid.shape + C.shape[1:])
    # count_m = np.zeros(grid.shape, np.int64)
    # for x in np.linspace(0, 1, points_per_bond):
    #     su, co = grid_m.count_sum_discreet((1-x)*link_start_0 + x*link_end_0, C, (1-x)*link_start_1 + x*link_end_1)
    #     sumw += su
    #     count_m += co
    

    t = parameter_range_across_link(points_per_link)[:, None, None]
    texture_points_0 = (1-t)*link_start_0 + t*link_end_0
    texture_points_1 = (1-t)*link_start_1 + t*link_end_1

    coords_0 = grid.coords(texture_points_0)
    coords_1 = grid.coords(texture_points_1)

    sum_C = np.zeros_like(C, shape=grid.shape + C.shape[1:])
    count = np.zeros(grid.shape, int)

    for idx in range(points_per_link):
        c0, c1 = coords_0[idx], coords_1[idx]
        mask = (c0 == c1).all(axis=1)
        coords = c0[mask]

        sum_C += grid.sum_by_cell(
            values=C[mask],
            coords=coords,
            zero_value=np.zeros_like(C[0]),
            remove_out_of_bounds=True,
            simplify_shape=True,
        )
        
        count += grid.count(
            coords,
            remove_out_of_bounds=True,
            simplify_shape=True,
        )
    
    return sum_C, count

def bin_topological_changes_sum(
    points_0: np.ndarray, points_1: np.ndarray, 
    links_ids_0: np.ndarray, links_ids_1: np.ndarray, 
    dt:float, grid: RegularGrid, points_per_link=3):
    '''
    Bin on a grid topological changes of the texture tensor between two frames. 
    It is based on links which appeared or disappeared between both frames.

    Parameters
    ----------
    points_0, points_1: 
        (P, D) arrays of coordinates for particles that exist at both frames ordered, i.e,
        `points_0[i]` and `points_1[i]` refers to the same point at different frames.
    
    links_ids_0: 
        (B0, 2) array of indices defining links in the first frame.
    
    links_ids_1: 
        (B1, 2) array of indices defining links in the second frame.

    dt:
        Time interval between frames 1 and 0.

    grid: 
        Grid instance that performs the binning.
    
    points_per_link:
        The number of points per link: 
        
        1: only the link middle.
        2: only the ends.
        3: ends and middle.
        n: n points evenly spaced along the link.
        
    Returns
    -------
    sum_T: 
        Sum of the T matrices on each grid element. 
        
        Caution: Intensive matrix T is obtained by dividing sum_T of the present function 
        by the count of `bin_texture_sum()` (averaged between frames).
    
    count_a: 
        Number of appearing links binned in each grid element. Each `points_per_link` counts for 1.
    
    count_d: 
        Number of disappearing links binned in each grid element. Each `points_per_link` counts for 1.
    '''
    if points_0.shape[1] != grid.num_dims:
        raise ValueError(f"points_0 dimension ({points_0.shape[1]}) and grid dimension ({grid.num_dims}) don't match!")
    if points_1.shape[1] != grid.num_dims:
        raise ValueError(f"points_1 dimension ({points_1.shape[1]}) and grid dimension ({grid.num_dims}) don't match!")
    if points_0.shape[0] != points_1.shape[0]:
        raise ValueError(f"Number of points in points_0 ({points_0.shape[0]}) and points_1 ({points_1.shape[0]}) don't match!")

    if points_per_link < 1:
        raise ValueError(f"points_per_bond should be > 1, but it is {points_per_link}")
    if not isinstance(points_per_link, int):
        raise TypeError(f"points_per_bond should be an integer, but it is {type(points_per_link).__name__}")

    # Links that appeared and disappeared between frames
    link_ids_a, link_ids_d, _ = links_appeared_disappeared(links_ids_0, links_ids_1)
    
    # Bin the texture of links that appeared
    sum_texture_a, count_a = bin_texture_sum(points_1, link_ids_a, grid, points_per_link)
    
    #bin the texture of links that disappeared
    sum_texture_d, count_d = bin_texture_sum(points_0, link_ids_d, grid, points_per_link)
    
    return (sum_texture_a - sum_texture_d)/dt, count_a, count_d
    
def bin_changes(pos0, pos1, pairs0, pairs1, grid, points_per_bond=3):
    """bin on a grid geometrical and topological changes of the texture tensor between two times.
    Caution: intensive matrices C and T are obtained by dividing sumC and sumT of the present function by the count of bin_texture (averaged between t0 and t1).
    
    Parameters
    ----------
    pos0, pos1: (P,D) arrays of coordinates for particles that exist at both times
    pairs0: (B0,2) array of indices defining bounded pairs of particles at t0
    pairs1: (B1,2) array of indices defining bounded pairs of particles at t1
    grid: a D-dimentional Grid instance that performs the binning
    points_per_bond: int>1 The number of points per bond: 2 only the ends, 3 ends and middle, etc.
    
    Returns
    ----------
    sumC: the sum of the C matrices on each grid element.
    countc: the number of matrices binned in each grid element. Each end of a bond that remains on the same grid element between t0 and t1 counts for 1. The middle of a bond also counts for 1 if it emains on the same grid element between t0 and t1.
    sumT: the sum of the T matrices on each grid element.
    counta: the number of appearing matrices binned in each grid element. Each end of an appearing bond, as well as intermediate points, at t1 counts for 1.
    countd: the number of disappearing matrices binned in each grid element. Each end of a disappearing bond, as well as intermediate points, at t0 counts for 1.
    """
    assert pos0.shape[1] == grid.ndim
    assert pos0.shape[0] == pos1.shape[0]
    assert pos0.shape[1] == pos1.shape[1]
    #bonds that appeared, disappeared, or were conserved between t0 and t1
    pairsa, pairsd, pairsc = links_appeared_disappeared(pairs0, pairs1)
    #bin the texture of bonds that appeared
    sumwa, counta = bin_texture_sum(pos1, pairsa, grid, points_per_bond)
    #bin the texture of bonds that disappeared
    sumwd, countd = bin_texture_sum(pos0, pairsd, grid, points_per_bond)
    #bin the geometrical changes of the conserved bonds
    sumC, countc = bin_geometrical_changes_sum(pos0, pos1, pairsc, grid, points_per_bond)
    return sumC, countc, sumwa-sumwd, counta, countd


#==
# Continuos Calculators
#==
def symmetrized_velocity_gradient(M, C, inv_M=None, triangular=True):
    '''
    Computes statistical symmetrized velocity gradient V from the texture matrix M and matrix C. 
    Keeps only the symmetric coefficients if `triangular=True`.
    
    OBS: M and C can also be given in a grid, where its shape is (grid shape, M/C shape).

    Parameters
    ----------
    M:
        Texture in the triangular form.
    
    C:
        C matrix
    
    inv_M:
        Inverse of M. If None, it is calculated.

    triangular:
        If True, the return value will be in triangular form.
    '''
    M = square_from_triangular(M)
    
    # Set texture to unity matrix where its determinant is zero (impossible to inverse)
    # Since M was symmetric, it corresponds to null matrix, thus probably grid elements with no bond inside.
    M[np.linalg.det(M)==0] = np.eye(M.shape[-1])
    
    if inv_M is None:
        inv_M = np.linalg.inv(M)
    
    V = (np.matmul(inv_M, C) + np.matmul(np.swapaxes(C, -1, -2), inv_M)) / 2
    
    if triangular:
        # This should be symmetric within numerical errors, so we keep only the upper triangle
        i,j = np.triu_indices(V.shape[-1])
        V = V[..., i, j]
    
    return V

def statistical_rotation_rate(M: np.ndarray, C: np.ndarray, inv_M=None, triangular=True):
    '''
    Statistical rotation rate Omega from the texture matrix M and matrix C. 
    Keeps only the asymmetric coefficients if `triangular=True`.
    '''
    M = square_from_triangular(M)
    
    # Set texture to unity matrix where its determinant is zero (impossible to inverse)
    # Since M was symmetric, it corresponds to null matrix, thus probably grid elements with no bond inside.
    M[np.linalg.det(M)==0] = np.eye(M.shape[-1])
    
    if inv_M is None:
        inv_M = np.linalg.inv(M)
    
    omega = (np.matmul(inv_M, C) - np.matmul(np.swapaxes(C, -1, -2), inv_M)) / 2
    
    if triangular:
        # This should be antisymmetric within numerical errors, so we keep only the upper triangle, diagonal excluded
        i,j = np.triu_indices(omega.shape[-1], 1)
        if len(i) == 1:
            omega = omega[...,i[0],j[0]]
        else:
            omega = omega[...,i,j]
    
    return omega

def statistical_topological_rearrangement_rate(M: np.ndarray, T: np.ndarray, inv_M=None, triangular=True):
    '''
    Statistical topological rearrangement rate P from the texture matrix M and topological change matrix T. 
    Keeps only the symmetric coefficients if `triangular=True`.
    '''
    M = square_from_triangular(M)
    T = square_from_triangular(T)
    
    # Set texture to unity matrix where its determinant is zero (impossible to inverse)
    # Since M was symetric, it corresponds to null matrix, thus probably grid elements with no bond inside.
    M[np.linalg.det(M)==0] = np.eye(M.shape[-1])
    
    if inv_M is None:
        inv_M = np.linalg.inv(M)
    
    P = - (np.matmul(inv_M, T) + np.matmul(T, inv_M)) / 4
    
    if triangular:
        # This should be symetric within numerical errors, so we keep only the upper triangle
        i,j = np.triu_indices(P.shape[-1])
        P = P[...,i,j]
    
    return P

def statistical_relative_deformations(M,C,T):
    """Computes the statistical velocity gradient V,  the statistical rotation rate Omega and the statistical topological rearrangement rate P from the texture matrix M, matrix C and topological change matrix T. Keeps only independant coefficients."""
    W = leastsq(square_from_triangular(M), C)
    v = (W + np.swapaxes(W, -1, -2)) / 2
    omega = (W - np.swapaxes(W, -1, -2)) / 2
    p = leastsq(square_from_triangular(M), square_from_triangular(T))
    p = (-p + np.swapaxes(p, -1,-2))/4
    #V and T should be symmetric within numerical errors, so we keep only the upper triangle
    i,j = np.triu_indices(v.shape[-1])
    V = v[...,i,j]
    P = p[...,i,j]
    #Omega should be antisymmetric within numerical errors, so we keep only the upper triangle, diagonal excluded
    i,j = np.triu_indices(omega.shape[-1], 1)
    if len(i) == 1:
        Omega = omega[...,i[0],j[0]]
    else:
        Omega = omega[...,i,j]
    return V, Omega, P


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    import texture
    from texture import grids
    import numpy as np
    
    from texture import display

    np.random.seed(43)


    num_points = 30
    grid_size = 10
    num_add = 4
    num_remove = 3 
    dl = 1

    grid = grids.RegularGrid(
        length=grid_size, height=grid_size,
        num_cols=5, num_rows=5,
    )

    # num_cols, num_rows = 5, 6
    # grid = grids.RetangularGrid((
    #     grid_size * np.linspace(0, 1, num_cols + 1)**0.5 - grid_size/2,
    #     grid_size * np.linspace(0, 1, num_rows + 1)**1.5 - grid_size/2,
    # ))


    points_1 = (np.random.random((num_points, grid.num_dims)) - 1/2) * (grid_size - 2*dl*0)
    links_ids_1 = links_from_voronoi(points_1, grid_size/3)
    ids_1 = np.arange(num_points)

    theta = np.random.random(num_points) * 2 * np.pi

    points_2  = points_1 + np.array([np.cos(theta), np.sin(theta)]).T * dl
    last_idx = ids_1.max()
    
    # Remove n random points to points_2
    remove_indices = np.random.choice(np.arange(num_points), num_remove, replace=False)
    points_2 = np.delete(points_2, remove_indices, axis=0)
    ids_2 = np.delete(ids_1, remove_indices)

    # Add n random points to points_2
    new_points = (np.random.random((num_add, grid.num_dims)) - 1/2) * grid_size
    points_2 = np.vstack([points_2, new_points])
    new_ids = np.arange(last_idx + 1, last_idx + 1 + num_add)
    ids_2 = np.hstack([ids_2, new_ids])
    
    # points_2 = (np.random.random((num_points, grid.num_dims)) - 1/2) * (grid_size - 2*dl)
    
    links_ids_2 = links_from_voronoi(points_2, grid_size/3)

    sum_M, count_M = texture.bin_texture_sum(points_1, links_ids_1, grid, points_per_link=1)

    points_inter_1, points_inter_2 = data_in_both_frames(points_1, points_2, ids_1, ids_2)
    links_inter_1, links_inter_2 = links_from_voronoi(points_inter_1, grid_size/3), links_from_voronoi(points_inter_2, grid_size/3)
    links_inter = texture.links_intersect_same_points(links_inter_1, links_inter_2)
    sum_C, count_C = texture.bin_geometrical_changes_sum(
        points_inter_1, points_inter_2, links_inter, 0.01, grid
    )

    V = symmetrized_velocity_gradient(sum_M, sum_C)
    # V = symmetrized_velocity_gradient(grid_data_mean(sum_M, count_M), grid_data_mean(sum_C, count_M))

    points_inter_1, points_inter_2 = data_in_both_frames(points_1, points_2, ids_1, ids_2)
    links_inter_1, links_inter_2 = links_from_voronoi(points_inter_1, grid_size/3), links_from_voronoi(points_inter_2, grid_size/3)
    sum_T, count_a, count_d = texture.bin_topological_changes_sum(
        points_inter_1, points_inter_2, links_inter_1, links_inter_2, 0.01, grid 
    )

    # for link in links_ids_1:
    # for link in links_inter:
    #     plt.plot(*points_1[link].T, color='black')

    # grid.plot_grid(plt.gca())
    # plt.scatter(*points_1.T, c='b')
    # plt.scatter(*points_2.T, c='r')
    # plt.show()

    ax = plt.gca()
    grid.plot_grid(ax)

    # display.draw_count_2D(ax, grid, count_M)
    # display.draw_points_links(ax, points_1, links_ids_1)
    display.draw_matrices(ax, grid, grid_data_mean(sum_M, count_M), adjust_lims=False)
    # display.display_matrices(ax, grid, V, adjust_lims=False)

    # plt.scatter(*(points_1[links_ids_1].sum(axis=1) / 2).T)


    # plt.savefig("im.png")
    plt.show()
