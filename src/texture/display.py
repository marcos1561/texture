import numpy as np
import grids
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import EllipseCollection, LineCollection

from texture import square_from_triangular, errors

def draw_links(ax: Axes, points, links, **kwargs):
    kwargs_default = {'color': 'r'}
    kwargs = {**kwargs_default, **kwargs}

    lc = LineCollection(points[links], **kwargs)
    ax.add_collection(lc)

    return lc

def draw_points(ax: Axes, points, **kwargs):
    kwargs_default = {'marker': 'o', 'c': 'k'}
    kwargs = {**kwargs_default, **kwargs}
    
    return ax.scatter(*points.T, **kwargs)

def draw_points_links(ax: Axes, points, links, points_kw=None, links_kw=None):
    if points_kw is None:
        points_kw = {}
    if links_kw is None:
        links_kw = {}

    l = draw_links(ax, points, links, **links_kw)
    
    if "zorder" not in points_kw.keys():
        points_kw["zorder"] = l.get_zorder() + 1
    p = draw_points(ax, points, **points_kw)

    return l, p

def draw_count_2D(ax: Axes, grid: grids.RetangularGrid, count, colormesh_kw=None, colorbar_kw=None):
    "Draw count for a 2D grid as color map."
    if colormesh_kw is None:
        colormesh_kw = {}
    if colorbar_kw is None:
        colorbar_kw = {}
    
    p = ax.pcolormesh(*grid.edges, count, shading="flat", **colormesh_kw)
    cbar = plt.colorbar(p, ax=ax, **colorbar_kw)
    return p, cbar

def display_scalar(ax, grid, scalar, kw_scatter = {'marker': 'o'}):
    """Display on a matplotlib axis a scalar field defined for each element of a grid. Valid for any type of grid"""
    XY = grid.mesh()
    if scalar.shape == grid.shape:
        sc = scalar[grid.mask()]
    else:
        sc = scalar
    return ax.scatter(XY[:,0], XY[:,1], c = sc, **kw_scatter)
    
def draw_matrices(ax: Axes, grid: grids.RetangularGrid, matrix: np.ndarray, 
    scale: float=None, col=None, adjust_lims=True, 
    ellipse_kwargs=None, line_kwargs=None):
    '''
    Draw ellipses representing the symmetric matrix data in `matrix` at each grid element of `grid`. 
    
    * Each axis of the ellipse corresponds to an eigenvalue and is oriented along its eigenvector. 
    * Only axis corresponding to positive eigenvalues are drawn.

    Examples:

    * A 'coffee bean' has a negative eigenvalue smaller in absolute value than its positive eigenvalue. 
    * A 'capsule' has a negative eigenvalue larger in absolute value than its positive eigenvalue. 
    * A circle is when the two eigenvalues are equal in absolute value.
    
    Parameters
    ----------
    ax:
        Axes where things will be drawn
    
    grid:
        Grid where matrix data was calculated.

    matrix:
        Matrix data (in triangular form) on a grid. Its shape is (L, C, 3), where
        L is the number of lines in the grid and C the number of columns.

    scale:
        Scale of lengths, if a line has length l, then it will be drawn with length l*scale. 
        If None, an automatic scale will be calculated, such that the characteristic ellipse 
        length at 90 percentile is the same as the grid element characteristic length.
        If a shape has area A, then its characteristic length is sqrt(A).

    col:
        Color used to draw ellipses and axis. If none, ellipses will be colored according
        to the trace of the respective matrix and the axis will be gray.

    adjust_lims:
        If True, adjust axis limits to grid limits.

    Returns
    -------
    ellipse_col, line_col:
        Ellipses and line matplotlib collections.
    '''
    if ellipse_kwargs is None:
        ellipse_kwargs = {}
    if line_kwargs is None:
        line_kwargs = {}

    mask = np.ones(grid.shape_mpl, bool)

    # Convert the texture back to an array of matrices
    matrix = square_from_triangular(matrix)
    
    # Compute matrix eigenvalues and eigenvectors for each grid element
    evalues, evectors = np.linalg.eigh(matrix)
    
    # Width and height are the larger and smaller eigenvalues respectively
    width = evalues[..., 1]
    height = evalues[..., 0]
    
    very_big_values_mask = np.full_like(width, fill_value=False)
    for evalue_i in [width, height]:
        evalue_abs = np.abs(evalue_i)
        evalue_abs_90p = evalue_abs[evalue_abs < np.percentile(evalue_abs, 90)]
        very_big_values_mask = np.logical_or(very_big_values_mask, evalue_abs > (evalue_abs_90p.mean() + 20 * evalue_abs_90p.std()))

    mask = mask & (~very_big_values_mask)

    width = width[mask] 
    height = height[mask] 
    # print("big values count:", very_big_values_mask.sum())

    # Angle is given by the angle of the larger eigenvector
    angle = np.rad2deg(np.arctan2(evectors[..., 1, 1], evectors[..., 0, 1]))[mask]
    
    # Sum of the eigenvalues (trace of the matrix)
    trace = width + height #np.where(np.abs(ww)>np.abs(hh), ww, hh)#ww*hh

    cell_centers = np.column_stack((
        grid.meshgrid[0][mask], grid.meshgrid[1][mask]
    ))

    # Ellipses and lines scale
    if scale is None: 
        ellipse_areas = np.pi * np.abs(np.prod(evalues, axis=-1))
        relative_sqrt_area = np.sqrt((ellipse_areas / grid.cell_area)[mask])
        inv_scale = np.percentile(relative_sqrt_area, 90)
        
        if inv_scale == 0:
            inv_scale = np.max(relative_sqrt_area)
            if inv_scale == 0:
                raise errors.AllMatricesNullError()
        
        scale = 1/inv_scale
        
    # Show ellipses
    ellipse_col = EllipseCollection(
        (np.abs(width)*scale), (np.abs(height)*scale), angle, 
        units='xy', offsets=cell_centers,
        transOffset=ax.transData, 
        edgecolors=col, facecolors='none',
        **ellipse_kwargs,
    )

    # Major and minor axes (only for positive eigenvalues)
    segments = 0.5 * scale * np.transpose(evectors*np.maximum(0, evalues)[..., None, :], (0,1,3,2))[mask]
    segments = np.repeat(segments, 2, axis=1).reshape(-1, 2, 2)
    segments[:, 1, :] *= -1
    
    # xyps = scale * np.transpose(evectors*np.maximum(0, evalues)[..., None, :], (0,1,3,2))[total_mask].reshape(2*len(ww),2)*0.5
    # segments = np.array([[-xyp, xyp] for xyp in xyps])

    offsets = np.repeat(cell_centers, 2, axis=0)
    segments += offsets[:, None, :]
    line_col = LineCollection(
        segments,
        color=(0.5,0.5,0.5),
        **line_kwargs,
    )

    if col is None:
        if np.ptp(trace) > 0:
            ellipse_col.set_array(trace)
            line_col.set_array(np.repeat(trace, 2))
    else:
        ellipse_col.set_edgecolor(col)
        line_col.set_edgecolor(col)

    if adjust_lims:
        ax.set_xlim(*grid.dim_extremes[0])
        ax.set_ylim(*grid.dim_extremes[1])

    ax.add_collection(ellipse_col)
    ax.add_collection(line_col)
    return ellipse_col, line_col

def draw_polar_grid(ax,grid, color="b"):
    """Show the limits for the polar grid cells"""
    for r in grid.radii:
        ax.add_artist(plt.Circle((0,0), r, fc='none', ec=color))
    for rs, nc, ot in zip(np.column_stack((grid.radii[:-1], grid.radii[1:])), grid.ncells, grid.theta_offset):
        if nc==1:continue
        for theta in 2*np.pi*np.arange(nc)/nc + ot:
            ax.add_artist(plt.Line2D(np.cos(theta)*rs, np.sin(theta)*rs, c=color))
            
def fill_polar_grid(ax, grid, scalar, definition=45, **kwarg):
    """Display on a matplotlib axis a scalar field defined for each element of a polar grid."""
    if scalar.shape == grid.shape:
        data = [s[:nc] for s, nc in zip(scalar, grid.ncells)]
    else:
        cnc = np.concatenate([0], np.cumsum(grid.ncells))
        data = [scalar[i:j] for i,j in zip(cnc[:-1], cnc[1:])]
    dmin = min(min(d) for d in data)
    dmax = max(max(d) for d in data)
    norm = plt.Normalize(vmin=dmin, vmax=dmax)
    for i,nc in enumerate(grid.ncells):
        ntheta = np.lcm(definition, nc)
        T, R = np.meshgrid(
            2*np.pi * np.arange(ntheta+1)/(ntheta) + grid.theta_offset[i], 
            grid.radii[i:i+2]
        )
        ax.pcolormesh(
            R*np.cos(T), R*np.sin(T), 
            np.repeat(data[i], ntheta//nc)[None,:], 
            vmin=dmin, vmax=dmax, **kwarg
        )
            
