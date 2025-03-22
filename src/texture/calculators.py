import numpy as np
from enum import Enum, auto

import texture
from texture.grids import RegularGrid
from texture.links import LinkCfg
from collections import namedtuple

class Tool(Enum):
    texture = auto()
    topology = auto()
    geometry = auto()
    symmetrized_velocity_gradient = auto()
    rotation_rate = auto()
    topological_rearrangement_rate = auto()

class FramesArray:
    def __init__(self, frames1: np.ndarray, frames2, grid: RegularGrid, links_cfg: LinkCfg,
        dt: float=None, uids1: np.ndarray=None, uids2: np.ndarray=None,
    ):
        '''
        Calculates all discrete and continuos tools in given frames doing an average for all frames.
        `frames1[i]` should be on time `t = t_i` and `frames2[i]` should be on time `t = t_i + dt`.

        Parameters
        ----------
        frames1:
            Positions in first frame. Array with shape (number of points, number of dimensions)
        
        frames2:
            Positions in second frame. Array with shape (number of points, number of dimensions)
        
        grid:
            Grid where tolls will be calculated.
        
        links_cfg:
            Configuration to how calculate links.
        
        dt:
            Time interval between frames2 and frames1.

        uids1, uids2:
            Unique identifiers for points. If the number of points varies between 
            `frames1` adn `frames2`, this should be given.

        Return
        ------
            results:
                Named tuple with tools. For exemple, `r.M` contains the texture.
        '''
        has_uids1 = uids1 is not None
        has_uids2 = uids2 is not None
        if has_uids1 != has_uids2:
            raise ValueError("Both uids1 and uids2 must be provided or neither.")
        
        self.has_uids = has_uids1
        
        self.frames1 = frames1
        self.frames2 = frames2
        self.dt = dt
        self.grid = grid
        self.links_cfg = links_cfg
        self.uids1 = uids1
        self.uids2 = uids2

        self.links = None
        self.links_intersect = None

    def calculate(self):
        sum_M = np.zeros(self.grid.shape + (3,), dtype=float) 
        count_M = np.zeros(self.grid.shape, dtype=int)
        
        sum_C = np.zeros_like(sum_M, shape=self.grid.shape + (2,2)) 
        sum_T = np.zeros_like(sum_M) 
    
        count_a = np.zeros_like(count_M)
        count_d = np.zeros_like(count_M)

        for idx in range(self.frames1.shape[0]):
            f1, f2 = self.frames1[idx], self.frames2[idx]
            
            if self.has_uids:
                uid1, uid2 = self.uids1[idx], self.uids2[idx]
                f1, f2 = texture.data_in_both_frames(f1, f2, uid1, uid2)

            l1, l2 = self.links_cfg.link_func(f1), self.links_cfg.link_func(f2)
            
            sum_M_i, count_M_i = texture.bin_texture_sum(f1, l1, self.grid)
            sum_M += sum_M_i
            count_M += count_M_i

            links_inter = texture.links_intersect_same_points(l1, l2)
            
            sum_C_i, _ = texture.bin_geometrical_changes_sum(
                f1, f2, links_inter, self.dt, self.grid
            )
            sum_C += sum_C_i

            sum_T_i, count_a_i, count_d_i = texture.bin_topological_changes_sum(
               f1, f2, l1, l2, self.dt, self.grid, 
            )
            sum_T += sum_T_i
            count_a += count_a_i
            count_d += count_d_i

        M = texture.grid_data_mean(sum_M, count_M)
        C = texture.grid_data_mean(sum_C, count_M)
        B = texture.B_from_C(C)
        T = texture.grid_data_mean(sum_T, count_M)

        V = texture.symmetrized_velocity_gradient(sum_M, sum_C)
        
        M_square = texture.square_from_triangular(M)
        M_square[np.linalg.det(M_square)==0] = np.eye(M_square.shape[-1])
        inv_M = np.linalg.inv(M_square)
    
        omega = texture.statistical_rotation_rate(M, C, inv_M)
        P = texture.statistical_topological_rearrangement_rate(M, T, inv_M)

        Results = namedtuple('Results', ['M', 'C', 'B', 'T', 'V', 'omega', 'P'])
        return Results(M=M, C=C, B=B, T=T, V=V, omega=omega, P=P)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from texture import links

    num_frames = 20
    num_points = 30
    grid_size = 10
    num_add = 4
    num_remove = 3 
    dl = 0.1

    grid = RegularGrid(
        length=grid_size, height=grid_size,
        num_cols=5, num_rows=5,
    )

    frames1 = np.empty((num_frames, num_points, 2))
    frames2 = np.empty((num_frames, num_points, 2))

    frames1[0] = (np.random.random((num_points, grid.num_dims)) - 1/2) * grid_size
    
    theta = np.random.random(num_points) * 2 * np.pi
    frames2[0] = frames1[0] + np.array([np.cos(theta), np.sin(theta)]).T * dl
    for idx in range(1, num_frames):
        theta = np.random.random(num_points) * 2 * np.pi
        frames1[idx] = frames2[idx-1] + np.array([np.cos(theta), np.sin(theta)]).T * dl
        theta = np.random.random(num_points) * 2 * np.pi
        frames2[idx] = frames1[idx] + np.array([np.cos(theta), np.sin(theta)]).T * dl

    # plt.scatter(*frames1[-1].T)
    # grid.plot_grid(plt.gca())
    # plt.show()

    calc = FramesArray(
        frames1, frames2, grid, 
        links_cfg=links.VoronoiLink(grid_size/3),
        dt=0.01,
    )
    r = calc.calculate()

    texture.display.draw_matrices(plt.gca(), calc.grid, r.P)
    plt.show()