from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.artist import Artist
from matplotlib.text import Text

import texture
from texture.grids import RegularGrid
import traceback

class Action(Enum):
    add = auto()
    remove = auto()
    update = auto()

class MatrixType(Enum):
    texture = auto()
    geometry = auto()
    topology = auto()

class Frame:
    def __init__(self, ax: Axes, grid: RegularGrid, init_pos=None):
        self.ax = ax
        self.points_mpl: list[Line2D] = []
        self.uids = []
        self.links_col: LineCollection = LineCollection([])
        self.annotations: list[Text] = []

        self.matrix_view = None

        self.showing_annotations = False

        self.grid = grid

        self.points: np.ndarray = np.empty((0, 2))
        self.links: np.ndarray = None

        ax.add_collection(self.links_col)

        if init_pos is not None:
            for x, y in init_pos:
                self.add_point(self.create_point(x, y))
            self.update()

    def update(self, point: Line2D=None, action=Action.update):
        if point is not None:
            if action is Action.remove:
                self.remove_point(point)
            elif action is Action.add:
                self.add_point(point)

        self.points = self.get_points_array()
        
        if self.showing_annotations:
            self.update_uids()
        
        self.update_links()
        
        if self.matrix_view is not None: 
            self.matrix_view.update()

    def create_point(self, x, y):
        return self.ax.plot(x, y, 'ro')[0]

    def add_point(self, point: Line2D, uid: int=None):
        self.points_mpl.append(point)
        if uid is None:
            if len(self.uids) == 0:
                uid = 0
            else:
                uid = max(self.uids) + 1
        
        if uid in self.uids:
            raise ValueError(f"Uid {uid} já existe.")
        
        self.uids.append(uid)

        x, y = point.get_xdata()[0], point.get_xdata()[0]
        annotation = self.ax.text(x, y, str(self.uids[-1]), color='black')
        
        if not self.showing_annotations:
            annotation.set_visible(False)
        self.annotations.append(annotation)

    def remove_point(self, point: Line2D):
        idx = self.points_mpl.index(point)
        point.remove()
        self.points_mpl.pop(idx)
        self.uids.pop(idx)

        self.annotations[idx].remove()
        self.annotations.pop(idx)

    def update_links(self):
        try:
            self.links = self.get_links(self.points)
            self.links_col.set_segments(self.points[self.links])
        except Exception:
            self.links_col.set_segments(np.empty((0, 2)))

    def get_links(self, points):
        return texture.links_from_voronoi(points, self.grid.size[0]/3)

    def get_points_array(self):
        points = np.empty((len(self.points_mpl), 2), dtype=float)
        for idx, p in enumerate(self.points_mpl):
            points[idx] = p.get_xdata()[0], p.get_ydata()[0]

        return points
    
    def update_uids(self):
        for a, p in zip(self.annotations, self.points):
            a.set_x(p[0])
            a.set_y(p[1])

    def show_uids(self):
        if self.showing_annotations:
            return
        
        self.update_uids()
        for a in self.annotations:
            a.set_visible(True)
        
        self.showing_annotations = True

    def remove_uids(self):
        for a in self.annotations:
            a.set_visible(False)
        self.showing_annotations = False

    def toggle_uids(self):
        if self.showing_annotations:
            self.remove_uids()
        else:
            self.show_uids()
            
    def clear(self):
        while len(self.points_mpl):
            self.remove_point(self.points_mpl[0])

        self.update()

class MatrixView:
    def __init__(self, frames: list[Frame], matrix_type: MatrixType):
        self.frames = frames
        self.ellipse_artists: list[list[Artist]] = [None, None]

        self.matrix_type = matrix_type
        self.update_func = {
            MatrixType.texture: self.update_texture,
            MatrixType.geometry: self.update_geometry,
            MatrixType.topology: self.update_topology,
        }

        for f in self.frames:
            f.matrix_view = self

    def update_texture(self):
        for frame in self.frames:
            if frame.links is None or frame.links.shape[0] == 0 or frame.points.shape[0] == 0:
                continue

            sum_m, count = texture.bin_texture_sum(frame.points, frame.links, frame.grid)
            m = texture.grid_data_mean(sum_m, count)
            
            self.draw_matrix(m, frame, scale=1)
    
    def update_geometry(self):
        p1, p2 = texture.data_in_both_frames(
            self.frames[0].points, self.frames[1].points,
            self.frames[0].uids, self.frames[1].uids,
        )

        try:
            links_1 = self.frames[0].get_links(p1)
            links_2 = self.frames[1].get_links(p2)
        except Exception as e:
            return

        links_ids = texture.links_intersect_same_points(links_1, links_2)

        sum_C, _ = texture.bin_geometrical_changes_sum(
            p1, p2, links_ids, 0.01, self.frames[0].grid
        )
        sum_B = texture.B_from_C(sum_C)
        
        count = texture.bin_count(p1, links_ids, self.frames[0].grid)

        B = texture.grid_data_mean(sum_B, count)

        for f in self.frames:
            self.draw_matrix(B, f)
    
    def update_topology(self):
        p1, p2 = texture.data_in_both_frames(
            self.frames[0].points, self.frames[1].points,
            self.frames[0].uids, self.frames[1].uids,
        )

        try:
            l1 = self.frames[0].get_links(p1)
            l2 = self.frames[1].get_links(p2)
        except Exception as e:
            return

        try:
            sum_T, *_ = texture.bin_topological_changes_sum(
                p1, p2, l1, l2, 0.01, self.frames[0].grid
            )
        except Exception as e:
            print(f"Exception occurred: {e}")
            traceback.print_exc()
            return

        count = texture.bin_count(p1, l1, self.frames[0].grid)

        T = texture.grid_data_mean(sum_T, count)

        for f in self.frames:
            self.draw_matrix(T, f)

    def draw_matrix(self, matrix, frame: Frame, scale=None):
        idx = self.frames.index(frame)
        try:
            ellipse, lines = texture.display.draw_matrices(
                frame.ax, frame.grid, matrix, 
                adjust_lims=False,
                scale=scale,
            )
        except texture.errors.AllMatricesNullError as e:
            return

        if self.ellipse_artists[idx] is not None:
            for a in self.ellipse_artists[idx]:
                a.remove()

        self.ellipse_artists[idx] = [ellipse, lines]

    def update(self):
        self.update_func[self.matrix_type]()

class PlayGround:
    links_axs: list[Axes]

    def __init__(self, 
        init_points_1=None, init_points_2=None, 
        matrix_type=MatrixType.texture, show_uids=False,
        ):
        self.fig = plt.figure(constrained_layout=True)

        m_type_to_title = {
            MatrixType.texture: "Textura (M)",
            MatrixType.geometry: "Geometria (B)",
            MatrixType.topology: "Topologia (T)",
        }

        self.fig.suptitle(m_type_to_title[matrix_type])

        gs = self.fig.add_gridspec(2, 2, height_ratios=[8, 1])

        self.links_axs = [
            self.fig.add_subplot(gs[0, 0]),
            self.fig.add_subplot(gs[0, 1]),
        ]

        self.buttons = [
            plt.Button(
                self.fig.add_subplot(gs[1, 0]), 
                "Copy other frame",
            ),
            plt.Button(
                self.fig.add_subplot(gs[1, 1]), 
                "Copy other frame",
            ),
        ]
        self.buttons[0].on_clicked(lambda e, o=1, d=0: self.copy_frame(o, d))
        self.buttons[1].on_clicked(lambda e, o=0, d=1: self.copy_frame(o, d))

        grid = RegularGrid(10, 10, 1, 1)

        self.frames = [
            Frame(self.links_axs[0], grid, init_points_1), 
            Frame(self.links_axs[1], grid, init_points_2),
        ]

        self.matrix_view = MatrixView(self.frames, matrix_type)

        for f in self.frames:
            f.grid.plot_grid(f.ax)

        self.selected_point: Line2D = None
        self.dragging_point = False

        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        for f in self.frames:
            f.update()

        if show_uids:
            # Simulate a 'u' key press to toggle UIDs visibility
            self.on_key_press(type('test', (object,), {'key': 'u'})())

    def copy_frame(self, origin: int, destiny: int):
        frame_origin = self.frames[origin]
        frame_destiny = self.frames[destiny]

        frame_destiny.clear()
        for uid, p in zip(frame_origin.uids, frame_origin.points_mpl):
            x, y = p.get_xdata()[0], p.get_ydata()[0]
            p = frame_destiny.create_point(x, y)
            frame_destiny.add_point(p, uid)
        frame_destiny.update()

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'u':
            for f in self.frames:
                f.toggle_uids()
            
            self.fig.canvas.draw()
        # elif event.key == 'c':
        #     self.frames[0].clear()
        #     self.fig.canvas.draw()
    
    def get_frame_id(self, ax: Axes) -> int:
        for idx, ax_i in enumerate(self.links_axs):
            if ax is ax_i:
                return idx

    def on_click(self, event):
        if event.inaxes not in self.links_axs:
            return
        
        if event.button == 1:  # Left mouse button
            ax = event.inaxes
            frame_idx = self.get_frame_id(ax)

            for p in self.frames[frame_idx].points_mpl:
                if p.contains(event)[0]:
                    self.selected_point = p

            if self.selected_point is None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                frame = self.frames[frame_idx]
                p = frame.create_point(event.xdata, event.ydata)
                frame.update(p, action=Action.add)
                self.selected_point = p

                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                self.fig.canvas.draw()

            self.dragging_point = True

    def on_release(self, event):
        if event.button == 1:  # Left mouse button
            if self.dragging_point:
                self.dragging_point = False
                self.selected_point = None
        if event.button == 3:  # Right mouse button
            frame_idx = self.get_frame_id(event.inaxes)
            for p in self.frames[frame_idx].points_mpl:
                if p.contains(event)[0]:
                    self.frames[frame_idx].update(p, action=Action.remove)
            self.fig.canvas.draw()

    def on_motion(self, event):
        if self.dragging_point:
            self.selected_point.set_data([event.xdata], [event.ydata])

            ax = self.selected_point.axes
            frame_id = self.get_frame_id(ax)
            self.frames[frame_id].update(self.selected_point)

            # if frame.links_col is not None:
            #     frame.links_col.remove()
            # frame.links_col = display.draw_links(ax, points, links)

            self.fig.canvas.draw()

    def run(self):
        plt.show()

if __name__ == "__main__":
    playground = PlayGround(
        init_points_1=[
            [-0.5, 0],
            [0.5, 0],
            [0, 1],
            [0, -1],
        ],
        init_points_2=[
            [-1, 0],
            [1, 0],
            [0, 0.5],
            [0, -0.5],
        ],
        matrix_type=MatrixType.topology,
        show_uids=True,
    )
    playground.run()
