import numpy as np
import yaml

class RetangularGrid:
    def __init__(self, edges: tuple[np.ndarray]) -> None:
        '''
        Grade retangular dado as posições das bordas das células.

        Parâmetros:
        -----------
        edges:
            Tupla com dois arrays:

            edges[0]: Posições das bordas das colunas, incluindo os extremos. 
            edges[1]: Posições das bordas das linhas, incluindo os extremos. 
        '''
        self.edges = edges

        self.num_dims = len(edges)

        # Comprimento da grade em cada dimensão
        self.size = []
        
        # Centro das células em cada dimensão
        self.dim_cell_center = []
        
        # Comprimento das células em cada dimensão
        self.dim_cell_size = []
       
        for dim in range(self.num_dims):
            self.size.append(self.edges[dim][-1] - self.edges[dim][0])
            self.dim_cell_center.append((self.edges[dim][1:] + self.edges[dim][:-1])/2)
            self.dim_cell_size.append(self.edges[dim][1:] - self.edges[dim][:-1])

        self.dim_extremes = []
        for e in self.edges:
            self.dim_extremes.append([e[0], e[-1]])

        # Quantidade de células em cada dimensão.
        # `shape_t` é o shape considerando as células fora da grade.
        self.shape = (self.edges[0].size-1, self.edges[1].size-1) 
        self.shape_t = tuple(s + 2 for s in self.shape)

        # Quantidade de células em cada dimensão, ordenados
        # na forma geralmente utilizada pelo matplotlib
        self.shape_mpl = (self.shape[1], self.shape[0])
        self.shape_mpl_t = tuple(s + 2 for s in self.shape_mpl)

        # Centro das células em cada dimensão
        self.dim_cell_center = []
        for dim in range(self.num_dims):
            self.dim_cell_center.append((self.edges[dim][1:] + self.edges[dim][:-1])/2)

        # Meshgrid do centro das células
        self.meshgrid = np.meshgrid(*self.dim_cell_center)

        # Área das células.
        # self.cell_area[i, j] = Área da célula na linha i e coluna j
        w, h = np.meshgrid(*self.dim_cell_size)
        self.cell_area = w * h


    def adjust_shape(self, arr: np.ndarray, expected_order=2, arr_name="arr"):
        '''
        Se `arr` possui `expected_order` índices, ajusta seu shape para ter 
        `expected_order + 1` índices.
        Por exemplo, se `expected_order=2` e o shape de `arr` é (N, M), o ajuste 
        muda o shape para (1, N, M).
        '''
        num_indices = len(arr.shape) 
        if num_indices == expected_order:
            arr = arr.reshape(1, *arr.shape)
        elif num_indices != (expected_order + 1):
            order1, order2 = expected_order, expected_order + 1
            raise Exception(f"`len({arr_name}.shape)` é {len(arr.shape)}, mas deveria ser {order1} ou {order2}.")

        return arr

    def simplify_shape(self, arr: np.array):
        if arr.shape[0] == 1:
            arr = arr.reshape(*arr.shape[1:])
        return arr

    def get_out_mask(self, coords: np.array):
        "Retorna a máscara dos pontos em `coords` que estão fora da grade."
        coords = self.adjust_shape(coords, arr_name="coords")
        x = coords[:, :, 0]
        y = coords[:, :, 1]
        out_x = (x < 0) | (x >= self.shape[0])
        out_y = (y < 0) | (y >= self.shape[1])
        out = out_x | out_y
        return self.simplify_shape(out)

    def remove_out_of_bounds(self, coords: np.array):
        '''
        Retorna um array que somente contém as coordenadas em `coords`
        que estão dentro da grade.
        '''
        return coords[np.logical_not(self.get_out_mask(coords))]

    def remove_cells_out_of_bounds(self, data, many_layers=False):
        '''
        Dado o array `data` de dados da grade, ou seja, com shape (num_lines, num_cols, ...),
        retira as células que estão fora da grade. Se `data` possui várias camadas, o parâmetro
        `many_layers` dever ser `True`.
        '''
        if many_layers:
            return  data[:, :-2, :-2]
        else:
            return  data[:-2, :-2]

    def count(self, coords: np.ndarray, end_id: np.ndarray=None, remove_out_of_bounds=False, simplify_shape=False):
        '''
        Contagem da quantidade de pontos em cada célula da grade, dado as coordenadas dos pontos
        na grade `coords`.

        Parâmetros:
        -----------
        end_id:
            Array 1-D com os elementos a serem considerados em coords. Ver doc de `self.sum_by_cell`.

        Retorno:
        --------
        count_grid: ndarray
            Array com a contagem dos pontos em cada célula. O elemento de índice (i, j)
            é a contagem da célula localizada na i-ésima linha e j-ésima coluna da grade.
            O índice (0, 0) é a célula no canto esquerdo inferior.
        '''
        coords = self.adjust_shape(coords, arr_name="coords")

        count_grid_shape = [coords.shape[0], *[i+2 for i in self.shape]]
        count_grid = np.zeros(count_grid_shape, dtype=int)
        for idx, coords_i in enumerate(coords):
            if end_id is not None:
                coords_i = coords_i[:end_id[idx]]

            unique_coords, count = np.unique(coords_i, axis=0, return_counts=True)
            count_grid[idx, unique_coords[:, 0], unique_coords[:, 1]] = count 

        count_grid = np.transpose(count_grid, axes=(0, 2, 1))

        if remove_out_of_bounds:
            count_grid = self.remove_cells_out_of_bounds(count_grid, many_layers=True)

        if simplify_shape:
            count_grid = self.simplify_shape(count_grid)
        
        return count_grid

    def sum_by_cell(self, values: np.array, coords: np.array, end_id: np.ndarray=None, zero_value=0, 
        remove_out_of_bounds=False, simplify_shape=False):
        '''
        Soma dos valores que estão na mesma célula (possuem a mesma coordenada) da grade. 
        Cada elemento em `values` possui uma coordenada na grade associada em `coords`.

        Parâmetros:
        -----------
        values:
            Array com N (número de pontos) elementos, que são os valores associados a cada coordenada.
            O tipo dos elementos pode ser outro array, nesse caso values iria ser um array multidimensional,
            ou algum tipo definido pelo usuário, nesse caso o tipo precisa ter definido `__add__()` e
            é necessário passar o elemento nulo em `zero_value`,

        coords:
            Coordenadas na grade que cada elemento em `values` possui.
            
        end_id:
            Array 1-D com o número de elementos a serem considerados em `coords`. Apenas os
            seguintes elementos são considerados:
            
            >>> coords[layer_id, :end_id[layer_id]]

        remove_out_of_bounds:
            Se for `True`, remove as células fora da grade antes de retornar o resultado.

        simplify_shape:
            Se for `True`, simplifica o shape do resultado. O shape é simplificado se ele apenas
            possui uma única camada.

        Retorno:
        --------
        values_sum: ndarray
            Array com a soma dos valores que estão na mesma célula. 
            Se `remove_out_of=True`, seu shape é:
            
            (N_l, N_c, [shape dos elementos em `values`])
            
            em que N_l é o número de linhas da grade e N_c o número de colunas.
            Caso `remove_out_of=False`, seu shape é:
            
            (N_l + 2, N_c + 2, [shape dos elementos em `values`])

            nesse caso o índice -1 se refere a linha/coluna antes da primeira linha/coluna,
            e o índice N_l/N_c se refere a linha/coluna logo depois da última linha/coluna.
        '''
        if len(coords.shape) == 2:
            coords = self.adjust_shape(coords, arr_name="coords")
            order = len(values.shape)
            values = self.adjust_shape(values, expected_order=order)

        v_shape = [values.shape[0], *reversed(self.shape[:2]), *self.shape[2:], *values.shape[2:]]
        for idx in range(1, 1+len(self.shape)):
            v_shape[idx] += 2

        values_sum = np.full(v_shape, fill_value=zero_value, dtype=values.dtype)

        layer_ids = list(range(coords.shape[0]))

        if end_id is None:
            for idx in range(coords.shape[1]):
                values_sum[layer_ids, coords[:, idx, 1], coords[:, idx, 0]] += values[:, idx]
        else:
            for layer_id in layer_ids:
                for idx, c in enumerate(coords[layer_id,:end_id[layer_id]]):
                    values_sum[layer_id, c[1], c[0]] += values[layer_id, idx]

        if remove_out_of_bounds:
            values_sum = self.remove_cells_out_of_bounds(values_sum, many_layers=True)

        if simplify_shape:
            values_sum = self.simplify_shape(values_sum)
        
        return values_sum

    def mean_by_cell(self, values: np.array, coords: np.array, end_id=None, count: np.array=None,
        simplify_shape=False, remove_out_of_bound=False):
        '''
        Mesma função de `self.sum_by_cell`, mas divide o resultado pela
        contagem de pontos em cada célula, assim realizando a média por célula.
        '''
        if len(coords.shape) == 2:
            coords = self.adjust_shape(coords, arr_name="coords")
            order = len(values.shape)
            values = self.adjust_shape(values, expected_order=order)

        values_mean = self.sum_by_cell(
            values, coords, 
            end_id=end_id,
        )
        
        if count is None:
            count = self.count(coords, end_id=end_id)

        non_zero_mask = count > 0

        # if len(coords.shape) == 3:
        #     num_new_axis = len(values.shape) - 2
        # else:
        #     num_new_axis = len(values.shape) - 1
        num_new_axis = len(values.shape) - 2

        values_mean[non_zero_mask] /= count[non_zero_mask].reshape(-1, *[1 for _ in range(num_new_axis)])
        
        if remove_out_of_bound:
            values_mean = self.remove_cells_out_of_bounds(values_mean, many_layers=True)
        
        if simplify_shape:
            values_mean = self.simplify_shape(values_mean)

        return values_mean

    def circle_mask(self, radius, center=(0, 0), mode="outside"):
        '''
        Máscara das células para o círculo de centro `center` e raio `radius`. 
        A máscara em questão depende de `mode`, quu pode assumir os seguintes valores:

        outsise: Células fora do círculo. 
        inside: Células dentro do círculo. 
        intersect: Células intersectando o perímetro do círculo. 
        '''
        valid_modes = ["outside", "inside", "intersect"]
        if mode not in valid_modes:
            raise ValueError(f"Valor inválido de `mode`: {mode}. Os valores válidos são: {valid_modes}.")

        x_max = self.meshgrid[0] + self.dim_cell_size[0]/2
        x_min = self.meshgrid[0] - self.dim_cell_size[0]/2
        
        y_max = (self.meshgrid[1].T + self.dim_cell_size[1]/2).T
        y_min = (self.meshgrid[1].T - self.dim_cell_size[1]/2).T
        
        x_max_sqr = np.square(x_max - center[0]) 
        x_min_sqr = np.square(x_min - center[0]) 
        
        y_max_sqr = np.square(y_max - center[1]) 
        y_min_sqr = np.square(y_min - center[1]) 

        d1 = np.sqrt(x_max_sqr + y_max_sqr)
        d2 = np.sqrt(x_max_sqr + y_min_sqr)
        d3 = np.sqrt(x_min_sqr + y_max_sqr)
        d4 = np.sqrt(x_min_sqr + y_min_sqr)

        if mode == "outside":
            return np.logical_not((d1 < radius) | (d2 < radius) | (d3 < radius) | (d4 < radius))
        if mode == "inside":
            return (d1 < radius) & (d2 < radius) & (d3 < radius) & (d4 < radius)
        if mode == "intersect":
            inside = (d1 < radius) & (d2 < radius) & (d3 < radius) & (d4 < radius)
            outside = np.logical_not((d1 < radius) | (d2 < radius) | (d3 < radius) | (d4 < radius))
            return (~outside) & (~inside)


    def plot_grid(self, ax, adjust_lims=True):
        from matplotlib.axes import Axes
        from matplotlib.collections import LineCollection
        ax: Axes = ax

        x1 = self.dim_cell_center[0] - self.dim_cell_size[0]/2
        y1 = self.dim_cell_center[1] - self.dim_cell_size[1]/2

        max_x = x1[-1] + self.dim_cell_size[0][-1]
        max_y = y1[-1] + self.dim_cell_size[1][-1]

        lines = []
        for x in x1:
            lines.append([(x, y1[0]), (x, max_y)])
        lines.append([(max_x, y1[0]), (max_x, max_y)])
        for y in y1:
            lines.append([(x1[0], y), (max_x, y)])
        lines.append([(x1[0], max_y), (max_x, max_y)])
        
        if adjust_lims:
            offset = 0.3
            ax.set_xlim(
                self.dim_cell_center[0][0] - self.dim_cell_size[0][0]/2 * (1 + offset),
                self.dim_cell_center[0][-1] + self.dim_cell_size[0][-1]/2 * (1 + offset),
            )
            ax.set_ylim(
                self.dim_cell_center[1][0] - self.dim_cell_size[1][0]/2 * (1 + offset),
                self.dim_cell_center[1][-1] + self.dim_cell_size[1][-1]/2 * (1 + offset),
            )

        ax.add_collection(LineCollection(lines, color="black"))

    def plot_center(self, ax):
        from matplotlib.axes import Axes
        ax: Axes = ax
        ax.scatter(self.meshgrid[0], self.meshgrid[1], c="black")
    
    def plot_corners(self, ax):
        from matplotlib.axes import Axes
        ax: Axes = ax
        
        x1 = self.meshgrid[0] + self.dim_cell_size[0]/2
        x2 = self.meshgrid[0] - self.dim_cell_size[0]/2
        
        y1 = (self.meshgrid[1] + self.dim_cell_size[1].reshape(-1, 1)/2)
        y2 = (self.meshgrid[1] - self.dim_cell_size[1].reshape(-1, 1)/2)
        
        ax.scatter(x1, y1, c="black")
        ax.scatter(x1, y2, c="black")
        ax.scatter(x2, y1, c="black")
        ax.scatter(x2, y2, c="black")

class RegularGrid(RetangularGrid):
    def __init__(self, length: float, height: float, num_cols: int, num_rows: int, center=(0, 0)) -> None:
        "Grade retangular para uma retângulo centrado em `center`, com lados `length`x`height`."
        self.center = center
        self.length = length
        self.height = height
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.center = center

        edges = (
            np.linspace(-length/2 + center[0], length/2 + center[0], num_cols+1),
            np.linspace(-height/2 + center[1], height/2 + center[1], num_rows+1),
        )
        super().__init__(edges)

        # Tamanho da célula em cada dimensão.
        self.cell_size = (
            self.edges[0][1] - self.edges[0][0],
            self.edges[1][1] - self.edges[1][0],
        )
    
    @classmethod
    def from_edges(Cls, edges):
        x_max, x_min = edges[0][-1], edges[0][0] 
        y_max, y_min = edges[1][-1], edges[1][0] 
        return Cls(
            length = x_max - x_min,
            height = y_max - y_min,
            num_cols = len(edges[0])-1, num_rows = len(edges[1])-1,
            center = ((x_max + x_min)/2, (y_max + y_min)/2),
        )

    def coords(self, points: np.ndarray, check_out_of_bounds=True, simplify_shape=False):
        '''
        Calcula as coordenadas dos pontos em `points` na grade.

        Parâmetros:
        -----------
        points:
            Array com os pontos, cujo shape pos ser:
                * (N, 2): 
                    N é o número de pontos e 2 vem das duas dimensões da grade 
                    (0 para o eixo x e 1 para o eixo y).

                * (M, N, 2):
                    Essencialmente uma lista de M arrays com N pontos.
        
        Retorno:
        --------
        coords: ndarray
            Array com o mesmo shape de `points` em que o i-ésimo elemento é a coordenada do 
            i-ésimo ponto na grade. O segundo índice do shape indica a dimensão da coordenada:
            
                coords[i, 0] -> Coordenada no eixo x (Coluna) \n
                coords[i, 1] -> Coordenada no eixo y (Linha)
            
            A célula da grade que está no canto esquerdo inferior possui coordenada (0, 0).
            Se `points` tem shape (M, N, 2), então o exemplo acima fica `coords[m, i, 0]`.
        '''
        num_indices = len(points.shape) 
        if num_indices == 2:
            points = points.reshape(1, *points.shape)
        elif num_indices != 3:
            raise Exception(f"`len(points.shape)` é {len(points.shape)}, mas deveria ser 2 ou 3.")

        x = points[:, :, 0] - self.center[0] + self.length/2
        y = points[:, :, 1] - self.center[1] + self.height/2

        col_pos = np.floor(x / self.cell_size[0]).astype(int)
        row_pos = np.floor(y / self.cell_size[1]).astype(int)

        if check_out_of_bounds:
            col_pos[col_pos >= self.shape[0]] = self.shape[0]
            row_pos[row_pos >= self.shape[1]] = self.shape[1]
            col_pos[col_pos < 0] = -1
            row_pos[row_pos < 0] = -1

        coords = np.empty(points.shape, int)

        coords[:, :, 0] = col_pos
        coords[:, :, 1] = row_pos

        if simplify_shape and coords.shape[0] == 1:
            coords = coords.reshape(*coords.shape[1:])

        return coords

    @property
    def configs(self):
        return {
            "length": self.length, "height": self.height,
            "num_cols": self.num_cols, "num_rows": self.num_rows,
            "center": self.center,
        }

    def save_configs(self, path):
        with open(path, "w") as f:
            yaml.dump(self.configs, f)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            configs = yaml.unsafe_load(f)
        return cls(**configs)

if __name__ == "__main__":
    l, h = 10, 5
    grid = RegularGrid(l, h, 7, 5)

    n = 10
    xs = (np.random.random(n) - 0.5) * l
    ys = (np.random.random(n) - 0.5) * h
    ps = np.array([xs, ys]).T

    coords = grid.coords(ps)

    coords = np.array([
        [0, 0],
        [0, 0],
        [1, 2],
        [1, 2],
        [1, 2],
    ])

    values = np.array([
        [1 , 1 ],
        [-3, 4 ],
        [3 , -3],
        [6 , 2 ],
        [-1, 11],
    ], dtype=float)

    r_all = grid.mean_by_cell(values, coords) 

    check_x = grid.mean_by_cell(values[:, 0], coords) == r_all[:, :, 0]
    check_y = grid.mean_by_cell(values[:, 1], coords) == r_all[:, :, 1]
    print("Mean by cell Test:", check_x.all(), check_y.all())

    # import matplotlib.pyplot as plt
    # plt.scatter(xs, ys, c="black")
    # plt.show()