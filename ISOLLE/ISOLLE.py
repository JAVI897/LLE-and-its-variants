from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils.graph_shortest_path import graph_shortest_path
import warnings
import networkx as nx
import scipy as sp
import sklearn as skl

class ISOLLE:
    def __init__(self, X, k_n, dim, reg = 1e-03, verbose = False, k_n_graph = False, 
                 eps = None, k_n_geodesic = 3, sparsity = False, opt_eps_jumps = 1.5):
        """
        LLE object
        Parameters
        ----------
        
        X: dxn matrix
        k_n: number of neighbours
        dim: number of coordinates
        reg: regularization constant
        sparsity: if set to True the dense eigensolver of numpy is used
        
        ### Parameters for getting the geodesic distance
        k_n_graph: if set to True, two points are neighbours 
        if one is the k nearest point of the other. 
        If set to False, two points are neighbours if their 
        distance is less than epsilon
        
        eps: epsilon parameter. If is set to None, then epsilon
        is computed to be the minimum one which guarantees G to 
        be connected
        
        k_n_geodesic: k neighbours for computing the k-graph.
        This parameter will be used just if k_n_graph is set
        to True
        
        opt_eps_jumps: increasing factor for epsilon
        """
        self.X = X
        self.k_n = k_n
        self.dim = dim
        self.reg = reg if self.k_n > self.dim else 0
        self.sparsity = sparsity
        self.n = self.X.shape[0]
        self.verbose = verbose
        self.eps = eps
        self.k_n_graph = k_n_graph
        self.k_n_geodesic = k_n_geodesic
        self.opt_eps_jumps = opt_eps_jumps
    
    def __optimum_epsilon(self):
        """
        Compute epsilon
        
        To chose the minimum epsilon which guarantees G to be 
        connected, first, epsilon is set to be equal to the distance 
        from observation i = 0 to its nearest neighbour. Then
        we check if the Graph is connected, if it's not, epsilon
        is increased and the process is repeated until the Graph
        is connected
        """
        dist_matrix = pairwise_distances(self.X)
        self.eps = min(dist_matrix[0,1:])
        con = False
        while not con:
            self.eps = self.opt_eps_jumps * self.eps
            self.__construct_nearest_graph()
            G=nx.from_numpy_matrix(self._G)
            con = nx.is_connected(G)
        self.eps = np.round(self.eps, 3)
    def __construct_nearest_graph(self):
        """
        Compute weighted graph G
        """
        dist_matrix = pairwise_distances(self.X)
        if self.k_n_graph:
            nn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k_n_geodesic + 1]
        else:
            nn_matrix = np.array([ [index for index, d in enumerate(dist_matrix[i,:]) if d < self.eps and index != i] for i in range(self.n) ])
        self._D = []
        for i in range(self.n):
            d_aux = np.zeros((1, self.n))
            np.put(d_aux, nn_matrix[i], np.array([ dist_matrix[i,v] for v in nn_matrix[i]]) )
            self._D.append(d_aux[0])
        self._D = np.array(self._D)
        self._G = self._D.copy() # Adjacency matrix
        self._G[self._G > 0] = 1
    
    def __geodesic_distances(self):
        """
        Compute geodesic distances
        
        Geodesic distances are computed using 
        Dijkstra Algorithm from Sklearn
        """
        if self.eps is None:
            self.__optimum_epsilon()
        if self.verbose:
            print('[INFO] Optimum epsilon = {}'.format(self.eps))
        self.__construct_nearest_graph()
        self.G_d = graph_shortest_path(self._D, directed = False, method = 'D')
    
    def plot_neighbors_graph_2d(self, grid = False, size = (15, 10), fig = False):
        if self.X.shape[1] != 2:
            warnings.warn("Data is not 2-dimensional")
            return
        
        self.__geodesic_distances()
        # k_n nearest neighbor indices
        vecinos = np.argsort(self.G_d, axis = 1)[:, 1 : self.k_n + 1]
        if not fig:
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure(figsize=size)
        plt.scatter(self.X[:,0], self.X[:,1])
        for i in range(self.n):
            veins = vecinos[i]
            for j in range(len(veins)):
                plt.plot(self.X[[i, veins[j]], 0], self.X[[i, veins[j]], 1], c='gray')
        plt.grid(grid)
        plt.axis(False)
        plt.title('ISOLLE with k = {}'.format(self.k_n))
        if not fig:
            plt.show()
    
    def __compute_weights(self):
        """
        Compute weights
        """
        self.__geodesic_distances()
        # k_n nearest neighbor indices
        knn_matrix = np.argsort(self.G_d, axis = 1)[:, 1 : self.k_n + 1]
        
        W = [] # Initialize nxn weight matrix
        for i in range(self.n):
            x_i = self.X[i]
            G = [] # Local covariance matrix
            for j in range(self.k_n):
                x_j = self.X[knn_matrix[i][j]]
                G_aux = []
                for k in range(self.k_n):
                    x_k = self.X[knn_matrix[i][k]]
                    gjk = np.dot((x_i - x_j), (x_i - x_k))
                    G_aux.append(gjk)
                G.append(G_aux)
            G = np.array(G)
            G = G + self.reg*np.eye(*G.shape) # Regularization for G
            w = np.linalg.solve(G, np.ones((self.k_n))) # Calculate weights for x_i
            w = w / w.sum() # Normalize weights; sum(w)=1
            
            if self.verbose and i % 30 == 0:
                print('[INFO] Weights calculated for {} observations'.format(i + 1))
                
            # Create an 1xn array that will contain a 0 if x_j is not a 
            # neighbour of x_i, otherwise it will cointain the weight of x_j
            w_all = np.zeros((1, self.n))
            np.put(w_all, knn_matrix[i], w)
            W.append(list(w_all[0]))
            
        self.W_ = np.array(W)
    
    def transform(self):
        """
        Compute embedding
        """
            
        self.__compute_weights() # Compute weights
        
        if self.sparsity:
            # Implementation taken from SLT-CE-1: Locally Linear Embedding; tdiggelm
            # https://github.com/tdiggelm/eth-slt-ce-1/blob/master/slt-ce-1.ipynb
            
            W_s = sp.sparse.lil_matrix(self.W_)
            # Compute matrix M
            M = (sp.sparse.eye(*W_s.shape) - W_s).T * (sp.sparse.eye(*W_s.shape) - W_s)
            M = M.tocsc()
            
            r = skl.utils.check_random_state(3)
            v0 = r.uniform(-1, 1, M.shape[0])
            eigval, eigvec = sp.sparse.linalg.eigsh(M, k=self.dim+1, sigma=0.0, v0 = v0)
            self.Y = eigvec[:, 1:]
        else:
            # Compute matrix M
            M = (np.eye(*self.W_.shape) - self.W_).T @ (np.eye(*self.W_.shape) - self.W_) 
            eigval, eigvec = np.linalg.eigh(M) # Decompose matrix M
            self.Y = eigvec[:, 1:self.dim +1]
        
        return self.Y
    
    def plot_embedding_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, cmap = None, size = (15, 10)):
        if self.dim < 2 and dim_2 <= self.dim and dim_1 <= self.dim:
            warnings.warn("There's not enough coordinates")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        if cmap is None:
            plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors)
            
        plt.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], c = colors, cmap=cmap)
        plt.grid(grid)
        plt.title('ISOLLE with k = {} $\epsilon$ = {}'.format(self.k_n, self.eps))
        plt.xlabel('Coordinate {}'.format(dim_1))
        plt.ylabel('Coordinate {}'.format(dim_2))
        plt.show()
    
    def plot_embedding_3d(self, colors, grid = True, dim_1 = 1, dim_2 = 2, dim_3 = 3, cmap = None, size = (15, 10)):
        if self.dim < 3 and dim_2 <= self.dim and dim_1 <= self.dim and dim_3 <= self.dim:
            warnings.warn("There's not enough coordinates")
            return
        
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection="3d")
        if cmap is None:
            ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors)
        ax.scatter(self.Y[:, dim_1 - 1], self.Y[:, dim_2 - 1], self.Y[:, dim_3 - 1], c = colors, cmap = cmap)
        plt.grid(grid)
        ax.axis('on')
        plt.title('ISOLLE with k = {} $\epsilon$ = {}'.format(self.k_n, self.eps))
        ax.set_xlabel('Coordinate {}'.format(dim_1))
        ax.set_ylabel('Coordinate {}'.format(dim_2))
        ax.set_zlabel('Coordinate {}'.format(dim_3))
        plt.show()