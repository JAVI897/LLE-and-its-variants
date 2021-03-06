from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import sklearn as skl

class LLE:
    def __init__(self, X, k_n, dim, reg = 1e-03, verbose = False, sparsity = False):
        """
        LLE object
        Parameters
        ----------
        
        X: nxd matrix
        k_n: number of neighbours
        dim: number of coordinates
        reg: regularization constant
        """
        self.X = X
        self.k_n = k_n
        self.dim = dim
        self.reg = reg if self.k_n > self.dim else 0
        self.n = self.X.shape[0]
        self.verbose = verbose
        self.sparsity = sparsity
        
    def __compute_weights(self):
        """
        Compute weights
        """
        
        dist_matrix = pairwise_distances(self.X)
        # k_n nearest neighbor indices
        knn_matrix = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k_n + 1]
        
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
    
    def OutOfSampleExtension(self, P, k = 3):
        """
        Compute embedding for novel points
        Parameters
        ----------
        
        P: nxd matrix of new data points
        k: number of neighbours to use
        """
        
        n_obs = P.shape[0]
        knn = []
        for obs in range(n_obs):
            x_obs = P[obs]
            # get nearest neighbors
            dist = pairwise_distances(np.vstack([self.X, x_obs]))
            knn_aux = np.argsort(dist[self.n:,:], axis = 1)[:, 1 : k + 1]
            knn.append(knn_aux[0])             
        knn = np.array(knn)
        # compute weights
        
        W = [] # Initialize nxn weight matrix
        for i in range(n_obs):
            x_i = P[i]
            G = [] # Local covariance matrix
            for j in range(k):
                x_j = self.X[knn[i][j]]
                G_aux = []
                for k_2 in range(k):
                    x_k = self.X[knn[i][k_2]]
                    gjk = np.dot((x_i - x_j), (x_i - x_k))
                    G_aux.append(gjk)
                G.append(G_aux)
            G = np.array(G)
            G = G + self.reg*np.eye(*G.shape) # Regularization for G
            w = np.linalg.solve(G, np.ones((k))) # Calculate weights for x_i
            w = w / w.sum() # Normalize weights; sum(w)=1
            
            if self.verbose and i % 3 == 0:
                print('[INFO] Weights calculated for {} observations'.format(i + 1))
                
            W.append(w)
            
        W = np.array(W)
        y = []
        for i in range(n_obs):
            y_aux =  W[i] @ self.Y[knn[i]]
            y.append(y_aux)
        y = np.array(y)
        return y
    
    ### Metrics
    def PNE(self):
        '''
        Error de Conservación de Vecindarios - ECV
        Valencia-Aguirre et al. Global and local choice of the number of nearest neighbors in locally linear embedding
        Pattern Recognition Letters, Volume 32, Issue 16, (2011), Pages 2171-2177
        https://www.sciencedirect.com/science/article/pii/S0167865511001553
        '''
        # Nearest neighbours in X
        dist_matrix_X = pairwise_distances(self.X, metric = 'seuclidean')
        knn_matrix_X = np.argsort(dist_matrix_X, axis = 1)[:, 1 : self.k_n + 1]
        # Nearest neighbours in Y
        dist_matrix_Y = pairwise_distances(self.Y, metric = 'seuclidean')
        knn_matrix_Y = np.argsort(dist_matrix_Y, axis = 1)[:, 1 : self.k_n + 1]
        # Neighbours in Y that aren't neigbhours in X
        alpha = [[e for e in knn_matrix_Y[i] if e not in knn_matrix_X[i]] for i in range(self.n)]
        
        sum1 = sum( sum((dist_matrix_X[i,j] - dist_matrix_Y[i,j])**2 for j in knn_matrix_X[i])/self.k_n for i in range(self.n))
        sum2 = sum( sum((dist_matrix_X[i,j] - dist_matrix_Y[i,j])**2 for j in alpha[i])/len(alpha[i]) for i in range(self.n) if len(alpha[i]) > 0)
        
        return np.round((sum1 + sum2)/(2*self.n), 3)
    
    def PVC(self):
        '''
        Promedio de Vecinos Preservados – PVC
        Valencia-Aguirre et al. Comparación de Métodos de Reducción de Dimensión Basados en Análisis por Localidades
        Tecno Lógicas. 10.22430/22565337.127.
        '''
        # Nearest neighbours in X
        dist_matrix_X = pairwise_distances(self.X, metric = 'seuclidean')
        knn_matrix_X = np.argsort(dist_matrix_X, axis = 1)[:, 1 : self.k_n + 1]
        # Nearest neighbours in Y
        dist_matrix_Y = pairwise_distances(self.Y, metric = 'seuclidean')
        knn_matrix_Y = np.argsort(dist_matrix_Y, axis = 1)[:, 1 : self.k_n + 1]
        # Neighbours in Y that are neigbhours in X
        intersec_Y_X = [len([e for e in knn_matrix_Y[i] if e in knn_matrix_X[i]]) for i in range(self.n)]
        
        return sum( intersec_Y_X[i]/self.k_n for i in range(self.n)) / self.n
    
    def plot_neighbors_graph_2d(self, grid = False, size = (15, 10), fig = False):
        if self.X.shape[1] != 2:
            warnings.warn("Data is not 2-dimensional")
            return
        
        dist_matrix = pairwise_distances(self.X)
        # k_n nearest neighbor indices
        vecinos = np.argsort(dist_matrix, axis = 1)[:, 1 : self.k_n + 1]
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
        plt.title('LLE with k = {}'.format(self.k_n))
        if not fig:
            plt.show()
    
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
        plt.title('LLE with k = {}'.format(self.k_n))
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
        plt.title('LLE with k = {}'.format(self.k_n))
        ax.set_xlabel('Coordinate {}'.format(dim_1))
        ax.set_ylabel('Coordinate {}'.format(dim_2))
        ax.set_zlabel('Coordinate {}'.format(dim_3))
        plt.show()