def project_onto_PC(X, pcs, n_components, feature_means):
    """
    Given principal component vectors pcs = principal_components(X)
    this function returns a new data array in which each sample in X
    has been projected onto the first n_components principcal components.
    
    Args:
    X - n x d NumPy array of n data points, each with d features
    pcs - d x d ?? full principal component matrix
    n_components - number of components to use from pcs
    feature_means - mean of feature vectors i X
    """
    X_centered = X - feature_means
    V = pcs[:,:n_components]
    return X_centered @ V

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.
        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel
        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    dot_products = np.matmul(X, np.transpose(Y)) + c
    return dot_products**p

def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.
        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)
        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    XT_X = np.mat([np.matmul(row, row.T) for row in X]).T
    YT_Y = np.mat([np.matmul(row, row.T) for row in Y]).T

    XTX_matrix = np.repeat(XT_X, Y.shape[0], axis=1)
    YTY_matrix = np.repeat(YT_Y, X.shape[0], axis=1).T

    distance_matrix = np.asarray(XTX_matrix + YTY_matrix - 2 * (np.matmul(X, Y.T)), dtype='float64')
    kernel_matrix = np.exp(-gamma * distance_matrix)
    return kernel_matrix
