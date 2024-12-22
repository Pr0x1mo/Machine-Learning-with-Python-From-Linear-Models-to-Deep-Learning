def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture
    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, pi = mixture  # Unpack mixture tuple
    K = mu.shape[0]
    
    # Compute normal dist. matrix: (N, K)
    pre_exp = (2*np.pi*var)**(d/2)
    
    # Calc exponent term: norm matrix/(2*variance)
    post = np.linalg.norm(X[:,None] - mu, ord=2, axis=2)**2   # Vectorized version
    post = np.exp(-post/(2*var))
    
#    post = np.zeros((n, K), dtype=np.float64) # For loop version: Array to hold posterior probs and normal matrix
#    for i in range(n):  # Use single loop to complete Normal matrix: faster than broadcasting in 3D
#        dist = X[i,:] - mu     # Compute difference: will be (K,d) for each n
#        norm = np.sum(dist**2, axis=1)  # Norm: will be (K,) for each n
#        post[i,:] = np.exp(-norm/(2*var))   # This is the exponent term of normal
    
    post = post/pre_exp     # Final Normal matrix: will be (n, K)

    numerator = post*pi
    denominator = np.sum(numerator, axis=1).reshape(-1,1) # This is the vector p(x;theta)
 
    post = numerator/denominator    # This is the matrix of posterior probs p(j|i)
    
    log_lh = np.sum(np.log(denominator), axis=0).item()    # Log-likelihood
    
    return post, log_lh

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    
    nj = np.sum(post, axis=0)   # shape is (K, )
    
    pi = nj/n   # Cluster probs; shape is (K, )
    
    mu = (post.T @ X)/nj.reshape(-1,1)  # Revised means; shape is (K,d)
    
    norms = np.linalg.norm(X[:, None] - mu, ord=2, axis=2)**2    # Vectorized version

#    norms = np.zeros((n, K), dtype=np.float64) # For loopy version: Matrix to hold all the norms: (n,K)
#    for i in range(n):
#        dist = X[i,:] - mu
#        norms[i,:] = np.sum(dist**2, axis=1)
        
    var = np.sum(post*norms, axis=0)/(nj*d)     # Revised variance; shape is (K, )
    
    return GaussianMixture(mu, var, pi)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    
    old_log_lh = None
    new_log_lh = None  # Keep track of log likelihood to check convergence
    
    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        
        old_log_lh = new_log_lh
        
        # E-step
        post, new_log_lh = estep(X, mixture)
        
        # M-step
        mixture = mstep(X, post)
            
    return mixture, post, new_log_lh

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture
    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, pi = mixture   # Unpack mixture tuple
    K = mu.shape[0]
    
######## Loop version to calculate norms: 2nd fastest ########
    
    # f(u,j) matrix that's used to store the normal matrix and log of posterior probs: (p(j|u))
#    f = np.zeros((n,K), dtype=np.float64)
#    
#    # Compute the normal matrix: Single loop implementation
#    for i in range(n):
#        # For each user pick only columns that have ratings
#        Cu_indices = X[i,:] != 0
#        # Dimension of Cu (no. of non-zero entries)
#        dim = np.sum(Cu_indices)
#        # log of pre-exponent for this user's gaussian dist.
#        pre_exp = (-dim/2.0)*np.log((2*np.pi*var))
#        # Calculate the exponent term of the gaussian
#        diff = X[i, Cu_indices] - mu[:, Cu_indices]    # This will be (K,|Cu|)
#        norm = np.sum(diff**2, axis=1)  # This will be (K,)
#        
#        # Now onto the final log normal matrix: log(N(...))
#        # We will need log(normal), exp will cancel, so no need to calculate it
#        f[i,:] = pre_exp - norm/(2*var)  # This is the ith users log gaussian dist vector: (K,)
    
######## End: loop version ########
    
######## Vectorized version to calculate norms ########
    
    # Create a delta matrix to indicate where X is non-zero, which will help us pick Cu indices
    delta = X.astype(bool).astype(int)
    # Exponent term: norm matrix/(2*variance)
#    f = np.sum(((X[:, None, :] - mu)*delta[:, None, :])**2, axis=2)/(2*var) # This is using 3D broadcasting: slowest of all
    f = (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T))/(2*var) # This is using indicator matrix: fastest of all
    # Pre-exponent term: A matrix of shape (n, K)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    # Put them together
    f = pre_exp - f
    
######## End: vectorized version ########
    
    f = f + np.log(pi + 1e-16)  # This is the f(u,j) matrix
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)  # Store this to calculate log_lh
    log_posts = f - logsums # This is the log of posterior prob. matrix: log(p(j|u))
    
    log_lh = np.sum(logsums, axis=0).item()   # This is the log likelihood
    
    return np.exp(log_posts), log_lh

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian
    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    mu_rev, _, _ = mixture
    K = mu_rev.shape[0]
    
    # Calculate revised pi(j): same expression as in the naive case
    pi_rev = np.sum(post, axis=0)/n
    
    # Create delta matrix indicating where X is non-zero
    delta = X.astype(bool).astype(int)
    
    # Update means only when sum_u(p(j|u)*delta(l,Cu)) >= 1
    denom = post.T @ delta # Denominator (K,d): Only include dims that have information
    numer = post.T @ X  # Numerator (K,d)
    update_indices = np.where(denom >= 1)   # Indices for update
    mu_rev[update_indices] = numer[update_indices]/denom[update_indices] # Only update where necessary (denom>=1)
    
    # Update variances
    denom_var = np.sum(post*np.sum(delta, axis=1).reshape(-1,1), axis=0) # Shape: (K,)
    
######## Loop version for norms calc. ##########
    
    # Norm matrix for variance calc
#    norms = np.zeros((n, K), dtype=np.float64)
#    
#    for i in range(n):
#        # For each user pick only columns that have ratings
#        Cu_indices = X[i,:] != 0
#        diff = X[i, Cu_indices] - mu_rev[:, Cu_indices]    # This will be (K,|Cu|)
#        norms[i,:] = np.sum(diff**2, axis=1)  # This will be (K,)
    
######## End: loop version #########
        
######## Vectorized version for norms calc. ########
    
#    norms = np.sum(((X[:, None, :] - mu_rev)*delta[:, None, :])**2, axis=2)
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu_rev.T**2) - 2*(X @ mu_rev.T)
    
######## End: vectorized version #########
    
    # Revised var: if var(j) < 0.25, set it = 0.25
    var_rev = np.maximum(np.sum(post*norms, axis=0)/denom_var, min_variance)  
    
    return GaussianMixture(mu_rev, var_rev, pi_rev)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples
    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log_lh = None
    new_log_lh = None  # Keep track of log likelihood to check convergence
    
    # Start the main loop
    while old_log_lh is None or (new_log_lh - old_log_lh > 1e-6*np.abs(new_log_lh)):
        
        old_log_lh = new_log_lh
        
        # E-step
        post, new_log_lh = estep(X, mixture)
        
        # M-step
        mixture = mstep(X, post, mixture)
            
    return mixture, post, new_log_lh

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component
    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture
    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    mu, var, pi = mixture   # Unpack mixture tuple
    K = mu.shape[0]
    
######## Loop version to calculate norms: 2nd fastest ########
    
    # f(u,j) matrix that's used to store the normal matrix and log of posterior probs: (p(j|u))
#    f = np.zeros((n,K), dtype=np.float64)
#    
#    # Compute the normal matrix: Single loop implementation
#    for i in range(n):
#        # For each user pick only columns that have ratings
#        Cu_indices = X[i,:] != 0
#        # Dimension of Cu (no. of non-zero entries)
#        dim = np.sum(Cu_indices)
#        # log of pre-exponent for this user's gaussian dist.
#        pre_exp = (-dim/2.0)*np.log((2*np.pi*var))
#        # Calculate the exponent term of the gaussian
#        diff = X[i, Cu_indices] - mu[:, Cu_indices]    # This will be (K,|Cu|)
#        norm = np.sum(diff**2, axis=1)  # This will be (K,)
#        
#        # Now onto the final log normal matrix: log(N(...))
#        # We will need log(normal), exp will cancel, so no need to calculate it
#        f[i,:] = pre_exp - norm/(2*var)  # This is the ith users log gaussian dist vector: (K,)
    
######## End: loop version ########
    
######## Vectorized version to calculate norms ########
    
    # Create a delta matrix to indicate where X is non-zero, which will help us pick Cu indices
    delta = X.astype(bool).astype(int)
    # Exponent term: norm matrix/(2*variance)
#    f = np.sum(((X[:, None, :] - mu)*delta[:, None, :])**2, axis=2)/(2*var) # This is using 3D broadcasting: slowest of all
    f = (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T))/(2*var) # This is using indicator matrix: fastest of all
    # Pre-exponent term: A matrix of shape (n, K)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    # Put them together
    f = pre_exp - f
    
######## End: vectorized version ########
    
    f = f + np.log(pi + 1e-16)  # This is the f(u,j) matrix
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)  # Store this to calculate log_lh
    log_posts = f - logsums # This is the log of posterior prob. matrix: log(p(j|u))
    
    log_lh = np.sum(logsums, axis=0).item()   # This is the log likelihood
    
    return np.exp(log_posts), log_lh

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model
    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians
    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()
    mu, _, _ = mixture
    
    post, _ = estep(X, mixture)
    
    # Missing entries to be filled
    miss_indices = np.where(X == 0)
    X_pred[miss_indices] = (post @ mu)[miss_indices]
    
    return X_pred

from typing import Tuple
import numpy as np
from scipy.special import logsumexp


def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean) ** 2).sum() / var
    return log_prob

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # Instructor's solution:
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        mask = (X[i, :] != 0)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        total = logsumexp(post[i, :])
        post[i, :] = post[i, :] - total
        ll += total

    return np.exp(post), ll

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    """ My solution:
    n, d = X.shape
    K = post.shape[1]

    n_hat = np.sum(post, axis=0)
    p_hat = n_hat / n
    mu_hat = np.asarray(
        [np.divide((post.T @ X)[j], n_hat[j]) for j in range(K)])
    var_hat = np.asarray([(post.T @ np.sum((X - mu_hat[j]) ** 2, axis=1))[j] /
                          (d * n_hat[j]) for j in range(K)])

    return GaussianMixture(mu=mu_hat, var=var_hat, p=p_hat)
    """
    # Instructor's solution: (same)
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        # Computing mean
        mu[j, :] = (X * post[:, j, None]).sum(axis=0) / n_hat[j]
        # Computing variance
        sse = ((mu[j] - X) ** 2).sum(axis=1) @ post[:, j]
        var[j] = sse / (d * n_hat[j])

    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # Instructor's solution
    prev_ll = None
    ll = None
    while prev_ll is None or ll - prev_ll > 1e-6 * np.abs(ll):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll

def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean) ** 2).sum() / var
    return log_prob
    

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # Instructor's solution:
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        mask = (X[i, :] != 0)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        total = logsumexp(post[i, :])
        post[i, :] = post[i, :] - total
        ll += total

    return np.exp(post), ll

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # Instructor's solution
    n, d = X.shape
    _, K = post.shape

    n_hat = post.sum(axis=0)
    p = n_hat / n

    mu = mixture.mu.copy()
    var = np.zeros(K)

    for j in range(K):
        sse, weight = 0, 0
        for l in range(d):
            mask = (X[:, l] != 0)
            n_sum = post[mask, j].sum()
            if n_sum >= 1:
                # Updating mean
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum
            # Computing variance
            sse += ((mu[j, l] - X[mask, l])**2) @ post[mask, j]
            weight += n_sum
        var[j] = sse / weight
        if var[j] < min_variance:
            var[j] = min_variance

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # Instructor's solution
    prev_ll = None
    ll = None
    while prev_ll is None or ll - prev_ll > 1e-6 * np.abs(ll):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, ll

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:

    n, d = X.shape
    X_pred = X.copy()
    K, _ = mixture.mu.shape
    for i in range(n):
        mask = X[i, :] != 0
        mask0 = X[i, :] == 0
        post = np.zeros(K)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask], mixture.var[j])
            post[j] = np.log(mixture.p[j]) + log_likelihood
        post = np.exp(post - logsumexp(post))
        X_pred[i, mask0] = np.dot(post, mixture.mu[:, mask0])
    return X_pred

def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:

    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return log_prob
