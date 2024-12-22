import numpy as np
from scipy.special import logsumexp
from typing import Tuple

class GaussianMixture:
    def __init__(self, mu: np.ndarray, var: np.ndarray, pi: np.ndarray):
        self.mu = mu  # (K, d) array of Gaussian means
        self.var = var  # (K,) array of Gaussian variances
        self.pi = pi  # (K,) array of mixture probabilities

def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probability of vector x under a normal distribution"""
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean) ** 2).sum() / var
    return log_prob

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a Gaussian component"""
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    ll = 0

    for i in range(n):
        mask = (X[i, :] != 0)  # Select only non-missing entries
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask], mixture.var[j])
            post[i, j] = np.log(mixture.pi[j] + 1e-16) + log_likelihood
        total = logsumexp(post[i, :])
        post[i, :] -= total
        ll += total

    return np.exp(post), ll

def mstep(X: np.ndarray, post: np.ndarray, min_variance: float = 0.25) -> GaussianMixture:
    """M-step: Updates the Gaussian mixture by maximizing the log-likelihood"""
    n, d = X.shape
    K = post.shape[1]

    n_hat = post.sum(axis=0)  # Cluster responsibilities
    pi = n_hat / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for j in range(K):
        sse, weight = 0, 0
        for l in range(d):
            mask = (X[:, l] != 0)
            n_sum = post[mask, j].sum()
            if n_sum >= 1:
                mu[j, l] = (X[mask, l] @ post[mask, j]) / n_sum
            sse += ((mu[j, l] - X[mask, l]) ** 2) @ post[mask, j]
            weight += n_sum
        var[j] = sse / weight
        var[j] = max(var[j], min_variance)  # Enforce minimum variance

    return GaussianMixture(mu, var, pi)

def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model"""
    prev_ll = None
    ll = None

    while prev_ll is None or ll - prev_ll > 1e-6 * np.abs(ll):
        prev_ll = ll
        post, ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, ll

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model"""
    X_pred = X.copy()
    K, _ = mixture.mu.shape

    for i in range(X.shape[0]):
        mask = X[i, :] != 0
        mask0 = X[i, :] == 0
        post = np.zeros(K)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask], mixture.var[j])
            post[j] = np.log(mixture.pi[j]) + log_likelihood
        post = np.exp(post - logsumexp(post))
        X_pred[i, mask0] = np.dot(post, mixture.mu[:, mask0])

    return X_pred
