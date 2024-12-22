def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians
    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data
    Returns:
        float: the BIC for this mixture
    """
    # Number of examples
    n = X.shape[0]
    
    # Number of adjustable params = total params - 1 (last prob = 1 - sum(other probs))
    p = 0
    for i in range(len(mixture)):
        if i == 0:
            p += mixture[i].shape[0] * mixture[i].shape[1]  # For means: K times d
        else:
            p += mixture[i].shape[0]    # Other params: just add K
    p = p - 1
    
    # BIC: log_lh - (1/2)*p*log(n)
    bic = log_likelihood - (p*np.log(n))/2.0
    
    return bic
    
