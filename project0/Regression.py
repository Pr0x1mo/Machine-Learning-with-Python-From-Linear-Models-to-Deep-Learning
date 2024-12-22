def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization
    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    theta = np.linalg.inv(X.T.dot(X) + lambda_factor*np.eye(X.shape[1])).dot(X.T).dot(Y)
    
    return theta

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation
    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    clf = LinearSVC(C = 0.1, random_state = 0)
    clf.fit(train_x, train_y)
    
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y

def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy
    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    clf = LinearSVC(C = 0.1, random_state = 0)
    clf.fit(train_x, train_y)
    
    pred_test_y = clf.predict(test_x)
    
    return pred_test_y 


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1
    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    
    # Compute the matrix of theta*X' (each row is a category, column an example)
    R = (theta.dot(X.T))/temp_parameter
    
    # Compute fixed deduction factor for numerical stability (c is a vector: 1xn)
    c = np.max(R, axis = 0)
    
    # Compute H matrix
    H = np.exp(R - c)
    
    # Divide H by the normalizing term
    H = H/np.sum(H, axis = 0)
    
    return H   

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.
    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns
        c - the cost value (scalar)
    """
    
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # avg error term
    
    # Clip prob matrix to avoid NaN instances
    clip_prob_matrix = np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15)
    
    # Take the log of the matrix of probabilities
    log_clip_matrix = np.log(clip_prob_matrix)
    
    # Create a sparse matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()
    
    # Only add terms of log(matrix of prob) where M == 1
    error_term = (-1/n)*np.sum(log_clip_matrix[M == 1])    
                
    # Regularization term
    reg_term = (lambda_factor/2)*np.linalg.norm(theta)**2
    
    return error_term + reg_term

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent
    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # Get number of labels
    k = theta.shape[0]
    
    # Get number of examples
    n = X.shape[0]
    
    # Create spare matrix of [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape=(k,n)).toarray()
    
    # Matrix of Probabilities
    P = compute_probabilities(X, theta, temp_parameter)
    
    # Gradient matrix of theta
    grad_theta = (-1/(temp_parameter*n))*((M - P) @ X) + lambda_factor*theta
    
    # Gradient descent update of theta matrix
    theta = theta - alpha*grad_theta
    
    return theta

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.
    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set
    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    
    return (train_y_mod3, test_y_mod3)


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)
    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    y_pred = get_classification(X, theta, temp_parameter)
    
    return 1 - (np.mod(y_pred, 3) == Y).mean()
