def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    y = np.dot(theta, feature_vector) + theta_0
    loss = max(0.0, 1 - y * label)
    return loss
    raise NotImplementedError

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    h_loss = 1 - labels*(np.dot(theta, feature_matrix.T) + theta_0)
    h_loss[h_loss < 0] = 0
    
    return h_loss.mean()
    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    eps = 1e-8
    
    agreement = float(label*(current_theta.dot(feature_vector) + current_theta_0))
    
    if abs(agreement) < eps or agreement < 0:   # 1st condition to check if = 0
            current_theta = current_theta + label*feature_vector
            current_theta_0 = current_theta_0 + label
            
    return (current_theta, current_theta_0)
    raise NotImplementedError


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = \
            perceptron_single_step_update(feature_matrix[i,:], labels[i], \
                                          current_theta, current_theta_0)
            
    return (current_theta, current_theta_0)
    raise NotImplementedError

def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])
    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.
    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0
    
    # Keep track of the sum through the loops
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0.0
    
    n = feature_matrix.shape[0]     # No of examples
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            current_theta, current_theta_0 = \
            perceptron_single_step_update(feature_matrix[i,:], labels[i], \
                                          current_theta, current_theta_0)
            
            theta_sum = theta_sum + current_theta
            theta_0_sum = theta_0_sum + current_theta_0
            
    theta_avg = (1/(n*T))*theta_sum
    theta_0_avg = (1/(n*T))*theta_0_sum
    
    return (theta_avg, theta_0_avg)
    raise NotImplementedError
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm
    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.
    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    agreement = float(label*(current_theta.dot(feature_vector) + current_theta_0))
    
    if agreement <= 1.0:
        current_theta = (1-eta*L)*current_theta + eta*label*feature_vector
        current_theta_0 = current_theta_0 + eta*label
    else:
        current_theta = (1-eta*L)*current_theta
        
    return (current_theta, current_theta_0)

    raise NotImplementedError

def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.
    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).
    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.
    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Counter
    c = 1
    
    # Initialize theta and theta0
    current_theta = np.zeros(feature_matrix.shape[1])
    current_theta_0 = 0.0
    
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta_t = 1/np.sqrt(c)  # Update eta every iteration
            c += 1 # Update counter
            
            # Run pegasos algorithm to get theta and theta0
            current_theta, current_theta_0 = pegasos_single_step_update(feature_matrix[i,:], \
             labels[i], L, eta_t, current_theta, current_theta_0)
            
    return (current_theta, current_theta_0)
    raise NotImplementedError

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.
    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.
    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Tolerance
    eps = 1e-8
    
    predictions = theta.dot(feature_matrix.T) + theta_0
    predictions[predictions > 0.0] = 1
    predictions[predictions < 0.0] = -1
    predictions[abs(predictions) < eps] = -1
    
    return predictions
    raise NotImplementedError

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier using the perceptron algorithm with a given T
    value. The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.
    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)
    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Train the algorithm to get theta, theta0
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    
    # Use these parameters to get predictions for training and validation sets
    pred_train = classify(train_feature_matrix, theta, theta_0)
    pred_val = classify(val_feature_matrix, theta, theta_0)
    
    # Calculate classification accuracy by comparing predictions with labels
    train_accuracy = accuracy(pred_train, train_labels)
    val_accuracy = accuracy(pred_val, val_labels)
    
    return (train_accuracy, val_accuracy)
    raise NotImplementedError

