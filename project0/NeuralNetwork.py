class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')

        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2, 1), 10), ((3, 3), 21), ((4, 5), 32),
                                ((6, 6), 42)]
        self.testing_points = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10)]

    def train(self, x1, x2, y):
        """My solution:
        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1
        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights * input_values + self.biases  # (3 by 1 matrix)
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)  # (3 by 1 matrix)
        output = self.hidden_to_output_weights * hidden_layer_activation  # scalar
        activated_output = output_layer_activation(output)  # scalar
        ### Backpropagation ###
        # Compute gradients
        output_layer_error = (activated_output - y) # * output_layer_activation_derivative(output)  # scalar
        hidden_layer_error = output_layer_error[0, 0] * \
                             np.multiply(self.hidden_to_output_weights.T,
                                         np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input))  # 3 x 1
        bias_gradients = hidden_layer_error
        input_to_hidden_weight_gradients = hidden_layer_error * input_values.T  # 3 x 2
        hidden_to_output_weight_gradients = output_layer_error[0, 0] * hidden_layer_activation.T  # 3 x 1
        # Use gradients to adjust weights and biases using gradient descent
        self.biases -= self.learning_rate * bias_gradients
        self.input_to_hidden_weights -= self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights -= self.learning_rate * hidden_to_output_weight_gradients
        """

        # Instructor's solution: (same)
        vec_relu = np.vectorize(rectified_linear_unit)
        vec_relu_derivative = np.vectorize(rectified_linear_unit_derivative)

        # Forward propagation
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        hidden_layer_weighted_input = self.input_to_hidden_weights*input_values  + self.biases #should be 3 by 1
        hidden_layer_activation = vec_relu(hidden_layer_weighted_input) # 3 by 1

        output = self.hidden_to_output_weights * hidden_layer_activation # 1 by 1
        activated_output = output_layer_activation(output) # 1 by 1

        # Compute gradients
        output_layer_error =  (activated_output - y) * output_layer_activation_derivative(output)  # 1 by 1
        hidden_layer_error = np.multiply((np.transpose(self.hidden_to_output_weights) * output_layer_error), vec_relu_derivative(hidden_layer_weighted_input)) # 3 by 1

        bias_gradients = hidden_layer_error
        hidden_to_output_weight_gradients = np.transpose(hidden_layer_activation * output_layer_error)# [3 by 1] * [1 by 1] = [3 by 1]
        input_to_hidden_weight_gradients = np.transpose(input_values * np.transpose(hidden_layer_error)) #  = [2 by 1] * [1 by 3] = [2 by 3]

        # Use gradients to adjust weights and biases
        self.biases = self.biases - self.learning_rate * bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate * hidden_to_output_weight_gradients


class NeuralNetwork(NeuralNetworkBase):

    def predict(self, x1, x2):

         # Instructor's solution:
        vec_relu = np.vectorize(rectified_linear_unit)

        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        hidden_layer_weighted_input = self.input_to_hidden_weights*input_values + self.biases #should be 3 by 1
        hidden_layer_activation = vec_relu(hidden_layer_weighted_input) # 3 by 1

        output = self.hidden_to_output_weights * hidden_layer_activation # 1 by 1
        activated_output = output_layer_activation(output) # 1 by 1

        return activated_output.item()



class MLP(nn.Module):
    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.flatten = Flatten()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear_first_digit = nn.Linear(64, 10)
        self.linear_second_digit = nn.Linear(64, 10)

    def forward(self, x):
        xf = self.flatten(x)
        out1 = F.relu(self.linear1(xf))
        out2 = F.relu(self.linear2(out1))
        out_first_digit = self.linear_first_digit(out2)
        out_second_digit = self.linear_second_digit(out2)
        return out_first_digit, out_second_digit


class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.linear1 = nn.Linear(input_dimension, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear_first_digit = nn.Linear(64, 10)
        self.linear_second_digit = nn.Linear(64, 10)

        self.encoder = nn.Sequential(
              nn.Conv2d(1, 8, (3, 3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              nn.Conv2d(8, 16, (3, 3)),
              nn.ReLU(),
              nn.MaxPool2d((2, 2)),
              Flatten(),
              nn.Linear(720, 128),
              nn.Dropout(0.5),
        )

        self.first_digit_classifier = nn.Linear(128,10)
        self.second_digit_classifier = nn.Linear(128,10)

    def forward(self, x):
        out = self.encoder(x)
        out_first_digit = self.first_digit_classifier(out)
        out_second_digit = self.second_digit_classifier(out)
        return out_first_digit, out_second_digit
