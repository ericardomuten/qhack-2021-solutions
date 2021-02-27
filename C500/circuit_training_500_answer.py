#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.
    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.
    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.
    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    # Define output labels as quantum state vectors
    def density_matrix(state):
        """Calculates the density matrix representation of a state.

        Args:
            state (array[complex]): array representing a quantum state vector

        Returns:
            dm: (array[complex]): array representing the density matrix
        """
        return state * np.conj(state).T


    label_0 = [[1], [0]]
    label_1 = [[0], [1]]
    state_labels = [label_0, label_1]

    dev = qml.device("default.qubit", wires=1)
    @qml.qnode(dev)
    def qcircuit(params, x, y):
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            x (array[float]): single input vector
            y (array[float]): single output state density matrix

        Returns:
            float: fidelity between output state and input
        """
        for p in params:
            qml.Rot(*x, wires=0)
            qml.Rot(*p, wires=0)
        return qml.expval(qml.Hermitian(y, wires=[0]))


    def cost(params, x, y, state_labels=None):
        """Cost function to be minimized.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            y (array[float]): 1-d array of targets
            state_labels (array[float]): array of state representations for labels

        Returns:
            float: loss value to be minimized
        """
        # Compute prediction for each input in data batch
        loss = 0.0
        dm_labels = [density_matrix(s) for s in state_labels]
        for i in range(len(x)):
            f = qcircuit(params, x[i], dm_labels[y[i]])
            loss = loss + (1 - f) ** 2
        return loss / len(x)


    def clf_predict(params, x, state_labels=None):
        """
        Tests on a given set of data.

        Args:
            params (array[float]): array of parameters
            x (array[float]): 2-d array of input vectors
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            predicted (array([int]): predicted labels for test data
        """
        dm_labels = [density_matrix(s) for s in state_labels]
        predicted = []

        for i in range(len(x)):
            fidel_function = lambda y: qcircuit(params, x[i], y)
            fidelities = [fidel_function(dm) for dm in dm_labels]
            best_fidel = np.argmax(fidelities)

            predicted.append(best_fidel)

        return np.array(predicted)


    def accuracy_score(y_true, y_pred):
        """Accuracy score.

        Args:
            y_true (array[float]): 1-d array of targets
            y_predicted (array[float]): 1-d array of predictions
            state_labels (array[float]): 1-d array of state representations for labels

        Returns:
            score (float): the fraction of correctly classified samples
        """
        score = y_true == y_pred
        return score.sum() / len(y_true)


    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data

        Args:
            inputs (array[float]): input data
            targets (array[float]): targets

        Returns:
            inputs (array[float]): one batch of input data of length `batch_size`
            targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

    def train(X_train, Y_train, learning_rate, X_test):
        # Train using Adam optimizer and evaluate the classifier
        num_layers = 3
        epochs = 2
        batch_size = 32

        accuracy_train_best = 0

        opt = qml.optimize.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)

        # initialize random weights
        params = np.random.uniform(size=(num_layers, 3))

        for it in range(epochs):
            for Xbatch, ybatch in iterate_minibatches(X_train, Y_train, batch_size=batch_size):
                params = opt.step(lambda v: cost(v, Xbatch, ybatch, state_labels), params)

            predicted_train = clf_predict(params, X_train, state_labels)
            accuracy_train = accuracy_score(Y_train, predicted_train)
            loss = cost(params, X_train, Y_train, state_labels)
            #print(accuracy_train)

            if accuracy_train > accuracy_train_best:
              best_params = params
              accuracy_train_best = accuracy_train

            if accuracy_train == 1:
              break

        return clf_predict(best_params, X_test, state_labels)
    
    
    # label 1 vs label 0 & -1
    Y_qubit_0 = np.zeros((len(Y_train),), dtype=int)
    Y_qubit_0[Y_train == 1] += 1
    # label -1 vs label 0 & 1
    Y_qubit_1 = np.zeros((len(Y_train),), dtype=int)
    Y_qubit_1[Y_train == -1] += 1
    
    # qubit 0
    Ypred_1 = train(X_train, Y_qubit_0, 0.6, X_test)
    # qubit 1
    Ypred_min1 = train(X_train, Y_qubit_1, 0.3, X_test)
    
    predictions = np.zeros((len(X_test),), dtype=int)
    predictions[Ypred_1 == 1] += 1
    predictions[Ypred_min1 == 1] += -1

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.
    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.
    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.
    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.
    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")
