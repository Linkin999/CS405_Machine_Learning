import numpy as np


def distanceFunc(metric_type, vec1, vec2):
    """
    Computes the distance between two d-dimension vectors.

    Please DO NOT use Numpy's norm function when implementing this function.

    Args:
        metric_type (str): Metric: L1, L2, or L-inf
        vec1 ((d,) np.ndarray): d-dim vector
        vec2 ((d,)) np.ndarray): d-dim vector

    Returns:
        distance (float): distance between the two vectors
    """
    diff = vec1 - vec2
    if metric_type == "L1":
        return np.sum(np.abs(diff))
    if metric_type == "L2":
        return np.sqrt(np.sum(np.square(diff)))
    if metric_type == "L-inf":
        return np.max(np.abs(diff))


def computeDistancesNeighbors(K, metric_type, X_train, y_train, sample):
    """
    Compute the distances between every datapoint in the train_data and the
    given sample. Then, find the k-nearest neighbors.

    Return a numpy array of the label of the k-nearest neighbors.

    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        sample ((p,) np.ndarray): Single sample whose distance is to computed with every entry in the dataset

    Returns:
        neighbors (list): K-nearest neighbors' labels
    """

    # You will also call the function "distanceFunc" here
    # Complete this function

    distance = []
    for i in range(len(X_train)):
        distance.append([distanceFunc(metric_type, X_train[i], sample), y_train[i]])
    neighbors = sorted(distance, key=lambda x: x[0])[:K]
    neighbors = np.array([i[1] for i in neighbors])
    return neighbors


def Majority(neighbors):
    """
    Performs majority voting and returns the predicted value for the test sample.

    Since we're performing binary classification the possible values are [0,1].

    Args:
        neighbors (list): K-nearest neighbors' labels

    Returns:
        predicted_value (int): predicted label for the given sample
    """

    # Performs majority voting
    # Complete this function

    sum0 = 0
    sum1 = 0
    for neighbor in neighbors:
        if neighbor == 0:
            sum0 += 1
        elif neighbor == 1:
            sum1 += 1
    return 1 if sum1 >= sum0 else 0


def KNN(K, metric_type, X_train, y_train, X_val):
    """
    Returns the predicted values for the entire validation or test set.

    Please DO NOT use Scikit's KNN model when implementing this function.

    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        X_val ((n, p) np.ndarray): Validation or test data

    Returns:
        predicted_values (list): output for every entry in validation/test dataset
    """

    # Complete this function
    # Loop through the val_data or the test_data (as required)
    # and compute the output for every entry in that dataset
    # You will also call the function "Majority" here

    predictions = []
    for sample in X_val:
        neighbors = computeDistancesNeighbors(K, metric_type, X_train, y_train, sample)
        predictions.append(Majority(neighbors))
    predictions = np.array(predictions, dtype='float64')
    return predictions


def main():
    n1 = int(input())
    X_train = []
    y_train = []
    for _ in range(n1):
        inputs = input().split(' ')
        y_train.append(int(inputs[0]))
        X_train.append([float(X) for X in inputs[1:31]])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    n2 = int(input())
    Ks = []
    X_test = []
    for _ in range(n2):
        inputs = input().split(' ')
        Ks.append(int(inputs[0]))
        X_test.append([float(X) for X in inputs[1:31]])
    X_test = np.array(X_test)
    for i in range(len(X_test)):
        print(int(KNN(Ks[i], 'L2', X_train, y_train, np.array([X_test[i]]))[0]))


if __name__ == '__main__':
    main()
