"""A set of numpy exercises"""
import numpy as np
import pandas as pd

def zero_insert(x):
    """
    Write a function that takes in a vector and returns a new vector where
    every element is separated by 4 consecutive zeros.
    Example:
    [4, 2, 1] --> [4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
    """
    v = np.zeros(4)
    solution = np.array([np.append(i, v) for i in x])
    solution = np.hstack(solution)
    return solution

QC_zero_insert = zero_insert(x=np.array([4,2,1,3,7,6]))
print("Zero insert solution:")
print(QC_zero_insert)


def return_closest(x, val):
    """
    Write a function that takes in a vector and returns the value contained in
    the vector that is closest to a given value.
    If two values are equidistant from val, return the one that comes first in
    the vector.
    Example:
    ([3, 4, 5], 2) --> 3
    """

    indx = np.argmin([np.sqrt((i - val)**2) for i in x])
    return (x[indx])
    # in case of same value, argmin will return the first instance by default
    raise NotImplementedError

QC_return_closest= return_closest([3, 4, 5 ], 2)
print("return closest solution:")
print(QC_return_closest)

def cauchy(x, y):
    """
    Write a function that takes in two vectors and returns the associated Cauchy
    matrix with entries a_ij = 1/(x_i-y_j).

    Example:
    ([1, 2], [3, 4]) --> [[-1/2, -1/3], [-1, -1/2]]

    Note: the function should raise an error of type ValueError if there is a
    pair (i,j) such that x_i=y_j

    :param x: input vector
    :type x: numpy.array of int/float
    :param y: input vector
    :type y: numpy.array of int/float
    :return: Cauchy matrix with entries 1/(x_i-y_j)
    :rtype: numpy.array of float
    :raise ValueError:
    """

    result = [1 / (i - j) for i in x for j in y]
    if (x != y).any():
        return result
    else:
        return "error"


QC_cauchy = cauchy(x=np.array([1, 2]), y=np.array([3, 4]))
print("Cauchy solution:")
print(QC_cauchy)


def most_similar(x, v_list):
    """
    Write a function that takes in a vector x and a list of vectors and finds,
    in the list, the index of the vector that is most similar to x using
    cosine similarity.

    Example:
    ([1, 1], [[1, 0.9], [-1, 1]]) --> 0 (corresponding to [1,0.9])

    """
    cosim = np.sum(x @ v_list) / (np.sum(x ** 2) * np.sum(v_list * 2))
    cosim = [np.sum(x @ v) / (np.sum(x ** 2) * np.sum(v ** 2)) for v in v_list]
    result = np.argmax(cosim)

    return result

QC_most_similar = most_similar(x=np.array([1, 1]), v_list=np.array([[1, 0.9], [-1, 1]]))
print("most similar solution")
print(QC_most_similar)


def gradient_descent(x_0, learning_rate, tol):
    """
    Write a function that does gradient descent with a fixed learning_rate
    on function f with gradient g and stops when the update has magnitude
    under a given tolerance level (i.e. when |xk-x(k-1)| < tol).
    Return a tuple with the position, the value of f at that position and the
    magnitude of the last update.
    h(x) = (x-1)^2 + exp(-x^2/2)
    f(x) = log(h(x))
    g(x) = (2(x-1) - x exp(-x^2/2)) / h(x)

    Example:
    (1.0, 0.1, 1e-3) --> approximately (1.2807, -0.6555, 0.0008)

    :param x_0: initial point
    :type x_0: float
    :param learning_rate: fixed learning_rate
    :type learning_rate: float
    :param tol: tolerance for the magnitude of the update
    :type tol: float
    :return: the position, the value at that position and the latest update
    :rtype: tuple of three float
    """

    h = lambda x: (x - 1) ** 2 + np.exp(-x ** 2 / 2)
    f = lambda x: np.log(h(x))  # this is the proper function
    g = lambda x: (2 * (x - 1) - x * np.exp(-x ** 2 / 2)) / h(x)  # this is the gradient:
    # Chain Rule: g(x) = f(h(x)) then g'(x) = f'(h(x))h'(x)

    x = x_0
    while g(x) * (-learning_rate) >= tol:
        x += g(x) * (-learning_rate)

    return (x, f(x), (g(x) * (-learning_rate)))


QC_gradient_descent = gradient_descent(x_0=1.0, learning_rate=0.1, tol=1e-3)
print("gradient descent solution:")
print(QC_gradient_descent)



































