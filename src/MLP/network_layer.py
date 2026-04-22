from numpy import ndarray
from dataclasses import dataclass


@dataclass
class Layer:
    W: ndarray
    b: ndarray

    a: ndarray
    z: ndarray

    dW: ndarray
    db: ndarray

    gradient: ndarray

    mW: ndarray
    vW: ndarray

    mb: ndarray
    vb: ndarray

    a_train: ndarray
    z_train: ndarray

    a_test: ndarray
    z_test: ndarray

