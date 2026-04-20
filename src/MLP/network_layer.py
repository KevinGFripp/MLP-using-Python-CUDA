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

    mW: ndarray
    vW: ndarray

    mb: ndarray
    vb: ndarray

