import numpy as np


def solve_bellman(P: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """
    Resuelve el sistema de ecuaciones lineales asociado al problema de Bellman.

    Parameters
    ----------
    P: np.ndarray
        Matriz de transición de la cadena de Markov.
    r: np.ndarray
        Vector de recompensas.
    gamma: float
        Factor de descuento.

    Returns
    -------
    v: np.ndarray
        Vector de valores óptimos.
    """
    I = np.eye(P.shape[1])  # noqa # Matriz identidad del tamaño adecuado
    A = I - gamma * P  # Matriz del sistema
    v = np.linalg.solve(A, r)  # Resolver el sistema A*v = r
    return v


def solve_bellman_inverse(P: np.ndarray, r: np.ndarray, gamma: float) -> np.ndarray:
    """
    Resuelve el sistema de ecuaciones lineales asociado al problema de Bellman usando la inversa de la matriz.

    Parameters
    ----------
    P: np.ndarray
        Matriz de transición de la cadena de Markov.
    r: np.ndarray
        Vector de recompensas.
    gamma: float
        Factor de descuento.

    Returns
    -------
    v: np.ndarray
        Vector de valores óptimos.
    """
    I = np.eye(P.shape[1])  # noqa # Matriz identidad del tamaño adecuado
    A = I - gamma * P  # Matriz del sistema
    A_inv = np.linalg.inv(A)  # Inversa de la matriz A
    v = A_inv @ r  # Resolver el sistema A*v = r utilizando la multiplicación matricial
    return v
