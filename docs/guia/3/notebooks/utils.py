import numpy as np


def v_k_step(
    P: np.ndarray, r: np.ndarray, gamma: float, tol: float = 1e-6, max_iter: int = 1000
) -> tuple[np.ndarray, int]:
    """
    Cálculo del vector de valores de una cadena de Markov con recompensas, matriz de transición y factor de descuento.

    Parameters
    ----------
    P: np.ndarray
        Matriz de transición de la cadena de Markov de dimensión (n, n) donde n es el número de estados.
    r: np.ndarray
        Vector de recompensas de dimensión (n,).
    gamma: float
        Factor de descuento.
    tol: float (default=1e-6)
        Tolerancia para la convergencia.
    max_iter: int (default=1000)
        Número máximo de iteraciones.

    Returns
    -------
    tuple[np.ndarray, int]
        - np.ndarray: Vector de valores de dimensión (n,).
        - int: Número de iteraciones realizadas.
    """
    iters = 0
    v_k = np.zeros(len(r))

    for iters in range(max_iter):
        v_k_plus_1 = r + gamma * P @ v_k
        if np.max(np.abs(v_k_plus_1 - v_k)) < tol:
            break

        v_k = v_k_plus_1

    return v_k, iters + 1


def v_k_inplace(r: np.ndarray, gamma: float, tol: float = 1e-6, max_iter: int = 1000) -> tuple[np.ndarray, int]:
    """
    Cálculo del vector de valores de una cadena de Markov con recompensas y factor de descuento.

    Parameters
    ----------
    r: np.ndarray
        Vector de recompensas de dimensión (n,).
    gamma: float
        Factor de descuento.
    tol: float (default=1e-6)
        Tolerancia para la convergencia.
    max_iter: int (default=1000)
        Número máximo de iteraciones.

    Returns
    -------
    tuple[np.ndarray, int]
        - np.ndarray: Vector de valores de dimensión (n,).
        - int: Número de iteraciones realizadas.
    """
    iters = 1
    v_k = np.zeros(len(r))

    for iters in range(max_iter):
        v_k_old = v_k.copy()
        v_k[0] = r[0] + gamma * v_k_old[1]
        v_k[1] = r[1] + gamma * v_k_old[0]

        if np.max(np.abs(v_k - v_k_old)) < tol:
            break

    return v_k, iters + 1


def gambler_problem(N: int, ph: float, tol: float = 1e-3) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Resolución del problema del apostador mediante programación dinámica.

    Parameters
    ----------
    N: int
        Capital total del apostador.
    ph: float
        Probabilidad de obtener cara en la moneda.
    tol: float (default=1e-6)
        Tolerancia para la convergencia.

    Returns
    -------
    tuple[np.ndarray, list[np.ndarray]]
        - np.ndarray: Vector de valores de dimensión (N+1,).
        - list[np.ndarray]: Lista de vectores de valores en cada iteración.
    """
    V = np.zeros(N + 1, dtype="float")
    V[N] = 1

    sweep = 0
    sweeps = []
    delta = tol

    while delta >= tol:
        V_old = V.copy()
        for n in range(1, N):
            values = []
            for a in np.arange(1, min(n, N - n) + 1, 1):
                action_value = (ph * V[n + a]) + (1 - ph) * V[n - a]
                values.append(action_value)

            values = np.array(values)
            V[n] = np.amax(values)
        sweeps.append(V_old)
        sweep += 1
        delta = np.max(np.abs(V_old - V))

    return V, sweeps


def find_policy(V: np.ndarray, ph: float, N: int) -> list[int]:
    """
    Encuentra la política óptima para el problema del apostador.

    Parameters
    ----------
    V: np.ndarray
        Vector de valores de dimensión (N+1,).
    ph: float
        Probabilidad de obtener cara en la moneda.
    N: int
        Capital total del apostador.

    Returns
    -------
    list[int]
        Lista de apuestas óptimas para cada capital.
    """
    stakes = []
    for n in range(1, N):
        a_vals = []
        for a in range(1, min(n, N - n) + 1):
            a_val = (ph * V[n + a]) + ((1 - ph) * V[n - a])
            a_vals.append(a_val)

        a_arr = np.array(a_vals)
        a_max = np.argmax(a_arr) + 1
        stakes.append(a_max)

    return stakes
