#!/usr/bin/env python3
import numpy as np

float_formatter = "{:.2f}".format

np.set_printoptions(formatter={"float_kind": float_formatter})


def is_upper_hessenberg(A: np.ndarray) -> bool:
    return np.isclose(np.tril(A, k=-2), 0.0, atol=1e-3).all()


def is_lower_hessenberg(A: np.ndarray) -> bool:
    return np.isclose(np.triu(A, k=2), 0.0, atol=1e-3).all()


# Adapted from: https://relate.cs.illinois.edu/course/cs450-s18/file-version/587fb0505883e0622b9b144b532b5e9a4e7a3684/demos/upload/04-eigenvalues/Arnoldi%20iteration.html
def arnoldi_iteration(A, b):
    n = A.shape[0]
    H = np.zeros((n, n))
    Q = np.zeros((n, n))
    # Normalize the input vector
    Q[:, 0] = b / np.linalg.norm(b)  # Use it as the first Krylov vector
    for k in range(n):
        u = A @ Q[:, k]
        for j in range(k + 1):
            qj = Q[:, j]
            H[j, k] = qj @ u
            u -= H[j, k] * qj
        if k + 1 < n:
            H[k + 1, k] = np.linalg.norm(u)
            Q[:, k + 1] = u / H[k + 1, k]

    return Q, H


if __name__ == "__main__":
    A = np.array([[1, 2, 3], [2, 2, 4], [3, 4, 4]])
    b = np.array([1, -1, 1])

    q, h = arnoldi_iteration(A, b)
    print("Did arnoldi work:")
    print(np.isclose((q.T @ A @ q - h) / np.linalg.norm(A), 0.0, atol=1e-3).all())

    print("Q:")
    print(q)
    print("H:")
    print(h)
    uh = is_upper_hessenberg(h)
    lh = is_lower_hessenberg(h)
    if uh:
        print("Wow its upper Hessenberg, whoda thunk it")
    if lh:
        print("Wow its lower Hessenberg, whoda thunk it")
    if lh and uh:
        print(
            "Wow it's tridiagonal, its cuz the matrix A is Hermitian (or Symmetric cuz its real)"
        )
