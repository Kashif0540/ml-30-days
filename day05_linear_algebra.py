
"""
Day 5 - Linear Algebra Basics
30 Day ML Challenge, Week 1

Covering: matrices, dot product, matrix multiplication, transpose, eigenvalues.
This is the math NumPy has been quietly using under the hood since Day 1.
"""

import numpy as np


def matrix_basics():
    print("--- Matrices ---")

    A = np.array([[1, 2],
                  [3, 4]])
    B = np.array([[5, 6],
                  [7, 8]])

    print("A =\n", A)
    print("B =\n", B)
    print("shape of A:", A.shape)

    return A, B


def dot_product_demo():
    print("\n--- Dot Product ---")

    # classic example: total bill = quantity * price, summed up
    qty = np.array([2, 3, 1])
    price = np.array([150, 100, 500])

    total_bill = np.dot(qty, price)
    print("qty:", qty)
    print("price:", price)
    print("total bill:", total_bill)


def matrix_multiplication_demo(A, B):
    print("\n--- Matrix Multiplication ---")

    result = np.dot(A, B)
    print("A dot B =\n", result)

    # the @ operator does the same thing, just shorter to write
    result_alt = A @ B
    print("A @ B =\n", result_alt)


def transpose_demo(A):
    print("\n--- Transpose ---")
    print("A =\n", A)
    print("A.T =\n", A.T)


def eigen_demo(A):
    print("\n--- Eigenvalues & Eigenvectors ---")

    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("eigenvalues:", eigenvalues)
    print("eigenvectors:\n", eigenvectors)


# ---------------------------------------------------------------------
# Practice task - try this yourself before looking at the solution below
#
# 1. marks = [80, 90, 70], weightage = [0.3, 0.5, 0.2]
#    find the weighted total score using a dot product
# 2. make any 2x2 matrix, find its transpose
# 3. find the eigenvalues of that same matrix
# 4. bonus: multiply two 2x2 matrices using @
# ---------------------------------------------------------------------

def practice_task():
    print("\n--- Practice Task ---")

    marks = np.array([80, 90, 70])
    weightage = np.array([0.3, 0.5, 0.2])
    weighted_score = np.dot(marks, weightage)
    print("weighted score:", weighted_score)

    M = np.array([[2, 0],
                  [1, 3]])
    print("M =\n", M)
    print("M.T =\n", M.T)

    eigenvalues, _ = np.linalg.eig(M)
    print("eigenvalues of M:", eigenvalues)

    N = np.array([[1, 1],
                  [0, 1]])
    print("M @ N =\n", M @ N)


if __name__ == "__main__":
    A, B = matrix_basics()
    dot_product_demo()
    matrix_multiplication_demo(A, B)
    transpose_demo(A)
    eigen_demo(A)
    practice_task()
