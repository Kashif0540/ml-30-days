# Day 5 Notes | Linear Algebra Basics

**Topic:** Matrices, Dot Product, Matrix Multiplication, Transpose, Eigenvalues
**Week:** 1 | Python & Math Foundations

---

## Why This Matters

Every ML model from a simple linear regression to a deep neural network is doing linear algebra under the hood. Data is stored as matrices, predictions come from multiplying matrices together, and techniques like PCA rely on eigenvalues to simplify data. Today's goal was to understand the math operations NumPy has been running quietly since Day 1.

---

## 1. Matrix

**Definition:** A matrix is a rectangular arrangement of numbers in rows and columns. It's the mathematical version of a table or spreadsheet.

```python
A = np.array([[1, 2],
              [3, 4]])
```

This is a 2x2 matrix — 2 rows, 2 columns. `A.shape` gives `(2, 2)`.

**Where it's used:** A dataset itself is a matrix — rows are records (e.g. students, houses, images), columns are features (e.g. marks, price, pixel values).

---

## 2. Dot Product

**Definition:** The dot product takes two vectors of the same length, multiplies them element by element, and adds up the results into a single number.

Formula:
```
dot([a1, a2, a3], [b1, b2, b3]) = a1*b1 + a2*b2 + a3*b3
```

**Example:**
```python
qty = np.array([2, 3, 1])
price = np.array([150, 100, 500])
total_bill = np.dot(qty, price)
# = (2*150) + (3*100) + (1*500) = 1100
```

This is the same idea as calculating a grocery bill — quantity times price, added up across all items.

**Where it's used:** In linear regression, a prediction is calculated as the dot product of a weights vector and a features vector.

---

## 3. Matrix Multiplication

**Definition:** An extension of the dot product to full matrices instead of single vectors. Each entry in the result is a dot product between a row of the first matrix and a column of the second.

```python
result = np.dot(A, B)
# or equivalently
result = A @ B
```

**Rule to remember:** For `A @ B` to work, the number of columns in `A` must match the number of rows in `B`.

**Where it's used:** Neural network layers are essentially repeated matrix multiplications — inputs multiplied by weight matrices, layer after layer.

---

## 4. Transpose

**Definition:** Flipping a matrix so its rows become columns and its columns become rows.

```python
A = [[1, 2],
     [3, 4]]

A.T = [[1, 3],
       [2, 4]]
```

**Where it's used:** Transpose shows up constantly when matrix shapes don't line up for multiplication — reshaping data so dimensions match is a daily task in ML code.

---

## 5. Eigenvalues & Eigenvectors

**Definition:** For a given matrix, an eigenvector is a direction that the matrix does not rotate — it only stretches or shrinks it. The eigenvalue is the number that tells you by how much it stretches or shrinks in that direction.

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**Intuition:** Imagine stretching a rubber sheet. Most directions on the sheet get both stretched and rotated. But there are a few special directions that only get stretched, not rotated — those are the eigenvectors, and the amount of stretch is the eigenvalue.

**Where it's used:** PCA (Principal Component Analysis) uses eigenvalues/eigenvectors to figure out which directions in a dataset carry the most information, which is how it reduces the number of features while keeping the important patterns.

---

## Quick Reference Table

| Concept | What it does | NumPy command |
|---|---|---|
| Matrix | Table of numbers | `np.array([[...]])` |
| Dot product | Multiply + sum, two vectors | `np.dot(a, b)` |
| Matrix multiplication | Dot product, full matrices | `A @ B` or `np.dot(A, B)` |
| Transpose | Flip rows/columns | `A.T` |
| Eigenvalues/vectors | Stretch directions of a matrix | `np.linalg.eig(A)` |

---

## Practice Task Covered

1. Weighted average of marks using a dot product
2. Transpose of a 2x2 matrix
3. Eigenvalues of that matrix
4. Multiplying two 2x2 matrices with `@`

Solutions are in `day05_linear_algebra.py`.

