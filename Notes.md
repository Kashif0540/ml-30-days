# 30 Days ML — Notes

Theory notes for each day of the challenge. Code and practice exercises live in the separate `dayXX_*.py` files — this file is just for the concepts, definitions, and examples so revision is quick.

---

## Day 1: Python Refresher | List Comprehensions & NumPy Arrays

**Topic:** List comprehensions, NumPy array basics, vectorization, indexing

### Why This Matters

Every ML algorithm works on arrays of numbers. Before touching any model, you need to be comfortable creating, transforming, and filtering arrays — that's the entire point of today.

---

### 1. List Comprehension

**Definition:** A compact way to build a list by applying an expression to every item in an iterable, optionally filtering with a condition, a one-line replacement for a `for` loop that appends to a list.

Formula:
```
[EXPRESSION for ITEM in ITERABLE if CONDITION]
```

**Example:**
```python
# Loop version
squares = []
for x in range(10):
    squares.append(x ** 2)

# Comprehension version — same result, one line
squares = [x ** 2 for x in range(10)]

# With a filter condition
even_squares = [x ** 2 for x in range(10) if x % 2 == 0]

# Nested — flattening a 2D list
matrix = [[1, 2, 3], [4, 5, 6]]
flat = [num for row in matrix for num in row]
```

**Where it's used:** Filtering, transforming, or extracting data — common in preprocessing steps before a dataset even reaches NumPy or Pandas.

---

### 2. NumPy Array

**Definition:** NumPy's core object, the `ndarray`, is a fixed-type, multi-dimensional array that supports element-wise math directly — unlike a Python list.

**The problem it solves:**
```python
my_list = [1, 2, 3]
my_list * 2   # [1, 2, 3, 1, 2, 3]  -> repeats the list, not real math

arr = np.array([1, 2, 3])
arr * 2       # [2 4 6]  -> multiplies each element
```

**Creating arrays:**
```python
a = np.array([1, 2, 3, 4, 5])          # 1D
b = np.array([[1, 2, 3], [4, 5, 6]])   # 2D (matrix)

np.zeros((3, 3))        # array of zeros, given shape
np.ones((2, 4))          # array of ones, given shape
np.arange(0, 10, 2)      # like Python's range(): [0 2 4 6 8]
np.linspace(0, 1, 5)     # 5 evenly spaced points between 0 and 1
```

**Array info:**
```python
b.shape   # (2, 3) -> rows, columns
b.ndim    # 2      -> number of dimensions
b.dtype   # int64  -> data type
```

---

### 3. Vectorized Operations

**Definition:** Applying a math operation across an entire array at once, without writing a loop. NumPy runs this in compiled C code internally, which is why it's dramatically faster than a Python-level loop.

```python
a = np.array([1, 2, 3, 4])

a * 2        # [2 4 6 8]
a + a        # [2 4 6 8]
a ** 2       # [1 4 9 16]
np.sqrt(a)   # square root of each element
```

---

### 4. Indexing & Boolean Masking

```python
arr = np.array([10, 20, 30, 40, 50])

arr[0]         # 10   -> first element
arr[-1]        # 50   -> last element
arr[1:3]       # [20 30]  -> slice
arr[arr > 25]  # [30 40 50]  -> boolean masking
```

**Boolean masking** filters an array based on a condition — this is the most common way data gets filtered in ML (removing outliers, selecting rows that meet a condition, etc).

---

### 5. Performance: List Comprehension vs NumPy

Timing the same calculation (squaring 1 million numbers) both ways shows NumPy is typically 10-50x faster, because it avoids the overhead of a Python-level loop entirely. This is why every ML library is built on NumPy arrays rather than plain Python lists.

---

### Quick Reference Table

| Concept | What it does | Example |
|---|---|---|
| List comprehension | One-line loop to build a list | `[x**2 for x in range(10)]` |
| NumPy array | Fixed-type array with vectorized math | `np.array([1,2,3])` |
| Vectorized op | Math applied to whole array at once | `a * 2`, `a + a` |
| Boolean masking | Filter array by condition | `arr[arr > 25]` |

### Practice Task Covered

1. List comprehension: cubes of numbers 1-15 divisible by 3
2. Convert that list to a NumPy array
3. Boolean mask to keep only values greater than 500

Solutions are in `day01_python-refresher.py`.




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

