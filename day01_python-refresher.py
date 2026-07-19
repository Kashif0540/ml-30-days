"""
Day 1 — Python Refresher: NumPy Arrays & List Comprehensions
30-Day Machine Learning Challenge — Week 1: Python & Math Foundations

Goal:
    Get comfortable with list comprehensions and NumPy arrays.
    Every ML algorithm in this challenge (linear regression, neural nets, etc.)
    operates on arrays of numbers, so fluency here is the foundation for everything else.

Run this file directly to see the output of each section:
    python day01_python_refresher.py
"""

import time
import numpy as np


# ---------------------------------------------------------------------------
# PART 1: List Comprehensions
# ---------------------------------------------------------------------------

def list_comprehension_basics():
    print("\n--- Part 1: List Comprehensions ---")

    # Traditional loop way of building a list
    squares_loop = []
    for x in range(10):
        squares_loop.append(x ** 2)
    print("Squares (loop):", squares_loop)

    # Same result using a list comprehension
    # Read as: "x squared, for each x, in range(10)"
    squares_comp = [x ** 2 for x in range(10)]
    print("Squares (comprehension):", squares_comp)

    # List comprehension with a filter condition
    # Formula: [EXPRESSION for ITEM in ITERABLE if CONDITION]
    even_squares = [x ** 2 for x in range(10) if x % 2 == 0]
    print("Even squares only:", even_squares)

    # Nested comprehension — flattening a 2D list
    matrix = [[1, 2, 3], [4, 5, 6]]
    flat = [num for row in matrix for num in row]
    print("Flattened matrix:", flat)


# ---------------------------------------------------------------------------
# PART 2: NumPy Arrays — creation & basics
# ---------------------------------------------------------------------------

def numpy_array_basics():
    print("\n--- Part 2: NumPy Array Basics ---")

    # Why NumPy? Regular Python lists don't do element-wise math.
    my_list = [1, 2, 3]
    print("Python list * 2:", my_list * 2, "(repeats the list, not what we want)")

    arr = np.array([1, 2, 3])
    print("NumPy array * 2:", arr * 2, "(multiplies each element)")

    # Creating arrays
    a = np.array([1, 2, 3, 4, 5])          # 1D array
    b = np.array([[1, 2, 3], [4, 5, 6]])   # 2D array (matrix)
    print("1D array:", a)
    print("2D array:\n", b)

    # Shortcut constructors
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    range_arr = np.arange(0, 10, 2)        # start, stop, step
    lin_arr = np.linspace(0, 1, 5)         # 5 evenly spaced points between 0 and 1
    print("Zeros (3x3):\n", zeros)
    print("Ones (2x4):\n", ones)
    print("arange(0,10,2):", range_arr)
    print("linspace(0,1,5):", lin_arr)

    # Array info
    print("Shape of b:", b.shape)   # (2, 3)
    print("Dimensions of b:", b.ndim)  # 2
    print("Data type of b:", b.dtype)  # int64


# ---------------------------------------------------------------------------
# PART 3: Vectorized operations & indexing
# ---------------------------------------------------------------------------

def numpy_vectorization_and_indexing():
    print("\n--- Part 3: Vectorized Ops & Indexing ---")

    a = np.array([1, 2, 3, 4])
    print("a * 2:", a * 2)
    print("a + a:", a + a)
    print("a ** 2:", a ** 2)
    print("sqrt(a):", np.sqrt(a))

    arr = np.array([10, 20, 30, 40, 50])
    print("arr[0]:", arr[0])
    print("arr[-1]:", arr[-1])
    print("arr[1:3]:", arr[1:3])
    print("arr[arr > 25] (boolean masking):", arr[arr > 25])


# ---------------------------------------------------------------------------
# PART 4: Performance comparison — list comprehension vs NumPy
# ---------------------------------------------------------------------------

def performance_comparison():
    print("\n--- Part 4: Performance — List Comprehension vs NumPy ---")

    n = 1_000_000

    start = time.time()
    py_squares = [x ** 2 for x in range(n)]
    py_time = time.time() - start
    print(f"List comprehension time: {py_time:.4f} seconds")

    start = time.time()
    np_squares = np.arange(n) ** 2
    np_time = time.time() - start
    print(f"NumPy vectorized time:   {np_time:.4f} seconds")

    if np_time > 0:
        print(f"NumPy was roughly {py_time / np_time:.1f}x faster")


# ---------------------------------------------------------------------------
# PRACTICE TASK (try to solve this yourself BEFORE reading the solution below)
#
# 1. Use a list comprehension to build a list of cubes of numbers 1-15
#    that are divisible by 3.
# 2. Convert that list into a NumPy array.
# 3. Use boolean masking to filter out values greater than 500.
# ---------------------------------------------------------------------------

def practice_task_solution():
    print("\n--- Practice Task Solution ---")

    # Step 1: cubes of numbers 1-15 divisible by 3
    cubes = [x ** 3 for x in range(1, 16) if x % 3 == 0]
    print("Cubes divisible by 3:", cubes)

    # Step 2: convert to NumPy array
    cubes_arr = np.array(cubes)
    print("As NumPy array:", cubes_arr)

    # Step 3: filter values greater than 500
    large_values = cubes_arr[cubes_arr > 500]
    print("Values > 500:", large_values)


if __name__ == "__main__":
    list_comprehension_basics()
    numpy_array_basics()
    numpy_vectorization_and_indexing()
    performance_comparison()

    # Comment this out first and try to solve the practice task yourself
    # in a separate scratch file before checking this solution.
    practice_task_solution()
