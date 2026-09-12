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

---

## Day 2: NumPy Deep Dive | Vectorized Ops & Broadcasting

**Topic:** Vectorized operations (recap and extension), broadcasting between arrays of different shapes

### Why This Matters

Real datasets need bulk operations applied across thousands of rows at once — normalizing features, applying a fixed adjustment across a whole batch, and so on. Broadcasting is what lets NumPy do this even when array shapes don't match exactly.

---

### 1. Vectorized Operations (recap)

**Definition:** Applying arithmetic or comparison operators directly to arrays, element by element, instead of writing a loop.

```python
a = np.array([2, 4, 6, 8])
b = np.array([1, 2, 3, 4])

a + b     # [3 6 9 12]
a - b     # [1 2 3 4]
a * b     # [2 8 18 32]
a / b     # [2. 2. 2. 2.]
a > 5     # [False False True True]

np.sum(a)   # 20
np.mean(a)  # 5.0
np.max(a)   # 8
np.min(a)   # 2
```

---

### 2. Broadcasting

**Definition:** When operating on arrays of different shapes, NumPy automatically "stretches" the smaller array so the operation can apply across the larger one — without actually copying any data in memory.

**Array + scalar:**
```python
marks = np.array([60, 70, 55, 80, 65])
marks + 5   # [65 75 60 85 70]  -> 5 applied to every element
```

**Array + array (different shapes):**
```python
prices = np.array([[100, 200, 300],
                    [150, 250, 350]])   # shape (2, 3)

discount = np.array([10, 20, 30])       # shape (3,)

prices - discount
# discount is broadcast across both rows
```

**Where it's used:** Feature normalization, applying a bias term across a batch of data, and pretty much anything in Week 3 (neural networks) where a single set of weights gets applied across a whole batch of inputs.

---

### Quick Reference Table

| Concept | What it does | Example |
|---|---|---|
| Vectorized op | Element-wise math, same-shape arrays | `a + b` |
| Broadcasting | Stretch a smaller array to match a bigger one | `array + scalar`, `(2,3) - (3,)` |

### Practice Task Covered

1. Per-student average of two mark arrays + overall class average
2. Filter students scoring above 75 average
3. Convert Celsius temperatures to Fahrenheit using broadcasting
4. Subtract a per-product tax array from a sales matrix using broadcasting

Solutions are in `day2_numpy.py`.

---

## Day 3: Pandas Basics | DataFrames, GroupBy, Merge

**Topic:** Creating DataFrames, filtering rows, grouping and summarizing data, joining tables

### Why This Matters

Real-world data almost never arrives as a clean NumPy array — it comes as a table with names, categories, and dates. Pandas is the library that handles this kind of tabular data, and it's built on top of NumPy under the hood.

---

### 1. DataFrame

**Definition:** Pandas' core object — a table of rows and columns, like an Excel sheet represented in code.

```python
import pandas as pd

data = {
    "Name": ["Ali", "Sara", "Ahmed"],
    "Marks": [85, 90, 78],
    "City": ["Lahore", "Karachi", "Lahore"]
}
df = pd.DataFrame(data)
```

---

### 2. Filtering Rows

**Definition:** Selecting only the rows that meet a condition, similar to filtering a column in Excel.

```python
lahore_students = df[df["City"] == "Lahore"]
high_scorers = df[df["Marks"] > 85]
```

---

### 3. GroupBy

**Definition:** Splitting rows into groups based on a column's value, then computing a summary statistic for each group.

```python
city_avg = df.groupby("City")["Marks"].mean()

# multiple statistics at once
city_stats = df.groupby("City")["Marks"].agg(["mean", "max", "min", "count"])
```

**Where it's used:** Answering questions like "what's the average value per category" — a constant task in EDA and feature engineering.

---

### 4. Merge

**Definition:** Joining two DataFrames together based on a shared column, similar to a VLOOKUP in Excel.

```python
attendance = pd.DataFrame({
    "Name": ["Ali", "Sara", "Ahmed"],
    "Attendance%": [95, 88, 76]
})

merged_df = pd.merge(df, attendance, on="Name")
```

---

### Quick Reference Table

| Concept | What it does | Example |
|---|---|---|
| DataFrame | Table of data | `pd.DataFrame(data)` |
| Filtering | Keep rows matching a condition | `df[df["City"] == "Lahore"]` |
| GroupBy | Group rows, compute summary per group | `df.groupby("City")["Marks"].mean()` |
| Merge | Join two tables on a shared column | `pd.merge(df1, df2, on="Name")` |

### Practice Task Covered

1. Products DataFrame with Category and Price
2. Average price per category using groupby
3. Merge with a Stock DataFrame on Product
4. Filter merged table for products with Stock < 10

Solutions are in `day03_pandas.py`.

---

## Day 4: Data Visualization | Matplotlib & Seaborn

**Topic:** Line and bar charts with Matplotlib, statistical plots with Seaborn

### Why This Matters

Numbers in a table are hard to interpret at a glance. Visualization is the first real step of EDA — before training any model, you need to see trends, distributions, and relationships in the data.

---

### 1. Matplotlib Basics

**Definition:** The base Python plotting library. Every chart follows roughly the same recipe: open a figure, draw something, add labels, then show it.

```python
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["Month"], df["Sales"], marker="o")   # line chart — good for trends
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

plt.figure()
plt.bar(df["Month"], df["Sales"])   # bar chart — good for comparing categories
plt.show()
```

---

### 2. Seaborn Basics

**Definition:** A plotting library built on top of Matplotlib, aimed at statistical visualization, with less code needed for common chart types. Instead of passing data directly, you pass column names as text plus `data=df`.

```python
import seaborn as sns

sns.barplot(x="Month", y="Sales", hue="City", data=df)
# hue adds an extra grouping dimension by color

sns.histplot(df["Sales"], bins=5, kde=True)
# shows the distribution/shape of a numeric column
# kde adds a smooth curve estimate on top of the bars

sns.scatterplot(x="Attendance", y="Marks", data=df)
# shows the relationship between two numeric columns
```

---

### Quick Reference Table

| Concept | What it does | Example |
|---|---|---|
| Line chart | Trend over time | `plt.plot(x, y)` |
| Bar chart | Compare categories | `plt.bar(x, y)` |
| Seaborn bar (hue) | Grouped comparison | `sns.barplot(x, y, hue, data)` |
| Histogram | Distribution/shape of data | `sns.histplot(data, bins, kde)` |
| Scatter plot | Relationship between two variables | `sns.scatterplot(x, y, data)` |

### Practice Task Covered

1. Bar chart of each student's marks
2. Seaborn bar chart of average marks per subject
3. Histogram of marks distribution
4. Scatter plot of marks vs attendance

Solutions are in `day04_data-visualization.py`.

---

## Day 5: Linear Algebra Basics | Matrices, Dot Product, Eigenvalues

**Topic:** Matrices, dot product, matrix multiplication, transpose, eigenvalues

### Why This Matters

Every ML model from a simple linear regression to a deep neural network is doing linear algebra under the hood. Data is stored as matrices, predictions come from multiplying matrices together, and techniques like PCA rely on eigenvalues to simplify data.

---

### 1. Matrix

**Definition:** A rectangular arrangement of numbers in rows and columns — the mathematical version of a table.

```python
A = np.array([[1, 2],
              [3, 4]])
```

`A.shape` gives `(2, 2)` — 2 rows, 2 columns.

**Where it's used:** A dataset itself is a matrix — rows are records, columns are features.

---

### 2. Dot Product

**Definition:** Multiplies two vectors of the same length element by element, then adds up the results into a single number.

```
dot([a1, a2, a3], [b1, b2, b3]) = a1*b1 + a2*b2 + a3*b3
```

```python
qty = np.array([2, 3, 1])
price = np.array([150, 100, 500])
total_bill = np.dot(qty, price)
# = (2*150) + (3*100) + (1*500) = 1100
```

**Where it's used:** In linear regression, a prediction is the dot product of a weights vector and a features vector.

---

### 3. Matrix Multiplication

**Definition:** Extends the dot product to full matrices — each entry in the result is a dot product between a row of the first matrix and a column of the second.

```python
result = np.dot(A, B)
result = A @ B   # equivalent, shorter syntax
```

**Rule to remember:** For `A @ B` to work, the number of columns in `A` must match the number of rows in `B`.

**Where it's used:** Neural network layers are repeated matrix multiplications.

---

### 4. Transpose

**Definition:** Flips a matrix so rows become columns and columns become rows.

```python
A = [[1, 2], [3, 4]]
A.T = [[1, 3], [2, 4]]
```

**Where it's used:** Reshaping data so matrix dimensions line up for multiplication.

---

### 5. Eigenvalues & Eigenvectors

**Definition:** An eigenvector is a direction a matrix doesn't rotate — only stretches or shrinks. The eigenvalue is the amount of stretch in that direction.

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

**Where it's used:** PCA uses eigenvalues/eigenvectors to find which directions in a dataset carry the most information, reducing feature count while keeping important patterns.

---

### Quick Reference Table

| Concept | What it does | NumPy command |
|---|---|---|
| Matrix | Table of numbers | `np.array([[...]])` |
| Dot product | Multiply + sum, two vectors | `np.dot(a, b)` |
| Matrix multiplication | Dot product, full matrices | `A @ B` or `np.dot(A, B)` |
| Transpose | Flip rows/columns | `A.T` |
| Eigenvalues/vectors | Stretch directions of a matrix | `np.linalg.eig(A)` |

### Practice Task Covered

1. Weighted average of marks using a dot product
2. Transpose of a 2x2 matrix
3. Eigenvalues of that matrix
4. Multiplying two 2x2 matrices with `@`

Solutions are in `day05_linear_algebra.py`.

---

## Day 6: Probability & Statistics | Mean, Variance, Distributions

**Topic:** Mean, variance, standard deviation, median, normal distribution

### Why This Matters

Statistics summarize data into a few meaningful numbers, and describe how spread out the data is — both essential before feeding data into any model, and directly used in feature scaling and outlier detection later on.

---

### 1. Mean

**Definition:** The average of a set of numbers.

```python
np.mean(marks)
```

Two datasets can have the exact same mean while looking completely different — which is why mean alone isn't enough to describe data.

---

### 2. Variance & Standard Deviation

**Definition:** Variance measures how spread out data is from the mean (average of squared differences from the mean). Standard deviation is the square root of variance, expressed in the same unit as the original data, which makes it easier to interpret.

```python
np.var(marks)
np.std(marks)
```

**Example:** Two classes can both have a mean of 70, but one class might have marks tightly clustered around 70 (low std dev) while the other has marks ranging from 30 to 100 (high std dev) — same average, very different consistency.

---

### 3. Median

**Definition:** The middle value of a sorted dataset. Less sensitive to extreme outliers than the mean.

```python
np.median(marks)
```

---

### 4. Normal Distribution

**Definition:** A common data pattern where most values cluster around the mean, and fewer values appear as you move further away — producing a bell-shaped curve when plotted.

```python
normal_data = np.random.normal(loc=70, scale=5, size=1000)
# loc = mean, scale = standard deviation, size = number of values generated
```

**Where it's used:** Many real-world measurements (heights, test scores, sensor noise) roughly follow a normal distribution, and many statistical methods assume data is normally distributed.

---

### Quick Reference Table

| Concept | What it does | NumPy command |
|---|---|---|
| Mean | Average value | `np.mean(data)` |
| Variance | Average squared distance from mean | `np.var(data)` |
| Standard deviation | Square root of variance, same unit as data | `np.std(data)` |
| Median | Middle value when sorted | `np.median(data)` |
| Normal distribution | Generate bell-curve-shaped random data | `np.random.normal(loc, scale, size)` |

### Practice Task Covered

1. Mean, median, std dev of a daily expenses array
2. Interpreting what a high std dev means for consistency
3. Generating normal distribution data and checking it matches the requested mean/std dev
4. Comparing `np.percentile(data, 50)` to `np.median()`

Solutions are in `day06_probability_stats.py`.

---

## Day 7: Mini Project | EDA on a Real Dataset

**Topic:** Applying everything from Week 1 (Pandas, NumPy, statistics, visualization) to a real dataset before building any model

### Why This Matters

This is the standard first step of any real ML project — understanding the data before touching a model. Skipping this step risks feeding a model missing values, undetected outliers, or bad assumptions about the data's shape.

---

### The EDA Process

1. **Load** the data
2. **Check structure** — shape, columns, data types
3. **Check missing values**
4. **Compute statistics** — mean, std dev, min/max, quartiles
5. **Visualize** — histograms, scatter plots, box plots
6. **Summarize insights** — grouped comparisons, patterns noticed

```python
df = sns.load_dataset("iris")

df.head()                 # quick preview of the first 5 rows
df.shape                  # (rows, columns)
df.info()                 # data types and non-null counts per column
df.isnull().sum()         # count of missing values per column
df.describe()             # mean, std, min, max, quartiles in one call

sns.histplot(df["sepal_length"], kde=True)
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
sns.boxplot(x="species", y="petal_length", data=df)

df.groupby("species").mean(numeric_only=True)
```

**New commands introduced:**
- `df.head()` — preview the first 5 rows
- `df.info()` — data type and non-null count per column
- `df.isnull().sum()` — missing value count per column
- `df.describe()` — all the basic statistics from Day 6, in one call
- `sns.boxplot()` — shows the spread and outliers of a numeric column across categories

---

### Quick Reference Table

| Step | Command |
|---|---|
| Preview | `df.head()` |
| Structure | `df.shape`, `df.info()` |
| Missing values | `df.isnull().sum()` |
| Statistics | `df.describe()` |
| Visualize | `sns.histplot()`, `sns.scatterplot()`, `sns.boxplot()` |
| Grouped insight | `df.groupby(col).mean()` |

### Practice Task Covered

1. EDA process applied to the Tips dataset (head, info, missing values, describe)
2. Histogram of total_bill
3. Scatter plot of total_bill vs tip, colored by time of day
4. Average tip per day using groupby
5. Written observations about patterns noticed in the data

Solutions are in `day07_eda_project.py`.

---

## How This File Will Keep Growing

Each new day of Week 2, 3, and 4 gets its own section appended below, following the same format: topic, why it matters, definitions with examples, a quick reference table, and a note on what the practice task covered. The code and practice solutions stay in their own `dayXX_*.py` files — this file only holds the theory.
