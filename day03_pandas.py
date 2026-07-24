"""
Day 3 — Pandas Basics: DataFrames, GroupBy, Merge
30-Day Machine Learning Challenge — Week 1: Python & Math Foundations

Goal:
    Learn Pandas, the library used to work with real-world tabular data
    (like an Excel sheet in code) — creating DataFrames, filtering rows,
    grouping data to compute summaries, and merging multiple tables together.

Run this file directly to see the output of each section:
    python day03_pandas.py

Requires pandas — install it first if needed:
    pip install pandas
"""

import pandas as pd


# ---------------------------------------------------------------------------
# PART 1: Creating a DataFrame
# ---------------------------------------------------------------------------

def dataframe_basics():
    print("\n--- Part 1: DataFrame Basics ---")

    data = {
        "Name": ["Ali", "Sara", "Ahmed", "Bilal", "Hina"],
        "City": ["Lahore", "Karachi", "Lahore", "Karachi", "Lahore"],
        "Marks": [85, 90, 78, 88, 92]
    }
    df = pd.DataFrame(data)
    print("Full DataFrame:\n", df)

    # Selecting a single column
    print("\nJust the Marks column:\n", df["Marks"])

    # Basic info about the DataFrame
    print("\nShape (rows, columns):", df.shape)
    print("Column names:", list(df.columns))

    return df


# ---------------------------------------------------------------------------
# PART 2: Filtering rows
# ---------------------------------------------------------------------------

def filtering_rows(df):
    print("\n--- Part 2: Filtering Rows ---")

    # Keep only rows where City is Lahore
    lahore_students = df[df["City"] == "Lahore"]
    print("Students from Lahore:\n", lahore_students)

    # Keep only rows where Marks > 85
    high_scorers = df[df["Marks"] > 85]
    print("\nStudents with Marks > 85:\n", high_scorers)


# ---------------------------------------------------------------------------
# PART 3: GroupBy — grouping rows and summarizing them
# ---------------------------------------------------------------------------

def groupby_basics(df):
    print("\n--- Part 3: GroupBy ---")

    # Group students by City, then find the average Marks per city
    city_avg = df.groupby("City")["Marks"].mean()
    print("Average marks per city:\n", city_avg)

    # You can also get multiple statistics at once
    city_stats = df.groupby("City")["Marks"].agg(["mean", "max", "min", "count"])
    print("\nMultiple stats per city:\n", city_stats)


# ---------------------------------------------------------------------------
# PART 4: Merge — joining two DataFrames on a common column
# ---------------------------------------------------------------------------

def merge_basics(df):
    print("\n--- Part 4: Merge ---")

    attendance = pd.DataFrame({
        "Name": ["Ali", "Sara", "Ahmed", "Bilal", "Hina"],
        "Attendance%": [95, 88, 76, 90, 99]
    })

    merged_df = pd.merge(df, attendance, on="Name")
    print("Merged DataFrame (Marks + Attendance):\n", merged_df)

    return merged_df


# ---------------------------------------------------------------------------
# PRACTICE TASK (try to solve this yourself BEFORE reading the solution below)
#
# 1. Create a DataFrame of 5 products with columns: Product, Category, Price
# 2. Use groupby("Category")["Price"].mean() to get average price per category
# 3. Create a second DataFrame with columns: Product, Stock
# 4. Merge both DataFrames on the "Product" column
# 5. From the merged table, filter products where Stock < 10
# ---------------------------------------------------------------------------

def practice_task_solution():
    print("\n--- Practice Task Solution ---")

    # Step 1: products DataFrame
    products = pd.DataFrame({
        "Product": ["Laptop", "Rice", "Phone", "Sugar", "Headphones"],
        "Category": ["Electronics", "Grocery", "Electronics", "Grocery", "Electronics"],
        "Price": [120000, 200, 60000, 150, 3500]
    })
    print("Products:\n", products)

    # Step 2: average price per category
    category_avg = products.groupby("Category")["Price"].mean()
    print("\nAverage price per category:\n", category_avg)

    # Step 3: stock DataFrame
    stock = pd.DataFrame({
        "Product": ["Laptop", "Rice", "Phone", "Sugar", "Headphones"],
        "Stock": [5, 50, 8, 40, 15]
    })
    print("\nStock:\n", stock)

    # Step 4: merge on Product
    merged = pd.merge(products, stock, on="Product")
    print("\nMerged products + stock:\n", merged)

    # Step 5: filter products with low stock
    low_stock = merged[merged["Stock"] < 10]
    print("\nProducts with Stock < 10:\n", low_stock)


if __name__ == "__main__":
    df = dataframe_basics()
    filtering_rows(df)
    groupby_basics(df)
    merge_basics(df)

    # Comment this out first and try to solve the practice task yourself
    # in a separate scratch file before checking this solution.
    practice_task_solution()
