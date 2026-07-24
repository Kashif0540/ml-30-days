"""
Day 3 — Pandas Basics: DataFrames, GroupBy, Merge
30-Day Machine Learning Challenge — Week 1: Python & Math Foundations

Goal:
    Learn Pandas, the library used to work with real-world tabular data
    (like an Excel sheet in code) — creating DataFrames, filtering rows,
    grouping data to compute summaries, and merging multiple tables together.

"""

import pandas as pd

def dataframe_basics():
    print("Muhammad Kashif AI/ML(pandas) Pracitce")
    print("\nPart1:DataFrame Basics\n")

    data={
            "Name": ["Ali", "Sara", "Ahmed", "Bilal", "Hina"],
        "City": ["Lahore", "Karachi", "Lahore", "Karachi", "Lahore"],
        "Marks": [85, 90, 78, 88, 92]
    }

    df=pd.DataFrame(data)
    print(df)

    print("\nName Col:\n",df["Name"])

    print("\nShape of dataframe:\n",df.shape)
    print("\nColumn names:\n",list(df.columns))
    
    return df


def filtering_basics(df):
        print("\nPart2: FIltering Basics\n")
        
        lahore_st=df[df["City"] == "Lahore"]
        print("Students from Lahore:\n",lahore_st)
        
df=dataframe_basics()
filtering_basics(df)

def groupby_basics(df):
    print("\nPart3:Groupby Basics\n")
    
    avg_city=df.groupby("City")["Marks"].mean()
    print("Avg/City:",avg_city) 
    
    city_stats=df.groupby("City")["Marks"].agg(["mean","max","min","max"])
    print("\nCity_Stats:\n",city_stats)
    

groupby_basics(df)

def merge_basics(df):
    print("\nPart3:Merge Basics")
    
    attendance=pd.DataFrame({
                "Name": ["Ali", "Sara", "Ahmed", "Bilal", "Hina"],
        "Attendance%": [95, 88, 76, 90, 99]
        })
        
    merged_df=pd.merge(df,attendance,on="Name")
    print("Merged DataFrame (Marks + Attendance):\n",merged_df)
        
merge_basics(df)


# ---------------------------------------------------------------------------
#PRACTICE TASK:
#
# 1. Create a DataFrame of 5 products with columns: Product, Category, Price
# 2. Use groupby("Category")["Price"].mean() to get average price per category
# 3. Create a second DataFrame with columns: Product, Stock
# 4. Merge both DataFrames on the "Product" column
# 5. From the merged table, filter products where Stock < 10
# ---------------------------------------------------------------------------
print("\n--------------------------\n")
def practice_task():
    print("\nPractice Task\n")
    
    data={
        "Product": ["Laptop", "Rice", "Phone","Sugar", "Headphones"],
        "Category": ["Electronics", "Grocery", "Electronics","Grocery", "Electronics"],
        "Price": [870, 53, 480,31, 230]
        }
        
    df=pd.DataFrame(data)
    print(df)
    
    avg_category=df.groupby("Category")["Price"].mean()
    print("\nAvg per category:\n",avg_category)
    
    stock=pd.DataFrame({
        "Product": ["Laptop", "Rice", "Phone", "Sugar", "Headphones"],
        "Stock": [5, 50, 8, 40, 15]
        })
        
    print("\nStock:\n",stock)
    
    merged_df=pd.merge(df,stock,on="Product")
    print("\nMerged products + stock:\n",merged_df)
    
    low_stock=merged_df[merged_df["Stock"]<10]
    print("\nProducts with Stock < 10:\n",low_stock)
        
practice_task()


    # Comment this out first and try to solve the practice task yourself
    # in a separate scratch file before checking this solution.
    practice_task_solution()
