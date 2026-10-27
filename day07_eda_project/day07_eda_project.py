"""
Day 7 - Mini Project: Exploratory Data Analysis (EDA)
30 Day ML Challenge, Week 1
 
This wraps up Week 1 by applying everything so far (Pandas, NumPy, stats,
visualization) to a real dataset. The goal isn't to build a model yet -
it's to understand the data before touching any model.
 
Dataset: Iris (built into seaborn, no download needed)
"""
# -------------------

import pandas as pd #stores/manipulates data in tables/dataframes structure
import seaborn as sns #data visualization library based on matplotlib
import matplotlib.pyplot as plt #seaborn is built on matplotlib, so we will use this too



def load_and_preview():
    df = sns.load_dataset("iris") #fethes a csv from internet, also auto converts to pandas dataframe
    print(df.head())
    return df


def structure_check(df):
    print("Shape:",df.shape)
    print("Columns:",df.columns.tolist())
    print(df.info())


def missing_values_check(df):
    print(df.isnull().sum())


def statistics_check(df):
    print(df.describe())


def visualize(df):
    plt.figure()
    sns.histplot(df["sepal_length"], kde=True)
    plt.title("Sepal Length Distribution")
    plt.savefig("day07_sepal_length_hist.png")
    plt.show()


    plt.figure()
    sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
    plt.title("Sepal Length vs Petal Length by Species")
    plt.savefig("day07_sepal_vs_petal_scatter.png")
    plt.show()


    plt.figure()
    sns.boxplot(x="species",y="petal_length",data=df)
    plt.title("Petal length by Species")
    plt.savefig("day07_petal_length_boxplot.png")
    plt.show()


def groupby_insights(df):
    print(df.groupby("species").mean(numeric_only=True))



if __name__ == "__main__":
    df = load_and_preview()
    structure_check(df)
    missing_values_check(df)
    statistics_check(df)
    visualize(df)
    groupby_insights(df)



# ---------------------------------------------------------------------
# Practice task - try this yourself before looking at the solution below
#
# 1. Load the "tips" dataset from seaborn
# 2. Run head(), info(), isnull().sum(), describe()
# 3. Histogram of total_bill
# 4. Scatterplot of total_bill vs tip, colored by time (Lunch/Dinner)
# 5. groupby("day")["tip"].mean() - which day gets the highest tips?
# 6. Write 3-4 lines in your own words about what you noticed
# ---------------------------------------------------------------------
 
def practice_task():
    print("----------Practice Task: Tips Dataset----------")

    tips = sns.load_dataset("tips")
    print(tips.head())
    print(tips.info())
    print("Missing Values:\n",tips.isnull().sum())
    print(tips.describe())


    plt.figure()
    sns.histplot(tips["total_bill"],kde=True)
    plt.title("Total Bill Distribution")
    plt.savefig("day07_practice_total_bill_hist.png")
    plt.show()

    plt.figure()
    sns.scatterplot(x="total_bill",y="tip",hue="time",data=tips)
    plt.title("Total Bill vs Tips by time of day")
    plt.savefig("day07_practice_bill_vs_tip_scatter.png")
    plt.show()

    day_avg_tip = tips.groupby("day")["tip"].mean()
    print("\nAverage tip per day:\n",day_avg_tip)

if __name__ == "__main__":
    practice_task()
