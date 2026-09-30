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