"""
Day 4 — Data Visualization: Matplotlib & Seaborn
30-Day Machine Learning Challenge — Week 1: Python & Math Foundations
Author: Muhammad Kashif

Goal:
    Learn to visualize data using Matplotlib (basic plotting) and Seaborn
    (statistical plotting built on top of Matplotlib). Visualization is the
    first real step of any ML project (EDA) — before training any model,
    you need to see your data's trends, distributions, and relationships.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Muhammad Kashif AI/ML (visualization) practice\n")


def matplotlib_basics():
    print("\n--- Part 1: Matplotlib Basics ---")
    data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Sales": [200, 250, 180, 300, 280],
        "City": ["Lahore", "Lahore", "Karachi", "Karachi", "Lahore"]
    }
    df = pd.DataFrame(data)
    print(df)

    # Line Chart
    plt.figure()
    plt.plot(df["Month"], df["Sales"], marker="o")
    plt.title("Monthly Sales (Line Chart)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.savefig("day04_line_chart.png")
    print("Saved: day04_line_chart.png")
    plt.show()

    # Bar Chart
    plt.figure()
    plt.bar(df["Month"], df["Sales"], color="green")
    plt.title("Monthly Sales (Bar Chart)")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.savefig("day04_bar_chart.png")
    print("Saved: day04_bar_chart.png")
    plt.show()

    return df


def seaborn_basics(df):
    print("\n--- Part 2: Seaborn Basics ---")

    # Grouped bar chart — hue splits bars by an extra category (City)
    plt.figure()
    sns.barplot(x="Month", y="Sales", hue="City", data=df)
    plt.title("Sales by Month and City (Seaborn)")
    plt.savefig("day04_seaborn_barplot.png")
    print("Saved: day04_seaborn_barplot.png")
    plt.show()

    # Histogram
    plt.figure()
    sns.histplot(df["Sales"], bins=5, kde=True)
    plt.title("Sales Distribution")
    plt.savefig("day04_histogram.png")
    print("Saved: day04_histogram.png")
    plt.show()


# ---------------------------------------------------------------------------
# PRACTICE TASK
#
# 1. Create a DataFrame of 6 students with columns: Name, Subject, Marks, Attendance
# 2. Use plt.bar() to chart each student's Marks
# 3. Use sns.barplot() to show average Marks per Subject
# 4. Use sns.histplot() to show the distribution of Marks
# 5. Bonus: use sns.scatterplot() to see the relationship between
#    Marks and Attendance
# ---------------------------------------------------------------------------

def practice_task():
    print("\nPractice Task\n")

    students = pd.DataFrame({
        "Name": ["Ali", "Sara", "Ahmed", "Bilal", "Hina", "Zara"],
        "Subject": ["Math", "Science", "Math", "Science", "Math", "Science"],
        "Marks": [78, 85, 92, 70, 88, 95],
        "Attendance": [80, 90, 95, 65, 88, 99]
    })

    print(students)

    plt.figure()
    plt.bar(students["Name"], students["Marks"], color="coral")
    plt.title("Marks per Student")
    plt.xlabel("Student")
    plt.ylabel("Marks")
    plt.savefig("day04_practice_student_marks.png")
    print("Saved: day04_practice_student_marks.png")
    plt.show()

    plt.figure()
    sns.barplot(x="Subject", y="Marks", data=students)
    plt.title("Average Marks per Subject")
    plt.savefig("day04_practice_subject_avg.png")
    print("Saved: day04_practice_subject_avg.png")
    plt.show()

    plt.figure()
    sns.histplot(students["Marks"], bins=5, kde=True)
    plt.title("Marks Distribution")
    plt.savefig("day04_practice_marks_distribution.png")
    print("Saved: day04_practice_marks_distribution.png")
    plt.show()

    plt.figure()
    sns.scatterplot(x="Attendance", y="Marks", data=students)
    plt.title("Marks vs Attendance")
    plt.savefig("day04_practice_scatter.png")
    print("Saved: day04_practice_scatter.png")
    plt.show()


if __name__ == "__main__":
    df = matplotlib_basics()
    seaborn_basics(df)
    practice_task()
