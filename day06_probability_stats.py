"""
Day 6 - Probability & Statistics
30 Day ML Challenge, Week 1

Covering: mean, variance, standard deviation, median, and normal distributions.
These are the tools used to summarize data and understand how spread out it is,
before any model ever sees it.
"""

import numpy as np


def mean_and_spread_demo():
    print("--- Mean, Variance, Std Dev ---")

    marks_class_a = np.array([70, 72, 68, 71, 69])
    marks_class_b = np.array([40, 100, 30, 95, 85])

    print("Class A:", marks_class_a)
    print("Class B:", marks_class_b)

    print("\nMean A:", np.mean(marks_class_a))
    print("Mean B:", np.mean(marks_class_b))
    # same mean, very different spread - that's the whole point of variance

    print("\nVariance A:", np.var(marks_class_a))
    print("Variance B:", np.var(marks_class_b))

    print("\nStd Dev A:", np.std(marks_class_a))
    print("Std Dev B:", np.std(marks_class_b))

    return marks_class_a, marks_class_b


def median_demo(marks_class_a):
    print("\n--- Median ---")
    print("Median of Class A:", np.median(marks_class_a))


def normal_distribution_demo():
    print("\n--- Normal Distribution ---")

    # loc = mean, scale = standard deviation, size = how many random values
    normal_data = np.random.normal(loc=70, scale=5, size=1000)

    print("Sample mean:", np.mean(normal_data))
    print("Sample std dev:", np.std(normal_data))
    # these should come out close to 70 and 5, since that's what we asked for


# ---------------------------------------------------------------------
# Practice task - try this yourself before looking at the solution below
#
# 1. daily_expenses = [500, 300, 700, 450, 600, 800, 350]
#    find mean, median, std dev
# 2. what does a high std dev tell you about spending habits?
# 3. generate 1000 values with np.random.normal(loc=500, scale=100, size=1000)
#    check if the generated mean/std dev roughly match what was asked for
# 4. bonus: try np.percentile(data, 50) and compare it to np.median()
# ---------------------------------------------------------------------

def practice_task():
    print("\n--- Practice Task ---")

    daily_expenses = np.array([500, 300, 700, 450, 600, 800, 350])
    print("daily expenses:", daily_expenses)
    print("mean:", np.mean(daily_expenses))
    print("median:", np.median(daily_expenses))
    print("std dev:", np.std(daily_expenses))

    # a high std dev here would mean spending is inconsistent day to day -
    # some days very low, some days very high, rather than a steady amount

    generated = np.random.normal(loc=500, scale=100, size=1000)
    print("\ngenerated mean:", np.mean(generated))
    print("generated std dev:", np.std(generated))

    print("\nmedian:", np.median(daily_expenses))
    print("50th percentile:", np.percentile(daily_expenses, 50))
    # these two should match - median is just the 50th percentile


if __name__ == "__main__":
    marks_a, marks_b = mean_and_spread_demo()
    median_demo(marks_a)
    normal_distribution_demo()
    practice_task()
