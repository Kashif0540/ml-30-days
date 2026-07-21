"""
Day 2 — NumPy Deep Dive: Vectorized Operations & Broadcasting
30-Day Machine Learning Challenge — Week 1: Python & Math Foundations
 
Goal:
    Go deeper into NumPy — understand vectorized operations (element-wise math
    without loops) and broadcasting (operating on arrays of different shapes).
    These two ideas are why every ML library can process huge datasets fast.
 
"""

import numpy as np 

# ---------------------------------------------------------------------------
# PART 1: Vectorized Operations — same-shape arrays
# ---------------------------------------------------------------------------
def vect():
    print("Vectorization\n")
    a=np.array([2,4,6,8])
    b=np.array([1,2,3,4])
    
        # Loop way (slow, shown only for comparison — never do this in real NumPy code)
    result_loop =[]
    for i in range(len(a)):
        result_loop.append(int(a[i]+b[i]))
        
    
    print("a",a)    
    print("b",b)    
    print("Result_sum:",result_loop)
    print("add",a+b)
    print("div",a*b)
        # Comparison operators are vectorized too — useful for boolean masking
    print(a>5)
    
    print("max a:",max(a))
    print("max b:",max(b))
    print("sum a:",sum(a))
    print("min:",min(a))
        
vect()

# ---------------------------------------------------------------------------
# PART 2: Broadcasting — different-shape arrays
# ---------------------------------------------------------------------------
def broadcast():
    print("\n\nBroadcasting\n")
    marks=np.array([60,75,85,90,95])
    marks_with_bonus=marks+5
    print("Marks:",marks_with_bonus)
    
    prices=np.array([[100,200,300],
    [150,250,350]])
    discount=np.array([10,20,30])
    
    final_prices=prices - discount
    print("\nFinal prices after broadcasting discount\n",final_prices)
    
broadcast()


# ---------------------------------------------------------------------------
# PRACTICE TASKS (try to solve these yourself BEFORE reading the solutions)
#
# Task 1:
#   - marks1 = [70, 80, 90, 60, 75], marks2 = [65, 85, 88, 70, 80]
#   - Use a vectorized operation to compute each student's average of the two.
#   - Use np.mean() to find the overall class average across both arrays.
#   - Use vectorized comparison to find which students scored above 75 average.
#
# Task 2 (broadcasting):
#   - temperatures = [30, 32, 28, 35, 31] in Celsius
#   - Convert all of them to Fahrenheit using broadcasting: F = C * 9/5 + 32
#
# Task 3 (broadcasting):
#   - sales = [[100, 200], [150, 250], [300, 100]]  (3 days, 2 products)
#   - tax = [5, 10]  (flat tax amount per product)
#   - Subtract tax from sales using broadcasting.
# ---------------------------------------------------------------------------
 
#Task
def Practice_Task():
    print("\nPractice_Tasks\n")
    #Task1
    marks1=np.array([70, 80, 90, 60, 75])
    marks2=np.array([65, 85, 88, 70, 80])
    
    student_avg=((marks1+marks2)/2)
    print("Per student avg:",student_avg)
    
    print("avg_marks1:",np.mean(marks1))
    print("avg_marks2:",np.mean(marks2))
    
    print(">75avg:",student_avg>75)
    
    #Task2
    temp=np.array([30, 32, 28, 35, 31])
    print("\nTemp in Fahrenheit:",(temp * 9/5 + 32))

    #Task3
    sales = np.array([[100, 200], [150, 250], [300, 100]])
    tax = np.array([5, 10])
    profit=sales -  tax
    print("\nSales after tax:\n",profit)
    
Practice_Task()