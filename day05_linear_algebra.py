
"""
Day 5 - Linear Algebra Basics
30 Day ML Challenge, Week 1

Covering: matrices, dot product, matrix multiplication, transpose, eigenvalues.
This is the math NumPy has been quietly using under the hood since Day 1.
"""

import numpy as np

def matrix():
    print("\n               MUHAMMAD KASHIF(AI/ML day5->linear algebra basics)       \n")
    print("\n---Matrix----\n")
    
    A =np.array([[1,2],
    [3,4]])
    B =np.array([[5,6],
    [7,8]])
    
    print("A:\n",A)
    print("B:\n",B)
    
    print("Shape of A:",A.shape)

    return A,B

#storing the returned values for lated use in other functions   
A,B=matrix()

def dot_product():
    print("\n\n-----DOT PRODUCT-----\n")
    
    quantity=np.array([12,4,6])
    prices=np.array([100,225,450])
    
    total=np.dot(quantity,prices)
    
    print("QUANTITY:",quantity)
    print("PRICES:",prices)
    print("TOTAL:",total)
    
dot_product()

def matrix_multiplication(A,B):
    print("\n\n-----MATRIX MULTIPLICATION------\n")
    
    result=np.dot(A,B)
    print("A DOT B:",result)
    
    # the operator @ does same multiplication thing
    
    result1=A @ B
    print("\n A @ B:",result1)

#this is how matrix multiplication happens
#[1×5 + 2×7    1×6 + 2×8]
#[3×5 + 4×7    3×6 + 4×8]    
matrix_multiplication(A,B)

def transpose():
    print("\n\n----TRANSPOSE----\n")
    
    print("A:",A)
    print("\nA.T:",A.T)
    
transpose()

def eigen():
    print("\n\n-----EIGEN VALUES AND EIGEN VECTORS------\n")
    
    eigenvalues,eigenvectors=np.linalg.eig(A)
    print("EIGEN VALUES:",eigenvalues)
    print("EIGEN VECTORS:",eigenvectors)
    
eigen()

# ---------------------------------------------------------------------
#
# 1. marks = [80, 90, 70], weightage = [0.3, 0.5, 0.2]
#    find the weighted total score using a dot product
# 2. make any 2x2 matrix, find its transpose
# 3. find the eigenvalues of that same matrix
# 4. bonus: multiply two 2x2 matrices using @
# ---------------------------------------------------------------------

def practice_task():
    print("\n\n-----PRACTICE TASK------\n")
    
    marks=([80,90,70])
    weightage=([0.3,0.5,0.2])
    
    #1.dot product
    total_score=np.dot(marks,weightage)
    print("TOTAL SCORE:",total_score)
    
    #2.2x2 and transpose of B
    A=np.array([[15,16],
    [17,18]])
    B=np.array([[19,20],
    [21,22]])
    
    print("\nA:",A)
    print("\nB:",B)
    print("\nTranspose B:",B.T)
    
    #3.eigenvalues of matrix B
    eigenvalues,eigenvectors=np.linalg.eig(B)
    print("\nEIGEN VALUES:",eigenvalues)
    print("\nEIGEN VECOTRS:",eigenvectors)
    
    #4.A@B
    multiply=A @ B
    print("\nA @ B:",multiply)
    
practice_task()
