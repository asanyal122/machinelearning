
# coding: utf-8

# In[84]:


import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    #slope
    m=0
    #bias
    b=0
    #no of data points
    n=len(x)
    #learning rate
    learning_rate = 0.001
    prev_cost = 0;
    for i in range(100000):
        y_pred = m*x + b
        cost = 1/n * sum((y-y_pred)**2)

        md = -2/n * sum(x*(y-y_pred))
        bd = 2/n * sum(y-y_pred)

        m = m - learning_rate*md
        b = b - learning_rate*bd
        
        if i>0 and cost >= prev_cost:
            break
        prev_cost = cost
        
        
        
    return m,b,min(cost,prev_cost)

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([1,2,3,4,5,6,7,8,9])
#y1 = np.array([1,4,9,16,25,36,49,64,81])

plt.plot(x,y)


m,b,c = gradient_descent(x,y)

y_pred = [m*i+b for i in x]

plt.plot(x,y_pred)
print(c)

#print(x,y,y_pred)



