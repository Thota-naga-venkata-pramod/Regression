import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
x=pd.read_csv("C:\\Users\\HP\\OneDrive\\Documents\\singlelinear.csv")
print(x)
x = x.rename(columns={'area ': 'area'})
plt.scatter(x.area,x.price)
reg=linear_model.LinearRegression()
reg.fit(x[['area']],x.price)
print(reg.predict([[3300]]))#where x is 3300 and y is the output(predicted value)
print(reg.coef_)#in y=mx+c this is 'm'
print(reg.intercept_)#in y=mx+c this is 'c'