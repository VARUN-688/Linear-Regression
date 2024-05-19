#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def get_variance(func):
        def wrap(a):
            x_mean=np.mean(a)
            temp_x=[t**2 for t in a-x_mean]
            variance_x=np.sum(temp_x)/(len(a))
            s_x=variance_x**0.5
            dic={'Mean':x_mean,'Variance':variance_x,'Standard_Deviation':s_x}
            return dic

        return wrap
@get_variance
def get_standard_deviation(a):
        pass


# In[3]:


def covariance(func):
        def wrap(a,b):
            x=get_standard_deviation(a)
            y=get_standard_deviation(b)
            temp_xy=(np.sum((a-x['Mean'])*(b-y['Mean'])))/(x['Standard_Deviation']*y['Standard_Deviation'])
            coefficient_r=temp_xy/len(a)
            return (coefficient_r,x,y)
        return wrap
@covariance
def get_coefficient(a,b):
        pass
    
def get_eqaution(a,b):
    c=get_coefficient(a,b)
    slope=c[0]*(c[2]['Standard_Deviation']/c[1]['Standard_Deviation'])
    intercept=c[2]['Mean']-c[1]['Mean']*slope
    return f'y={slope}x+{intercept}'


# In[4]:


class LinearRegression:
    def __init(self):
        self.x=[]
        self.y=[]
        self.variance_x=0
        self.variance_y=0
        self.s_x=0
        self.s_y=0
        self.coefficient_r=0

            
    def get_variance(func):
        def wrap(self,a):
            
            x_mean=np.mean(a)
            temp_x=[t**2 for t in a-x_mean]
            variance_x=np.sum(temp_x)/(len(a))
            s_x=variance_x**0.5
            dic={'Mean':x_mean,'Variance':variance_x,'Standard_Deviation':s_x}
            return dic

        return wrap
    
    @get_variance
    def get_standard_deviation(self,a):
        pass
    
    
    def covariance(func):
        def wrap(self):
            a=self.x
            b=self.y
            temp_x=self.get_standard_deviation(a)
            temp_y=self.get_standard_deviation(b)
            self.x_variance_x=temp_x['Variance']
            self.s_x=temp_x['Standard_Deviation']
            self.y_variance=temp_y['Variance']
            self.s_y=temp_y['Standard_Deviation']
            temp_xy=(np.sum((a-temp_x['Mean'])*(b-temp_y['Mean'])))/(temp_x['Standard_Deviation']*temp_y['Standard_Deviation'])
            coefficient_r=temp_xy/len(a)
            return (coefficient_r,temp_x,temp_y)
        return wrap
    
    @covariance
    def get_coefficient(self):
        pass
    
    
    def get_equation(self):
        c=self.get_coefficient()
        self.coefficient_r=c[0]
        self.slope=c[0]*(c[2]['Standard_Deviation']/c[1]['Standard_Deviation'])
        self.intercept=c[2]['Mean']-c[1]['Mean']*self.slope
        return f'y={self.slope}x+{self.intercept}'
    
    
    def fit(self,a,b):
        try:
            self.x=np.array(a)
            self.y=np.array(b)
            self.equation=self.get_equation()
        except:
            raise
    def predict(self,x_test,y_test):
        res=[]
        
        for i in x_test:
            res.append(self.slope*i+self.intercept)
            
        y_predict=res
        
        r_squa=self.r_squared(y_test,y_predict)
        
        n = len(y_test)
        
        rmse = np.sqrt(np.sum((y_test - y_predict)**2) / n)
        
        return f'The model has R-Sqaured = {r_squa} and RMSE={rmse}'
    
    def r_squared(self, y_test, y_predict):
        mean_actual = np.mean(y_test)
        r_squared = 1 - (np.sum((y_test - y_predict)**2) / np.sum((y_test - mean_actual)**2))
        return r_squared

    def __repr__(self):
        return f'The model has the equation as {self.equation}'


# In[5]:


from sklearn.model_selection import train_test_split
model1=LinearRegression()
X = np.random.randint(1, 101, size=(10,))
y = 2 * X + 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model1.fit(X_train,y_train)
print(model1)
print(model1.predict(X_test,y_test))

