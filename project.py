import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATASET
dataset= pd.read_csv("datasets_67983_135810_water_dataX.csv", encoding= 'unicode_escape')

#TAKING CARE OF MISSING DATA
dataset.fillna(0, inplace=True)

#CONVERTION OF DATATYPES FOR BETTER CALCULATION
dataset['Temp']=pd.to_numeric(dataset['Temp'],errors='coerce')
dataset['D.O. (mg/l)']=pd.to_numeric(dataset['D.O. (mg/l)'],errors='coerce')
dataset['PH']=pd.to_numeric(dataset['PH'],errors='coerce')
dataset['B.O.D. (mg/l)']=pd.to_numeric(dataset['B.O.D. (mg/l)'],errors='coerce')
dataset['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(dataset['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
dataset['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(dataset['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
dataset['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(dataset['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
print(dataset.dtypes)

#LABEL ENCODING
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['STATE']=le.fit_transform(dataset['STATE'])
dataset['LOCATIONS']=le.fit_transform(dataset['LOCATIONS'])

#Splitting Dataset in to Independent variable and Dependent variable
start=2
end=1992
station=dataset.iloc [start:end ,0]
location=dataset.iloc [start:end ,1]
state=dataset.iloc [start:end ,2]
do= dataset.iloc [start:end ,4].astype(np.float64)
value=0
ph = dataset.iloc[ start:end,5]  
co = dataset.iloc [start:end ,6].astype(np.float64)   
  
year=dataset.iloc[start:end,11]
tc=dataset.iloc [2:end ,10].astype(np.float64)

bod = dataset.iloc [start:end ,7].astype(np.float64)
na= dataset.iloc [start:end ,8].astype(np.float64)

dataset=pd.concat([station,location,state,do,ph,co,bod,na,tc,year],axis=1)
dataset. columns = ['station','location','state','do','ph','co','bod','na','tc','year']

#DATA VIZUALIZATION
plt.bar(year,do)
plt.xlabel('YEAR')
plt.ylabel('D.O.')
#plt.scatter(year,ph)

#plt.plot(year,co)

#plt.plot(year,bod)

#plt.plot(year,na)

#plt.plot(year,tc)
plt.show()

#Calculation of all the data in dataset for finding WATER QUALITY INDEX(WQI)
#calulation of Ph
dataset['npH']=dataset.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))
#calculation of dissolved oxygen
dataset['ndo']=dataset.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))

#calculation of total coliform
dataset['nco']=dataset.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))
#calculation of B.D.O
dataset['nbdo']=dataset.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))
#calculation of electrical conductivity
dataset['nec']=dataset.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))

#Calulation of nitrate
dataset['nna']=dataset.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))
#Calculation of Water Quality Index(wqi)
dataset['wph']=dataset.npH * 0.165
dataset['wdo']=dataset.ndo * 0.281
dataset['wbdo']=dataset.nbdo * 0.234
dataset['wec']=dataset.nec* 0.009
dataset['wna']=dataset.nna * 0.028
dataset['wco']=dataset.nco * 0.281
dataset['wqi']=dataset.wph+dataset.wdo+dataset.wbdo+dataset.wec+dataset.wna+dataset.wco
#print(dataset)

#calculation overall wqi for each year
ag=dataset.groupby('year')['wqi'].mean()
#print(ag.head())

#Data Filteration and Vizualization
dataset=ag.reset_index(level=0,inplace=False)
year=dataset['year'].values
AQI=dataset['wqi'].values
dataset['wqi']=pd.to_numeric(dataset['wqi'],errors='coerce')
dataset['year']=pd.to_numeric(dataset['year'],errors='coerce')

plt.rcParams['figure.figsize'] = (20.0, 10.0)
#3D Scatter plot of data points
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(year,AQI, color='red')
plt.show()
#print(dataset)

#scatter plot of data points
dataset = dataset[np.isfinite(dataset['wqi'])]
cols =['year']
y = dataset['wqi']
x=dataset[cols]

plt.scatter(x,y)
plt.show()
#print(dataset)

#Line plot of data point
dataset=dataset.set_index('year')
dataset.plot(figsize=(15,6))
plt.show()

#Training and Testing the Model
from sklearn import neighbors,datasets
dataset=dataset.reset_index(level=0,inplace=False)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
cols =['year']

y = dataset['wqi']
x=dataset[cols]
reg=linear_model.LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
reg.fit(x_train,y_train)
a=reg.predict(x_test)
#print(a)
#print(y_test)
from sklearn.metrics import mean_squared_error
print('mse:%.2f'%mean_squared_error(y_test,a))

dt = pd.DataFrame({'Actual': y_test, 'Predicted': a})
#using gradient descent to optimize it further
x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x]
#print(x)
alpha = 0.1 #Step size
iterations = 3000 #No. of iterations
m = y.size #No. of data points
np.random.seed(4) #Setting the seed
theta = np.random.rand(2) #Picking some random values to start with

def gradient_descent(x, y, theta, iterations, alpha):
    past_costs = []
    past_thetas = [theta]
    for i in range(iterations):
        prediction = np.dot(x, theta)
        error = prediction - y
        cost = 1/(2*m) * np.dot(error.T, error)
        past_costs.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))
        past_thetas.append(theta)
        
    return past_thetas, past_costs

past_thetas, past_costs = gradient_descent(x, y, theta, iterations, alpha)
theta = past_thetas[-1]

print("Gradient Descent: {:.2f}, {:.2f}".format(theta[0], theta[1]))

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost')
plt.plot(past_costs)
plt.show()

#Prediction of january(2013-2015) across india
import numpy as np
newB=[74.76, 2.13]

def rmse(y,y_pred):
    rmse= np.sqrt(sum(y-y_pred))
    return rmse
   

y_pred=x.dot(newB)

dt = pd.DataFrame({'Actual': y, 'Predicted': y_pred})  
dt=pd.concat([dataset, dt], axis=1)
#print(dt)

#testing the accuracy of the model

from sklearn import metrics
print("Accuracy: ")
print(np.sqrt(metrics.mean_squared_error(y,y_pred)))


#plotting the actual and predicted results
x_axis=dt.year
y_axis=dt.Actual
y1_axis=dt.Predicted
plt.scatter(x_axis,y_axis)
plt.plot(x_axis,y1_axis,color='r')
plt.title("linear regression")
plt.show()

import pickle
pickle.dump(reg,open('project.pkl','wb'))
model=pickle.load(open('project.pkl','rb'))
