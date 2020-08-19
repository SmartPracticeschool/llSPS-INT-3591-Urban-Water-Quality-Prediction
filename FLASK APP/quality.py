#import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Read the dataset/Data Preprocessing
#IMPORTING DATASET
dataset= pd.read_csv("datasets_67983_135810_water_dataX.csv", encoding= 'unicode_escape')
print(dataset)
#TAKING CARE OF MISSING DATA
#dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
#dataset.fillna(dataset.mean(), inplace=True)

#CONVERTION OF DATATYPES FOR BETTER CALCULATION
dataset['Temp']=pd.to_numeric(dataset['Temp'],errors='coerce')
dataset['D.O. (mg/l)']=pd.to_numeric(dataset['D.O. (mg/l)'],errors='coerce')
dataset['PH']=pd.to_numeric(dataset['PH'],errors='coerce')
dataset['B.O.D. (mg/l)']=pd.to_numeric(dataset['B.O.D. (mg/l)'],errors='coerce')
dataset['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(dataset['CONDUCTIVITY (µmhos/cm)'],errors='coerce')
dataset['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(dataset['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
dataset['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(dataset['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')

#LABEL ENCODING
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#le = LabelEncoder()
#dataset['STATE']=le.fit_transform(dataset['STATE'])
#dataset['LOCATIONS']=le.fit_transform(dataset['LOCATIONS'])
#print(dataset['STATE'])
#print(dataset['LOCATIONS'])


#Splitting Dataset in to Independent variable and Dependent variable
start=0
end=1992
#station=dataset.iloc [start:end ,0]
#location=dataset.iloc [start:end ,1]
#state=dataset.iloc [start:end ,2]
do= dataset.iloc [start:end ,4].astype(np.float64)

ph = dataset.iloc[ start:end,5]  
co = dataset.iloc [start:end ,6].astype(np.float64)   
  
year=dataset.iloc[start:end,11]
tc=dataset.iloc [2:end ,10].astype(np.float64)

bod = dataset.iloc [start:end ,7].astype(np.float64)
na= dataset.iloc [start:end ,8].astype(np.float64)

#print(location.shape)
#location.reshape(-1,1)
#One hot encoding
#ohe=OneHotEncoder()
#z= ohe.fit_transform(location).toarray()
#location = np.delete(location,axis=1)
#location=np.concatenate((z,location),axiz=1)
print(dataset)


dataset=pd.concat([do,ph,co,bod,na,tc,year],axis=1)
dataset. columns = ['do','ph','co','bod','na','tc','year']

print(dataset)

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
print(dataset)

#calculation overall wqi for each year
#ag=dataset.groupby('year')['wqi'].mean()
#print(ag)

#Data Filteration and Vizualization
#dataset=ag.reset_index(level=0,inplace=False)
#year=dataset['year'].values
#AQI=dataset['wqi'].values
#dataset['wqi']=pd.to_numeric(dataset['wqi'],errors='coerce')
#dataset['year']=pd.to_numeric(dataset['year'],errors='coerce')
#print(dataset)

#Training and Testing the Model
from sklearn import neighbors,datasets
dataset=dataset.reset_index(level=0,inplace=False)
from sklearn import linear_model
from sklearn.model_selection import train_test_split
cols =['do','ph','co','bod','na','tc']
cols2=['wqi']
y = dataset[cols2]
x=dataset[cols]

reg=linear_model.LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(np.nan_to_num(x),y,test_size=0.2,random_state=4)

reg.fit(x_train,y_train)
y_predict=reg.predict(x_test)
print(y_predict)

print(reg.predict([[0,0,110,0,0.1,6]]))

#np.save("quality.npy",le.classes_)
#print(le.classes_)

import pickle
pickle.dump(reg,open('quality.pkl','wb'))
model=pickle.load(open('quality.pkl','rb'))











