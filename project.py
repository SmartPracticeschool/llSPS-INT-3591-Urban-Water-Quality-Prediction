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

print(dataset) 

