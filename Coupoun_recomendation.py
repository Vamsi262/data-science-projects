#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import log_loss,roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import pickle
from prettytable import PrettyTable
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('in-vehicle-coupon-recommendation_final.csv')
data.head()


# In[3]:


print("Data points:", data.shape[0])
print("Features:", data.shape[1])
print("Data Attributes :", data.columns.values)


# In[4]:


Y_value_counts = data.groupby('Y').Y.count()
print('Coupons Accepted by Users :',Y_value_counts[1],',',round(Y_value_counts[1]/data.shape[0]*100,3),'%')
print('Coupons Rejected by Users :',Y_value_counts[0],',',round(Y_value_counts[0]/data.shape[0]*100,3),'%')


# # Train & Test Split

# In[5]:


X = data.drop(['Y'], axis=1)
y = data['Y'].values


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y)


# In[7]:


print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[8]:


X_train.info()


# In[9]:


print('Missing value Ture or False?',data.isnull().values.any())
missing_percentage = data.isnull().sum()*100/len(data)
missing_value_df = pd.DataFrame({'missing_count': data.isnull().sum(),'missing_percentage': missing_percentage})
missing_value_df[missing_value_df.missing_count != 0]


# In[10]:


X_train = X_train.drop(['car'], axis=1)
X_test = X_test.drop(['car'], axis=1)


# In[11]:


X_train.corr()


# In[12]:


X_train = X_train.drop(['direction_opp','toCoupon_GEQ5min'], axis=1)
X_test = X_test.drop(['direction_opp','toCoupon_GEQ5min'], axis=1)
X_train.shape, X_test.shape


# In[13]:


X_train.describe()


# In[14]:


print('Missing value True or False:',X_train.isnull().values.any())


# In[15]:


X_train['Bar'] = X_train['Bar'].fillna(X_train['Bar'].value_counts().index[0])
X_train['CoffeeHouse'] = X_train['CoffeeHouse'].fillna(X_train['CoffeeHouse'].value_counts().index[0])
X_train['CarryAway'] = X_train['CarryAway'].fillna(X_train['CarryAway'].value_counts().index[0])
X_train['RestaurantLessThan20'] = X_train['RestaurantLessThan20'].fillna(X_train['RestaurantLessThan20'].value_counts().index[0])
X_train['Restaurant20To50'] = X_train['Restaurant20To50'].fillna(X_train['Restaurant20To50'].value_counts().index[0])


# In[16]:


X_test['Bar'] = X_test['Bar'].fillna(X_train['Bar'].value_counts().index[0])
X_test['CoffeeHouse'] = X_test['CoffeeHouse'].fillna(X_train['CoffeeHouse'].value_counts().index[0])
X_test['CarryAway'] = X_test['CarryAway'].fillna(X_train['CarryAway'].value_counts().index[0])
X_test['RestaurantLessThan20'] = X_test['RestaurantLessThan20'].fillna(X_train['RestaurantLessThan20'].value_counts().index[0])
X_test['Restaurant20To50'] = X_test['Restaurant20To50'].fillna(X_train['Restaurant20To50'].value_counts().index[0])


# In[17]:


print('Missing value True or False:',X_train.isnull().values.any())


# In[18]:


print('Missing value True or False:',X_test.isnull().values.any())


# In[19]:


to_Coupon = []
for i in range(X_train.shape[0]):
    if (list(X_train['toCoupon_GEQ15min'])[i] == 0):
        to_Coupon.append(0)
    elif (list(X_train['toCoupon_GEQ15min'])[i] == 1)and(list(X_train['toCoupon_GEQ25min'])[i] == 0):
        to_Coupon.append(1)
    else:
        to_Coupon.append(2)
        
X_train['to_Coupon'] = to_Coupon
print('Unique Values:',X_train['to_Coupon'].unique())
X_train['to_Coupon'].describe()


# In[20]:


to_Coupon = []
for i in range(X_test.shape[0]):
    if (list(X_test['toCoupon_GEQ15min'])[i] == 0):
        to_Coupon.append(0)
    elif (list(X_test['toCoupon_GEQ15min'])[i] == 1)and(list(X_test['toCoupon_GEQ25min'])[i] == 0):
        to_Coupon.append(1)
    else:
        to_Coupon.append(2)
        
X_test['to_Coupon'] = to_Coupon
print('Unique values:',X_test['to_Coupon'].unique())
X_test['to_Coupon'].describe()


# In[21]:


coupon_freq = []
for i in range(X_train.shape[0]):
    if (list(X_train['coupon'])[i] == 'Restaurant(<20)'):
        coupon_freq.append(list(X_train['RestaurantLessThan20'])[i])
    elif (list(X_train['coupon'])[i] == 'Coffee House'):
        coupon_freq.append(list(X_train['CoffeeHouse'])[i])
    elif (list(X_train['coupon'])[i] == 'Carry out & Take away'):
        coupon_freq.append(list(X_train['CarryAway'])[i])
    elif (list(X_train['coupon'])[i] == 'Bar'):
        coupon_freq.append(list(X_train['Bar'])[i])
    elif (list(X_train['coupon'])[i] == 'Restaurant(20-50)'):
        coupon_freq.append(list(X_train['Restaurant20To50'])[i])
        
X_train['coupon_freq'] = coupon_freq
print('Unique values:',X_train['coupon_freq'].unique())
X_train['coupon_freq'].describe()


# In[22]:


coupon_freq = []
for i in range(X_test.shape[0]):
    if (list(X_test['coupon'])[i] == 'Restaurant(<20)'):
        coupon_freq.append(list(X_test['RestaurantLessThan20'])[i])
    elif (list(X_test['coupon'])[i] == 'Coffee House'):
        coupon_freq.append(list(X_test['CoffeeHouse'])[i])
    elif (list(X_test['coupon'])[i] == 'Carry out & Take away'):
        coupon_freq.append(list(X_test['CarryAway'])[i])
    elif (list(X_test['coupon'])[i] == 'Bar'):
        coupon_freq.append(list(X_test['Bar'])[i])
    elif (list(X_test['coupon'])[i] == 'Restaurant(20-50)'):
        coupon_freq.append(list(X_test['Restaurant20To50'])[i])
        
X_test['coupon_freq'] = coupon_freq
print('Unique values:',X_test['coupon_freq'].unique()) 
X_test['coupon_freq'].describe()


# In[23]:


X_train['occupation'].describe()


# In[24]:


occupation_dict = {'Healthcare Support':'High_Acceptance','Construction & Extraction':'High_Acceptance','Healthcare Practitioners & Technical':'High_Acceptance',
                   'Protective Service':'High_Acceptance','Architecture & Engineering':'High_Acceptance','Production Occupations':'Medium_High_Acceptance',
                    'Student':'Medium_High_Acceptance','Office & Administrative Support':'Medium_High_Acceptance','Transportation & Material Moving':'Medium_High_Acceptance',
                    'Building & Grounds Cleaning & Maintenance':'Medium_High_Acceptance','Management':'Medium_Acceptance','Food Preparation & Serving Related':'Medium_Acceptance',
                   'Life Physical Social Science':'Medium_Acceptance','Business & Financial':'Medium_Acceptance','Computer & Mathematical':'Medium_Acceptance',
                    'Sales & Related':'Medium_Low_Acceptance','Personal Care & Service':'Medium_Low_Acceptance','Unemployed':'Medium_Low_Acceptance',
                   'Farming Fishing & Forestry':'Medium_Low_Acceptance','Installation Maintenance & Repair':'Medium_Low_Acceptance','Education&Training&Library':'Low_Acceptance',
                    'Arts Design Entertainment Sports & Media':'Low_Acceptance','Community & Social Services':'Low_Acceptance','Legal':'Low_Acceptance','Retired':'Low_Acceptance'}
X_train['occupation_class'] = X_train['occupation'].map(occupation_dict)
print('Unique values:',X_train['occupation_class'].unique())
X_train['occupation_class'].describe()


# In[25]:


X_test['occupation'].describe()


# In[26]:


occupation_dict = {'Healthcare Support':'High_Acceptance','Construction & Extraction':'High_Acceptance','Healthcare Practitioners & Technical':'High_Acceptance',
                   'Protective Service':'High_Acceptance','Architecture & Engineering':'High_Acceptance','Production Occupations':'Medium_High_Acceptance',
                    'Student':'Medium_High_Acceptance','Office & Administrative Support':'Medium_High_Acceptance','Transportation & Material Moving':'Medium_High_Acceptance',
                    'Building & Grounds Cleaning & Maintenance':'Medium_High_Acceptance','Management':'Medium_Acceptance','Food Preparation & Serving Related':'Medium_Acceptance',
                   'Life Physical Social Science':'Medium_Acceptance','Business & Financial':'Medium_Acceptance','Computer & Mathematical':'Medium_Acceptance',
                    'Sales & Related':'Medium_Low_Acceptance','Personal Care & Service':'Medium_Low_Acceptance','Unemployed':'Medium_Low_Acceptance',
                   'Farming Fishing & Forestry':'Medium_Low_Acceptance','Installation Maintenance & Repair':'Medium_Low_Acceptance','Education&Training&Library':'Low_Acceptance',
                    'Arts Design Entertainment Sports & Media':'Low_Acceptance','Community & Social Services':'Low_Acceptance','Legal':'Low_Acceptance','Retired':'Low_Acceptance'}
X_test['occupation_class'] = X_test['occupation'].map(occupation_dict)
print('Unique values:',X_test['occupation_class'].unique())
X_test['occupation_class'].describe()


# In[27]:


X_train = X_train.drop(['occupation'], axis=1)
X_test = X_test.drop(['occupation'], axis=1)
print('X_Train:',X_train.shape,'\nX_Test:',X_test.shape)
print(X_train.columns.values)


# In[28]:


order = [['Work','Home','No Urgent Place'],['Kid(s)','Alone','Partner','Friend(s)'],['Rainy','Snowy','Sunny'],[30,55,80],['7AM','10AM','2PM','6PM','10PM'],
         ['Bar','Restaurant(20-50)','Coffee House','Restaurant(<20)','Carry out & Take away'],['2h','1d'],['Female','Male'],['below21','21','26','31','36','41','46','50plus'],
         ['Widowed','Divorced','Married partner','Unmarried partner','Single'],[0,1],
         ['Some High School','High School Graduate','Some college - no degree','Associates degree','Bachelors degree','Graduate degree (Masters or Doctorate)'],
         ['Less than $12500','$12500 - $24999','$25000 - $37499','$37500 - $49999','$50000 - $62499','$62500 - $74999','$75000 - $87499','$87500 - $99999','$100000 or More'],
         ['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],['never','less1','1~3','4~8','gt8'],
         [0,1],[0,1],[0,1],[0,1,2],['never','less1','1~3','4~8','gt8'],['Low_Acceptance','Medium_Low_Acceptance','Medium_Acceptance','Medium_High_Acceptance','High_Acceptance']]


# In[29]:


Ordinal_enc = OrdinalEncoder(categories=order)
X_train_Ordinal_encoding = Ordinal_enc.fit_transform(X_train)
X_train_Ordinal_encoding = pd.DataFrame(X_train_Ordinal_encoding,columns=X_train.columns.values)
print('X_Train_Ordinal_Encoding:',X_train_Ordinal_encoding.shape)


# In[30]:


Ordinal_enc = OrdinalEncoder(categories=order)
X_test_Ordinal_encoding = Ordinal_enc.fit_transform(X_test)
X_test_Ordinal_encoding = pd.DataFrame(X_test_Ordinal_encoding,columns=X_test.columns.values)
print('X_Test_Ordinal_Encoding:',X_test_Ordinal_encoding.shape)


# In[31]:


def frequency_enc(column_name,X):
    return X[column_name].map(X.groupby(column_name).size()/len(X))


# In[32]:


X_train_frequency_encoding = pd.DataFrame()
for i in range(X_train.shape[1]):
    X_train_frequency_encoding[X_train.columns.values[i]+'_freq_enc'] = frequency_enc(X_train.columns.values[i],X_train)

print('X_Train_Frequency_Encoding:',X_train_frequency_encoding.shape)


# In[33]:


X_test_frequency_encoding = pd.DataFrame()
for i in range(X_test.shape[1]):
    X_test_frequency_encoding[X_test.columns.values[i]+'_freq_enc'] = frequency_enc(X_test.columns.values[i],X_test)

print('X_Test_Frequency_Encoding:',X_test_frequency_encoding.shape)


# In[34]:


def target_enc(column_name,X):
    X['Y_train'] = y_train
    return X[column_name].map(X.groupby(column_name)['Y_train'].mean())

X_train_target_encoding = pd.DataFrame()
for i in range(X_train.shape[1]):
    X_train_target_encoding[X_train.columns.values[i]+'_target_enc'] = target_enc(X_train.columns.values[i],X_train)

print('X_Train_Target_Encoding:',X_train_target_encoding.shape)


# In[35]:


def target_enc(column_name,X):
    X['Y_test'] = y_test
    return X[column_name].map(X.groupby(column_name)['Y_test'].mean())

X_test_target_encoding = pd.DataFrame()
for i in range(X_test.shape[1]):
    X_test_target_encoding[X_test.columns.values[i]+'_target_enc'] = target_enc(X_test.columns.values[i],X_test)

print('X_Test_Target_Encoding:',X_test_target_encoding.shape)


# In[36]:


def response_coding(feature,X,Y):
    X[feature] = X[feature].str.replace('~','_')
    X[feature] = X[feature].str.replace('[^a-zA-Z0-9_ ]',' ')
    X[feature] = X[feature].str.replace(' +',' ')
    X[feature] = X[feature].str.strip()
    X[feature] = X[feature].str.replace(' ','_')
    X[feature] = X[feature].str.lower()
    response_code_0 = [];response_code_1 = []
    unique_cat_features = X[feature].unique()
    unique_cat_features = np.sort(unique_cat_features)
    for i in range(len(unique_cat_features)):
        total_count = X[feature][(X[feature] == unique_cat_features[i])].count()
        p0 = (X[feature][((X[feature] == unique_cat_features[i]) & (Y==0))].count())/total_count
        p1 = (X[feature][((X[feature] == unique_cat_features[i]) & (Y==1))].count())/total_count
        response_code_0.append(p0);response_code_1.append(p1)
    dict_response_code_0 = dict(zip(unique_cat_features, response_code_0))
    dict_response_code_1 = dict(zip(unique_cat_features, response_code_1))
    X_response_0 = X[feature].map(dict_response_code_0)
    X_response_1 = X[feature].map(dict_response_code_1)
    X_response_0 = X_response_0.values.reshape(-1,1)
    X_response_1 = X_response_1.values.reshape(-1,1)
    return X_response_0,X_response_1


# In[37]:


X_train_destination_0,X_train_destination_1 = response_coding('destination',X_train,y_train)
X_train_passanger_0,X_train_passanger_1 = response_coding('passanger',X_train,y_train)
X_train_weather_0,X_train_weather_1 = response_coding('weather',X_train,y_train)
X_train_time_0,X_train_time_1 = response_coding('time',X_train,y_train)
X_train_coupon_0,X_train_coupon_1 = response_coding('coupon',X_train,y_train)
X_train_expiration_0,X_train_expiration_1 = response_coding('expiration',X_train,y_train)
X_train_gender_0,X_train_gender_1 = response_coding('gender',X_train,y_train)
X_train_age_0,X_train_age_1 = response_coding('age',X_train,y_train)
X_train_maritalStatus_0,X_train_maritalStatus_1 = response_coding('maritalStatus',X_train,y_train)
X_train_education_0,X_train_education_1 = response_coding('education',X_train,y_train)
X_train_income_0,X_train_income_1 = response_coding('income',X_train,y_train)
X_train_Bar_0,X_train_Bar_1 = response_coding('Bar',X_train,y_train)
X_train_CoffeeHouse_0,X_train_CoffeeHouse_1 = response_coding('CoffeeHouse',X_train,y_train)
X_train_CarryAway_0,X_train_CarryAway_1 = response_coding('CarryAway',X_train,y_train)
X_train_RestaurantLessThan20_0,X_train_RestaurantLessThan20_1 = response_coding('RestaurantLessThan20',X_train,y_train)
X_train_Restaurant20To50_0,X_train_Restaurant20To50_1 = response_coding('Restaurant20To50',X_train,y_train)
X_train_coupon_freq_0,X_train_coupon_freq_1 = response_coding('coupon_freq',X_train,y_train)
X_train_occupation_class_0,X_train_occupation_class_1 = response_coding('occupation_class',X_train,y_train)

X_test_destination_0,X_test_destination_1 = response_coding('destination',X_test,y_test)
X_test_passanger_0,X_test_passanger_1 = response_coding('passanger',X_test,y_test)
X_test_weather_0,X_test_weather_1 = response_coding('weather',X_test,y_test)
X_test_time_0,X_test_time_1 = response_coding('time',X_test,y_test)
X_test_coupon_0,X_test_coupon_1 = response_coding('coupon',X_test,y_test)
X_test_expiration_0,X_test_expiration_1 = response_coding('expiration',X_test,y_test)
X_test_gender_0,X_test_gender_1 = response_coding('gender',X_test,y_test)
X_test_age_0,X_test_age_1 = response_coding('age',X_test,y_test)
X_test_maritalStatus_0,X_test_maritalStatus_1 = response_coding('maritalStatus',X_test,y_test)
X_test_education_0,X_test_education_1 = response_coding('education',X_test,y_test)
X_test_income_0,X_test_income_1 = response_coding('income',X_test,y_test)
X_test_Bar_0,X_test_Bar_1 = response_coding('Bar',X_test,y_test)
X_test_CoffeeHouse_0,X_test_CoffeeHouse_1 = response_coding('CoffeeHouse',X_test,y_test)
X_test_CarryAway_0,X_test_CarryAway_1 = response_coding('CarryAway',X_test,y_test)
X_test_RestaurantLessThan20_0,X_test_RestaurantLessThan20_1 = response_coding('RestaurantLessThan20',X_test,y_test)
X_test_Restaurant20To50_0,X_test_Restaurant20To50_1 = response_coding('Restaurant20To50',X_test,y_test)
X_test_coupon_freq_0,X_test_coupon_freq_1 = response_coding('coupon_freq',X_test,y_test)
X_test_occupation_class_0,X_test_occupation_class_1 = response_coding('occupation_class',X_test,y_test)


# In[38]:


def norm(column_name,X):
    normalizer = Normalizer()
    normalizer.fit(X[column_name].values.reshape(1,-1))
    X_norm = normalizer.transform(X[column_name].values.reshape(1,-1))
    return X_norm.reshape(-1,1)


# In[39]:


X_train_temperature_norm = norm('temperature',X_train)
X_train_has_children_norm = norm('has_children',X_train)
X_train_toCoupon_GEQ15min_norm = norm('toCoupon_GEQ15min',X_train)
X_train_toCoupon_GEQ25min_norm = norm('toCoupon_GEQ25min',X_train)
X_train_direction_same_norm = norm('direction_same',X_train)
X_train_to_Coupon_norm = norm('to_Coupon',X_train)

X_test_temperature_norm = norm('temperature',X_test)
X_test_has_children_norm = norm('has_children',X_test)
X_test_toCoupon_GEQ15min_norm = norm('toCoupon_GEQ15min',X_test)
X_test_toCoupon_GEQ25min_norm = norm('toCoupon_GEQ25min',X_test)
X_test_direction_same_norm = norm('direction_same',X_test)
X_test_to_Coupon_norm = norm('to_Coupon',X_test)


# In[40]:


X_train_response_encoding = np.hstack((X_train_destination_0,X_train_destination_1,X_train_passanger_0,X_train_passanger_1,X_train_weather_0,X_train_weather_1,X_train_time_0,X_train_time_1,X_train_coupon_0,X_train_coupon_1,X_train_expiration_0,X_train_expiration_1,X_train_gender_0,X_train_gender_1,X_train_age_0,X_train_age_1,X_train_maritalStatus_0,X_train_maritalStatus_1,X_train_education_0,X_train_education_1,X_train_income_0,X_train_income_1,X_train_coupon_freq_0,X_train_coupon_freq_1,X_train_occupation_class_0,X_train_occupation_class_1,X_train_Bar_0,X_train_Bar_1,X_train_CoffeeHouse_0,X_train_CoffeeHouse_1,X_train_CarryAway_0,X_train_CarryAway_1,X_train_RestaurantLessThan20_0,X_train_RestaurantLessThan20_1,X_train_Restaurant20To50_0,X_train_Restaurant20To50_1,X_train_temperature_norm,X_train_has_children_norm,X_train_toCoupon_GEQ15min_norm,X_train_toCoupon_GEQ25min_norm,X_train_direction_same_norm,X_train_to_Coupon_norm))
X_test_response_encoding = np.hstack((X_test_destination_0,X_test_destination_1,X_test_passanger_0,X_test_passanger_1,X_test_weather_0,X_test_weather_1,X_test_time_0,X_test_time_1,X_test_coupon_0,X_test_coupon_1,X_test_expiration_0,X_test_expiration_1,X_test_gender_0,X_test_gender_1,X_test_age_0,X_test_age_1,X_test_maritalStatus_0,X_test_maritalStatus_1,X_test_education_0,X_test_education_1,X_test_income_0,X_test_income_1,X_test_coupon_freq_0,X_test_coupon_freq_1,X_test_occupation_class_0,X_test_occupation_class_1,X_test_Bar_0,X_test_Bar_1,X_test_CoffeeHouse_0,X_test_CoffeeHouse_1,X_test_CarryAway_0,X_test_CarryAway_1,X_test_RestaurantLessThan20_0,X_test_RestaurantLessThan20_1,X_test_Restaurant20To50_0,X_test_Restaurant20To50_1 ,X_test_temperature_norm,X_test_has_children_norm,X_test_toCoupon_GEQ15min_norm,X_test_toCoupon_GEQ25min_norm,X_test_direction_same_norm,X_test_to_Coupon_norm))
print('X_Train_Response_Encoding:',X_train_response_encoding.shape,'\nX_Test_Response_Encoding:',X_test_response_encoding.shape)


# In[41]:


def ohe(column_name,X):
    X[column_name] = X[column_name].str.replace('~','_')
    X[column_name] = X[column_name].str.replace('[^a-zA-Z0-9_ ]',' ')
    X[column_name] = X[column_name].str.replace(' +',' ')
    X[column_name] = X[column_name].str.strip()
    X[column_name] = X[column_name].str.replace(' ','_')
    X[column_name] = X[column_name].str.lower()
    vectorizer = CountVectorizer(binary=True)
    return vectorizer.fit_transform(X[column_name].values)


# In[42]:


X_train_destination_ohe = ohe('destination',X_train)
X_train_passanger_ohe = ohe('passanger',X_train)
X_train_weather_ohe = ohe('weather',X_train)
X_train_time_ohe = ohe('time',X_train)
X_train_coupon_ohe = ohe('coupon',X_train)
X_train_expiration_ohe = ohe('expiration',X_train)
X_train_gender_ohe = ohe('gender',X_train)
X_train_age_ohe = ohe('age',X_train)
X_train_maritalStatus_ohe = ohe('maritalStatus',X_train)
X_train_education_ohe = ohe('education',X_train)
X_train_income_ohe = ohe('income',X_train)
X_train_Bar_ohe = ohe('Bar',X_train)
X_train_CoffeeHouse_ohe = ohe('CoffeeHouse',X_train)
X_train_CarryAway_ohe = ohe('CarryAway',X_train)
X_train_RestaurantLessThan20_ohe = ohe('RestaurantLessThan20',X_train)
X_train_Restaurant20To50_ohe = ohe('Restaurant20To50',X_train)
X_train_coupon_freq_ohe = ohe('coupon_freq',X_train)
X_train_occupation_class_ohe = ohe('occupation_class',X_train)

X_test_destination_ohe = ohe('destination',X_test)
X_test_passanger_ohe = ohe('passanger',X_test)
X_test_weather_ohe = ohe('weather',X_test)
X_test_time_ohe = ohe('time',X_test)
X_test_coupon_ohe = ohe('coupon',X_test)
X_test_expiration_ohe = ohe('expiration',X_test)
X_test_gender_ohe = ohe('gender',X_test)
X_test_age_ohe = ohe('age',X_test)
X_test_maritalStatus_ohe = ohe('maritalStatus',X_test)
X_test_education_ohe = ohe('education',X_test)
X_test_income_ohe = ohe('income',X_test)
X_test_Bar_ohe = ohe('Bar',X_test)
X_test_CoffeeHouse_ohe = ohe('CoffeeHouse',X_test)
X_test_CarryAway_ohe = ohe('CarryAway',X_test)
X_test_RestaurantLessThan20_ohe = ohe('RestaurantLessThan20',X_test)
X_test_Restaurant20To50_ohe = ohe('Restaurant20To50',X_test)
X_test_coupon_freq_ohe = ohe('coupon_freq',X_test)
X_test_occupation_class_ohe = ohe('occupation_class',X_test)


# In[43]:


def norm(column_name,X):
    normalizer = Normalizer()
    normalizer.fit(X[column_name].values.reshape(1,-1))
    X_norm = normalizer.transform(X[column_name].values.reshape(1,-1))
    return X_norm.reshape(-1,1)


# In[44]:


X_train_temperature_norm = norm('temperature',X_train)
X_train_has_children_norm = norm('has_children',X_train)
X_train_toCoupon_GEQ15min_norm = norm('toCoupon_GEQ15min',X_train)
X_train_toCoupon_GEQ25min_norm = norm('toCoupon_GEQ25min',X_train)
X_train_direction_same_norm = norm('direction_same',X_train)
X_train_to_Coupon_norm = norm('to_Coupon',X_train)

X_test_temperature_norm = norm('temperature',X_test)
X_test_has_children_norm = norm('has_children',X_test)
X_test_toCoupon_GEQ15min_norm = norm('toCoupon_GEQ15min',X_test)
X_test_toCoupon_GEQ25min_norm = norm('toCoupon_GEQ25min',X_test)
X_test_direction_same_norm = norm('direction_same',X_test)
X_test_to_Coupon_norm = norm('to_Coupon',X_test)


# In[45]:


from scipy.sparse import hstack
X_train_ohe = hstack((X_train_destination_ohe, X_train_passanger_ohe, X_train_weather_ohe, X_train_time_ohe, X_train_coupon_ohe, X_train_expiration_ohe, X_train_gender_ohe, X_train_age_ohe, X_train_maritalStatus_ohe, X_train_education_ohe, X_train_income_ohe, X_train_coupon_freq_ohe, X_train_occupation_class_ohe,X_train_Bar_ohe,X_train_CoffeeHouse_ohe,X_train_CarryAway_ohe,X_train_RestaurantLessThan20_ohe,X_train_Restaurant20To50_ohe,X_train_temperature_norm, X_train_has_children_norm,X_train_toCoupon_GEQ15min_norm,X_train_toCoupon_GEQ25min_norm,X_train_direction_same_norm,X_train_to_Coupon_norm)).tocsr()
X_test_ohe = hstack((X_test_destination_ohe, X_test_passanger_ohe, X_test_weather_ohe, X_test_time_ohe, X_test_coupon_ohe, X_test_expiration_ohe, X_test_gender_ohe, X_test_age_ohe, X_test_maritalStatus_ohe, X_test_education_ohe, X_test_income_ohe, X_test_coupon_freq_ohe, X_test_occupation_class_ohe,X_test_Bar_ohe,X_test_CoffeeHouse_ohe,X_test_CarryAway_ohe,X_test_RestaurantLessThan20_ohe,X_test_Restaurant20To50_ohe,X_test_temperature_norm, X_test_has_children_norm,X_test_toCoupon_GEQ15min_norm,X_test_toCoupon_GEQ25min_norm,X_test_direction_same_norm,X_test_to_Coupon_norm)).tocsr()
print('X_Train_ohe:',X_train_ohe.shape,'\nX_Test_ohe:',X_test_ohe.shape)


# # Modeling

# In[46]:


y_pred_test = []
for i in range(len(y_test)):
    r = np.random.random()
    if r<0.5:
        y_pred_test.append(0)
    else:
        y_pred_test.append(1)

print("Log_Loss on Test Data (Random Model) :",log_loss(y_test,y_pred_test))
print("ROC_AUC_Score on Test Data (Random Model) :",roc_auc_score(y_test,y_pred_test))


# In[47]:


def Logistic_Regression(x_train,y_train,x_test,y_test):
    clf = LogisticRegression(random_state=0,C=1.0)
    parameters = {'C':[0.01, 0.1, 1, 10, 100, 500]}
    model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
    model.fit(x_train, y_train)
    best_C = model.best_params_['C']

    clf = LogisticRegression(random_state=0,C=best_C).fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return best_C,Train_loss,Train_AUC,Test_loss,Test_AUC


# In[48]:


best_C_OrEnc, Train_loss_OrEnc, Train_AUC_OrEnc, Test_loss_OrEnc, Test_AUC_OrEnc = Logistic_Regression(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
best_C_FreEnc, Train_loss_FreEnc, Train_AUC_FreEnc, Test_loss_FreEnc, Test_AUC_FreEnc = Logistic_Regression(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
best_C_TarEnc, Train_loss_TarEnc, Train_AUC_TarEnc, Test_loss_TarEnc, Test_AUC_TarEnc = Logistic_Regression(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
best_C_ResEnc, Train_loss_ResEnc, Train_AUC_ResEnc, Test_loss_ResEnc, Test_AUC_ResEnc = Logistic_Regression(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
best_C_ohe, Train_loss_ohe, Train_AUC_ohe, Test_loss_ohe, Test_AUC_ohe = Logistic_Regression(X_train_ohe,y_train,X_test_ohe,y_test)


# In[49]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["Logistic Regression","Ordinal Encoding",best_C_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["Logistic Regression","Frequency Encoding",best_C_FreEnc,'',round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["Logistic Regression","Target Encoding",best_C_TarEnc,'',round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["Logistic Regression","Response Encoding",best_C_ResEnc,'',round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["Logistic Regression","One Hot Encoding",best_C_ohe,'',round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
Logistic_Regression_Result = table[0]
Logistic_Regression_Result


# In[50]:


def K_Neighbors_Classifier(x_train,y_train,x_test,y_test):
    clf = KNeighborsClassifier()
    parameters = {'n_neighbors':[11, 15, 21, 31, 41, 51]}
    model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
    model.fit(x_train, y_train)
    best_n_neighbors = model.best_params_['n_neighbors']

    clf = KNeighborsClassifier(n_neighbors=best_n_neighbors).fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return best_n_neighbors,Train_loss,Train_AUC,Test_loss,Test_AUC


# In[51]:


best_n_OrEnc, Train_loss_OrEnc, Train_AUC_OrEnc, Test_loss_OrEnc, Test_AUC_OrEnc = K_Neighbors_Classifier(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
best_n_FreEnc, Train_loss_FreEnc, Train_AUC_FreEnc, Test_loss_FreEnc, Test_AUC_FreEnc = K_Neighbors_Classifier(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
best_n_TarEnc, Train_loss_TarEnc, Train_AUC_TarEnc, Test_loss_TarEnc, Test_AUC_TarEnc = K_Neighbors_Classifier(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
best_n_ResEnc, Train_loss_ResEnc, Train_AUC_ResEnc, Test_loss_ResEnc, Test_AUC_ResEnc = K_Neighbors_Classifier(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
best_n_ohe, Train_loss_ohe, Train_AUC_ohe, Test_loss_ohe, Test_AUC_ohe = K_Neighbors_Classifier(X_train_ohe,y_train,X_test_ohe,y_test)


# In[52]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["K-Nearest Neighbor","Ordinal Encoding",best_n_OrEnc,'',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["K-Nearest Neighbor","Frequency Encoding",best_n_FreEnc,'',round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["K-Nearest Neighbor","Target Encoding",best_n_TarEnc,'',round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["K-Nearest Neighbor","Response Encoding",best_n_ResEnc,'',round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["K-Nearest Neighbor","One Hot Encoding",best_n_ohe,'',round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
K_Nearest_Neighbor_Result = table[0]
K_Nearest_Neighbor_Result


# In[53]:


def Decision_Tree_Classifier(x_train,y_train,x_test,y_test):
    clf = DecisionTreeClassifier(class_weight='balanced')
    parameters = {'max_depth':[1, 5, 10, 50], 'min_samples_split':[5, 10, 100, 500]}
    model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
    model.fit(x_train, y_train)
    best_depth = model.best_params_['max_depth']
    best_samples_split = model.best_params_['min_samples_split']

    clf = DecisionTreeClassifier(class_weight='balanced', max_depth=best_depth, min_samples_split=best_samples_split, random_state=0)
    clf.fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return best_depth,best_samples_split,Train_loss,Train_AUC,Test_loss,Test_AUC


# In[54]:


best_depth_OrEnc,best_samples_split_OrEnc,Train_loss_OrEnc,Train_AUC_OrEnc,Test_loss_OrEnc,Test_AUC_OrEnc = Decision_Tree_Classifier(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
best_depth_FreEnc,best_samples_split_FreEnc,Train_loss_FreEnc,Train_AUC_FreEnc,Test_loss_FreEnc,Test_AUC_FreEnc = Decision_Tree_Classifier(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
best_depth_TarEnc,best_samples_split_TarEnc,Train_loss_TarEnc,Train_AUC_TarEnc,Test_loss_TarEnc,Test_AUC_TarEnc = Decision_Tree_Classifier(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
best_depth_ResEnc,best_samples_split_ResEnc,Train_loss_ResEnc,Train_AUC_ResEnc,Test_loss_ResEnc,Test_AUC_ResEnc = Decision_Tree_Classifier(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
best_depth_ohe,best_samples_split_ohe,Train_loss_ohe,Train_AUC_ohe,Test_loss_ohe,Test_AUC_ohe = Decision_Tree_Classifier(X_train_ohe,y_train,X_test_ohe,y_test)


# In[55]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["Decision Tree","Ordinal Encoding",best_depth_OrEnc,best_samples_split_OrEnc,round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["Decision Tree","Frequency Encoding",best_depth_FreEnc,best_samples_split_FreEnc,round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["Decision Tree","Target Encoding",best_depth_TarEnc,best_samples_split_TarEnc,round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["Decision Tree","Response Encoding",best_depth_ResEnc,best_samples_split_ResEnc,round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["Decision Tree","One Hot Encoding",best_depth_ohe,best_samples_split_ohe,round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
Decision_Tree_Result = table[0]
Decision_Tree_Result


# In[56]:


def Gaussian_NB(x_train,y_train,x_test,y_test):

    clf = GaussianNB()
    clf.fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return Train_loss,Train_AUC,Test_loss,Test_AUC


# In[57]:


Train_loss_OrEnc, Train_AUC_OrEnc, Test_loss_OrEnc, Test_AUC_OrEnc = Gaussian_NB(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
Train_loss_FreEnc, Train_AUC_FreEnc, Test_loss_FreEnc, Test_AUC_FreEnc = Gaussian_NB(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
Train_loss_TarEnc, Train_AUC_TarEnc, Test_loss_TarEnc, Test_AUC_TarEnc = Gaussian_NB(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
Train_loss_ResEnc, Train_AUC_ResEnc, Test_loss_ResEnc, Test_AUC_ResEnc = Gaussian_NB(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
Train_loss_ohe, Train_AUC_ohe, Test_loss_ohe, Test_AUC_ohe = Gaussian_NB(X_train_ohe.toarray(),y_train,X_test_ohe.toarray(),y_test)


# In[58]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["Gaussian NB","Ordinal Encoding",'','',round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["Gaussian NB","Frequency Encoding",'','',round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["Gaussian NB","Target Encoding",'','',round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["Gaussian NB","Response Encoding",'','',round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["Gaussian NB","One Hot Encoding",'','',round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
Gaussian_NB_Result = table[0]
Gaussian_NB_Result


# In[59]:


def Random_Forest_Classifier(x_train,y_train,x_test,y_test):
    clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,max_features='log2',min_samples_leaf=3,random_state=42, n_jobs=-1)
    parameters = {'max_depth':[20, 50, 100], 'n_estimators':[1000,2000]}
    model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
    model.fit(x_train, y_train)
    best_depth = model.best_params_['max_depth']
    best_n_estimators = model.best_params_['n_estimators']

    clf = RandomForestClassifier(n_estimators=best_n_estimators,criterion='gini',max_depth=best_depth,max_features='log2',min_samples_leaf=3, random_state=42, n_jobs=-1)
    clf.fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return best_depth,best_n_estimators,Train_loss,Train_AUC,Test_loss,Test_AUC


# In[60]:


best_depth_OrEnc,best_n_estimators_OrEnc,Train_loss_OrEnc,Train_AUC_OrEnc,Test_loss_OrEnc,Test_AUC_OrEnc = Random_Forest_Classifier(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
best_depth_FreEnc,best_n_estimators_FreEnc,Train_loss_FreEnc,Train_AUC_FreEnc,Test_loss_FreEnc,Test_AUC_FreEnc = Random_Forest_Classifier(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
best_depth_TarEnc,best_n_estimators_TarEnc,Train_loss_TarEnc,Train_AUC_TarEnc,Test_loss_TarEnc,Test_AUC_TarEnc = Random_Forest_Classifier(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
best_depth_ResEnc,best_n_estimators_ResEnc,Train_loss_ResEnc,Train_AUC_ResEnc,Test_loss_ResEnc,Test_AUC_ResEnc = Random_Forest_Classifier(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
best_depth_ohe,best_n_estimators_ohe,Train_loss_ohe,Train_AUC_ohe,Test_loss_ohe,Test_AUC_ohe = Random_Forest_Classifier(X_train_ohe,y_train,X_test_ohe,y_test)


# In[61]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["Random Forest","Ordinal Encoding",best_depth_OrEnc,best_n_estimators_OrEnc,round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["Random Forest","Frequency Encoding",best_depth_FreEnc,best_n_estimators_FreEnc,round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["Random Forest","Target Encoding",best_depth_TarEnc,best_n_estimators_TarEnc,round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["Random Forest","Response Encoding",best_depth_ResEnc,best_n_estimators_ResEnc,round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["Random Forest","One Hot Encoding",best_depth_ohe,best_n_estimators_ohe,round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
Random_Forest_Classifier_Result = table[0]
Random_Forest_Classifier_Result


# In[62]:


import xgboost as xgb
def XGB_Classifier(x_train,y_train,x_test,y_test):
    clf = xgb.XGBClassifier()
    parameters = {'max_depth':[1, 5, 10, 50], 'n_estimators':[50,100,500,1000,2000]}
    model = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc') #scoring='roc_auc' or 'neg_log_loss'
    model.fit(x_train, y_train)
    best_depth = model.best_params_['max_depth']
    best_n_estimators = model.best_params_['n_estimators']

    clf = xgb.XGBClassifier(max_depth=best_depth, n_estimators=best_n_estimators)
    clf.fit(x_train, y_train)

    Train_loss = log_loss(y_train,clf.predict_proba(x_train))
    Train_AUC = roc_auc_score(y_train,clf.predict_proba(x_train)[:,1])
    Test_loss = log_loss(y_test,clf.predict_proba(x_test))
    Test_AUC = roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])

    return best_depth,best_n_estimators,Train_loss,Train_AUC,Test_loss,Test_AUC


# In[63]:


best_depth_OrEnc,best_n_estimators_OrEnc,Train_loss_OrEnc,Train_AUC_OrEnc,Test_loss_OrEnc,Test_AUC_OrEnc = XGB_Classifier(X_train_Ordinal_encoding,y_train,X_test_Ordinal_encoding,y_test)
best_depth_FreEnc,best_n_estimators_FreEnc,Train_loss_FreEnc,Train_AUC_FreEnc,Test_loss_FreEnc,Test_AUC_FreEnc = XGB_Classifier(X_train_frequency_encoding,y_train,X_test_frequency_encoding,y_test)
best_depth_TarEnc,best_n_estimators_TarEnc,Train_loss_TarEnc,Train_AUC_TarEnc,Test_loss_TarEnc,Test_AUC_TarEnc = XGB_Classifier(X_train_target_encoding,y_train,X_test_target_encoding,y_test)
best_depth_ResEnc,best_n_estimators_ResEnc,Train_loss_ResEnc,Train_AUC_ResEnc,Test_loss_ResEnc,Test_AUC_ResEnc = XGB_Classifier(X_train_response_encoding,y_train,X_test_response_encoding,y_test)
best_depth_ohe,best_n_estimators_ohe,Train_loss_ohe,Train_AUC_ohe,Test_loss_ohe,Test_AUC_ohe = XGB_Classifier(X_train_ohe,y_train,X_test_ohe,y_test)


# In[64]:


summary_table = PrettyTable(["Model","Encoding", "Hyperparameter1", "Hyperparameter2", "Train_log_loss", "Train_roc_auc_score", "Test_log_loss", "Test_roc_auc_score"]) #heading

summary_table.add_row(["XGB Classifier","Ordinal Encoding",best_depth_OrEnc,best_n_estimators_OrEnc,round(Train_loss_OrEnc,3),round(Train_AUC_OrEnc,3),round(Test_loss_OrEnc,3),round(Test_AUC_OrEnc,3)])
summary_table.add_row(["XGB Classifier","Frequency Encoding",best_depth_FreEnc,best_n_estimators_FreEnc,round(Train_loss_FreEnc,3),round(Train_AUC_FreEnc,3),round(Test_loss_FreEnc,3),round(Test_AUC_FreEnc,3)])
summary_table.add_row(["XGB Classifier","Target Encoding",best_depth_TarEnc,best_n_estimators_TarEnc,round(Train_loss_TarEnc,3),round(Train_AUC_TarEnc,3),round(Test_loss_TarEnc,3),round(Test_AUC_TarEnc,3)])
summary_table.add_row(["XGB Classifier","Response Encoding",best_depth_ResEnc,best_n_estimators_ResEnc,round(Train_loss_ResEnc,3),round(Train_AUC_ResEnc,3),round(Test_loss_ResEnc,3),round(Test_AUC_ResEnc,3)])
summary_table.add_row(["XGB Classifier","One Hot Encoding",best_depth_ohe,best_n_estimators_ohe,round(Train_loss_ohe,3),round(Train_AUC_ohe,3),round(Test_loss_ohe,3),round(Test_AUC_ohe,3)])

table = pd.read_html(summary_table.get_html_string())
XGB_Classifier_Result = table[0]
XGB_Classifier_Result


# In[65]:


Model_Result = [Logistic_Regression_Result,K_Nearest_Neighbor_Result,Decision_Tree_Result,Gaussian_NB_Result,
                Random_Forest_Classifier_Result,XGB_Classifier_Result] 
Result = pd.concat(Model_Result,ignore_index=True)
(Result).sort_values(by=['Test_roc_auc_score'],ascending=False).head(10)


# In[66]:


fig, ax = plt.subplots(1,2, figsize=(22,8))
sns.heatmap(Result.pivot('Model','Encoding','Test_roc_auc_score'), annot = True, fmt='.3g', ax=ax[0])
sns.heatmap(Result.pivot('Model','Encoding','Test_log_loss'), annot = True, fmt='.3g', cmap=sns.cm.rocket_r, ax=ax[1])
ax[0].set_title('Test AUC Score Heat Maps')
ax[1].set_title('Test Log Loss Heat Maps')
plt.show()


# In[ ]:




