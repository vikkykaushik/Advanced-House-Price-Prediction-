#!/usr/bin/env python
# coding: utf-8

# ![house%20prediction.png](attachment:house%20prediction.png)

# # <div align="center"> Advanced-House-Price-Prediction
# 
# 

# ## The Capstone Project for IBM Advanced Data Science Specialization Certification by Coursera
# 
# ----

# ### *Abstract*
# ------
# In this project, We build a model to predict House-Price. 
# For the selection of prediction methods we compare and explore various
# prediction methods. We utilize Random Forest as our model
# because of its adaptable and probabilistic methodology on model
# selection. Our result exhibit that our approach of the issue need
# to be successful, and has the ability to process predictions that
# would be comparative with other house cost prediction models. 
# XGBoost, lasso regression and neural system on look at their
# order precision execution. We in that point recommend a housing
# cost prediction model to support a house vender or a real estate person. 
# 
# 
# 

# ### *Objective*
# -----
# The objective of this project is to build a model to forecast the House-Price. You are provided with 79 explanatory variables describing (almost) every aspect of residential homes.
# 
# ###### Dataset to downloaded from the below link
# [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
# 

# ### *Motivation for The Case Study*
# ------
# Housing prices are an important reflection of the economy, and housing price ranges are of great interest for both buyers and sellers . Ask a home buyer to describe their dream house, and they probably won’t begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition’s data-set proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

# ### All the Lifecycle In A Data Science Projects
# ------
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 5. Model Deployment

# In[4]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install missingno')


# In[5]:


## Data Analysis Phase
## MAin aim is to understand more about the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
## Display all the columns of the dataframe

pd.pandas.set_option('display.max_columns',None)


# In[6]:


dataset=pd.read_csv('C:\\Users\\vkaush2\\Desktop\\House estimation kaggle\\train.csv')

## print shape of dataset with rows and columns
print(dataset.shape)


# In[7]:


## print the top5 records
dataset.head()


# #### In Data Analysis We will Analyze To Find out the below stuff
# 1. Missing Values
# 2. All The Numerical Variables
# 3. Distribution of the Numerical Variables
# 4. Categorical Variables
# 5. Cardinality of Categorical Variables
# 6. Outliers
# 7. Relationship between independent and dependent feature(SalePrice)

# ### Missing Values
# 

# In[8]:


## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(), 4),  ' % missing values')


# ### Since they are many missing values, we need to find the relationship between missing values and Sales Price
# 
# ##### Let's plot some diagram for this relationship

# In[9]:


for feature in features_with_na:
    data = dataset.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section
# 
# From the above dataset some of the features like Id is not required

# In[10]:


print("Id of Houses {}".format(len(dataset.Id)))


# ### Numerical Variables

# In[11]:


# list of numerical variables
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
dataset[numerical_features].head()


# ##### Temporal Variables(Eg: Datetime Variables)
# From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering which is the next video.

# In[12]:


# list of variables that contain year information
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

year_feature


# In[13]:


# let's explore the content of these year variables
for feature in year_feature:
    print(feature, dataset[feature].unique())


# In[14]:


## Lets analyze the Temporal Datetime Variables
## We will check whether there is a relation between year the house is sold and the sales price

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


# In[15]:


year_feature


# In[16]:


## Here we will compare the difference between All years feature with SalePrice

for feature in year_feature:
    if feature!='YrSold':
        data=dataset.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()


# In[17]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[18]:


discrete_feature


# In[19]:


dataset[discrete_feature].head()


# In[20]:


## Lets Find the realtionship between them and Sale PRice

for feature in discrete_feature:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# #### There is a relationship between variable number and SalePrice
# 

# ### Continuous Variable

# In[21]:


continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[22]:


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# In[23]:


## We will be using logarithmic transformation


for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()


# ### Outliers

# In[24]:


for feature in continuous_feature:
    data=dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# ### Categorical Variables

# In[25]:


categorical_features=[feature for feature in dataset.columns if data[feature].dtypes=='O']
categorical_features


# In[26]:


dataset[categorical_features].head()


# In[27]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))


# #### Find out the relationship between categorical variable and dependent feature SalesPrice

# In[29]:


for feature in categorical_features:
    data=dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# ### Advanced Housing Prices- Feature Engineering
# ##### We will be performing all the below steps in Feature Engineering
# 
# 1. Missing values
# 2. Temporal variables
# 3. Categorical variables: remove rare labels
# 4. Standarise the values of the variables to the same range

# In[31]:


## Always remember there way always be a chance of data leakage so we need to split the data first and then apply feature
## Engineering
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)


# In[32]:


X_train.shape, X_test.shape


# ### Missing Values

# In[33]:


## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[34]:


## Replace missing value with a new label
def replace_cat_feature(dataset,features_nan):
    data=dataset.copy()
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()


# In[35]:


dataset.head()


# In[36]:


## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))


# In[37]:


## Replacing the numerical Missing Values

for feature in numerical_with_nan:
    ## We will replace by using median since there are outliers
    median_value=dataset[feature].median()
    
    ## create a new feature to capture nan values
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)
    
dataset[numerical_with_nan].isnull().sum()


# In[38]:


dataset.head(50)


# In[39]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[40]:


dataset.head()


# In[41]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# ### Numerical Variables
# #### Since the numerical variables are skewed we will perform log normal distribution

# In[42]:


dataset.head()


# In[43]:


num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[44]:


dataset.head()


# ### Handling Rare Categorical Feature
# #### We will remove categorical variables that are present less than 1% of the observations

# In[45]:


categorical_features=[feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[46]:


categorical_features


# In[47]:


for feature in categorical_features:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df=temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'Rare_var')


# In[48]:


dataset.head(10)


# In[49]:


for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[50]:


dataset.head(10)


# In[51]:


scaling_feature=[feature for feature in dataset.columns if feature not in ['Id','SalePerice'] ]
len(scaling_feature)


# In[52]:


scaling_feature


# In[53]:


dataset.head()


# ### Feature Scaling

# In[54]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[55]:


scaler.transform(dataset[feature_scale])


# In[56]:


# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                    axis=1)


# In[57]:


data.head()


# In[58]:


data.to_csv('X_train.csv',index=False)


# ### Feature Selection Advanced House Price Prediction

# In[64]:


## for feature slection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# to visualise al the columns in the dataframe


# In[59]:


dataset=pd.read_csv('X_train.csv')


# In[60]:


dataset.head()


# In[61]:


## Capture the dependent feature
y_train=dataset[['SalePrice']]


# In[62]:


## drop dependent feature from dataset
X_train=dataset.drop(['Id','SalePrice'],axis=1)


# In[65]:


### Apply Feature Selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
feature_sel_model.fit(X_train, y_train)


# In[66]:


feature_sel_model.get_support()


# In[70]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))


# In[69]:


selected_feat


# In[71]:


X_train=X_train[selected_feat]


# In[72]:


X_train.head()


# In[ ]:




