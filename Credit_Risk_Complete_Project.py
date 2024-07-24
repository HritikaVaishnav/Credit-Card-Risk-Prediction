#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement: 

# ### Import all required labraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
import os

from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

import time


# In[2]:


# Time calculation for project(it is use for deployment)
print("Program is running")
print()
start_time = time.time()


# ### Load and read the datasets

# In[3]:


df1 = pd.read_excel("case_study1.xlsx")
df2 = pd.read_excel("case_study2.xlsx")


# In[4]:


df1.head()


# In[5]:


df1.tail()


# In[6]:


df1.info()


# In[7]:


df2.info()


# In[8]:


df1.shape


# In[9]:


df1.size


# In[10]:


df2.shape


# In[11]:


df2.size


# ### Descriptive Analysis of Data

# In[12]:


df1.describe().T


# In[13]:


df2.head()


# In[14]:


df2.tail()


# In[15]:


df2.describe().T


# ### Data Cleaning

# #### 1. Check Null Values
# #### For safe assuption if we have more null values in rows then we delete the columns and if we have more null values in columns then we delete that rows.

# In[16]:


df1.isnull().sum()


# In[17]:


df1.isnull().sum().sum()


# In[18]:


df2.isnull().sum()


# In[19]:


df2.isnull().sum().sum()


# **But in our both dataset -99999 values are Null so we remove this or delete that rows each have -99999 values**

# In[20]:


# Count occurrences of -99999
count = (df1 == -99999).sum().sum()
print(f"Count of -99999: {count}")


# In[21]:


# Calculate the proportion of -99999
total_elements = df1.size
proportion = count / total_elements
print(f"Proportion of -99999: {proportion}")


# **Dataset have only 2% of -99999 value so we delete that rows which contain -99999 value.**

# In[22]:


# Remove rows containing -99999
df1 = df1[~(df1 == -99999).any(axis=1)]
print("DataFrame after removing rows with -99999:")
print(df1)


# In[23]:


# Check any occurrences of -99999
count = (df1 == -99999).sum().sum()
print(f"Count of -99999: {count}")


# In[24]:


df1.shape


# In[25]:


# Count occurrences of -99999 in df2
count = (df2 == -99999).sum().sum()
print(f"Count of -99999: {count}")


# In[26]:


# Calculate the proportion of -99999
total_elements = df2.size
proportion = count / total_elements
print(f"Proportion of -99999: {proportion}")


# **Dataset have only 1% of -99999 value so we delete that columns according to the description of df2 which contain -99999 value.**

# In[27]:


# Count occurrences of -99999 in each column
column_counts = (df2 == -99999).sum()

print("Occurrences of -99999 in each column:")
print(column_counts)
column_counts = (df2 == -99999).sum().sum()
print("Total Occurrences of -99999 in columns:",column_counts)


# In[28]:


# Delete that columns which have ocurrence of -99999 more then 10000
columns_to_be_removed = []
for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)


# In[29]:


df2 = df2.drop(columns_to_be_removed, axis =1)


# In[30]:


df2.shape


# In[31]:


# Check any occurrences of -99999
count = (df2 == -99999).sum().sum()
print(f"Count of -99999: {count}")


# **That -99999 values is left in rows so we delete that rows**

# In[32]:


# Remove Rows which have -99999 values
for i in df2.columns:
    df2 = df2.loc[df2[i] != -99999 ]


# In[33]:


# Check any occurrences of -99999
count = (df2 == -99999).sum().sum()
print(f"Count of -99999: {count}")


# In[34]:


df2.shape


# #### 2. Check Duplictes

# In[35]:


df1.duplicated()


# In[36]:


df1.duplicated().sum()


# In[37]:


df2.duplicated()


# In[38]:


df2.duplicated().sum()


# ### Merge the both dataset using inner join
# we use inner join becuse we do not want any null value in other joins it will take null values

# In[39]:


# Identify common columns
common_columns = df1.columns.intersection(df2.columns)
print("Common columns:", common_columns)


# In[40]:


# Perform an inner join on these common columns
# Merge on common columns
df_new = pd.merge(df1, df2, on=list(common_columns))


# In[41]:


print("Result of the inner join:")
print(df_new)


# In[42]:


df_new.head()


# In[43]:


df_new.tail()


# In[44]:


df_new.info()


# In[45]:


df_new.isnull().sum().sum()


# In[46]:


df_new.shape


# In[47]:


df_new.size


# ### Separate numerical and categorical columns

# In[48]:


numerical_columns = df_new.select_dtypes(include=['number']).columns
categorical_columns = df_new.select_dtypes(include=['object', 'category']).columns


# In[49]:


print("Numerical columns:", numerical_columns)


# In[50]:


print("Categorical columns:", categorical_columns)


# ### Count the values for each numerical column

# In[51]:


print("\nValue counts for numerical columns:")
for col in numerical_columns:
    print(f"Value counts for {col}:")
    print(df_new[col].value_counts())


# ### Count the values for each categorical column

# In[52]:


print("\nValue counts for categorical columns:")
for col in categorical_columns:
    print(f"\nValue counts for {col}:")
    print(df_new[col].value_counts())


# ### Feature Selection
# In this we perform Feature Selection frist then EDA beacuse here 79 columns

# #### To Check Cardinality or Co-linearity in Categorical Features

# ##### Take Hypothesis Testing
# **H0 = Null Hypothesis = `MARITALSTATUS` and `Approved_Flag(Target Column)` are not associated**
# 
# **H1 = Alternate Hypothesis = `MARITALSTATUS` and `Approved_Flag(Target Column)` are associated**
# 
# **Alpha = 5% = 0.05**
# 
# **Confidence = 1-Alpha**
# 
# **Calculate avidence againt the Null Hypothesis H0**
# 
#   (i) calculate p-value
#   
#   (ii) Apply one Test like T-test, Chi-square test, ANOVA test
#   
#       * Chi-square test apply for categorical v/s categorical features
#       * T-test apply for categorical v/s Numerical features(2 Categories like P1,P2)
#       * ANOVA test apply for categorical v/s Numerical features(>= 3 Categories like P1,P2,P3,P4 Or so on)
#       
# **If p-value <= alpha then reject H0 and if p-value > alpha then fail to reject H0**
#   

# ##### Apply Chi-Square Testing Between categorical features v/s Target Column(which is categorical)

# In[53]:


for i in ['MARITALSTATUS','EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _,_ = chi2_contingency(pd.crosstab(df_new[i], df_new['Approved_Flag']))
    print(i, '----', pval)


# * According to the p-values of 'MARITALSTATUS','EDUCATION', 'GENDER', 'last_prod_enq2', and 'first_prod_enq2' are < alpha value so we reject the H0 that means 'MARITALSTATUS','EDUCATION', and 'GENDER' are not cardinality with 'Approved_Flag' 
# 
# **Since all the categoriacl features have p-value < alpha (0.05) so we will reject all Ho**

# ### Check Multicolinearity with Numerical columns

# In[54]:


# Numerical columns
numeric_columns = []
for i in df_new.columns:
    if df_new[i].dtype != 'object' and i not in ['PROSPECTID']:
        numeric_columns.append(i)    


# In[55]:


# Print all numerical columns
print("Numerical columns:", numeric_columns)


# In[56]:


# Print the count of numerical columns
print("Count of numerical columns:", len(numeric_columns))


# ### Plot Heatmap

# In[57]:


# Select only the numerical columns
numerical_df = df_new[numeric_columns]


# In[58]:


# Compute the correlation matrix
corr_matrix = numerical_df.corr()
corr_matrix


# In[59]:


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# #### Check VIF Sequentially

# In[60]:


vif_data = df_new[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0

for i in range (0,total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '----', vif_value)
    
    if vif_value <= 6:
        columns_to_be_kept.append(numeric_columns[i])
        column_index = column_index+1
    else :
        vif_data = vif_data.drop([numeric_columns[i]], axis=1)


# In[61]:


print(len(columns_to_be_kept))
print(columns_to_be_kept)


# **After Sequential VIF we have 39 columns**

# ### Check ANOVA Test with 39 Columns(columns_to_be_kept)

# In[62]:


columns_to_be_kept_numerical = []

# Assuming a and b are our lists or arrays
for i in columns_to_be_kept:
    a = list(df_new[i])
    b = list(df_new['Approved_Flag'])
    
    # Group data based on group identifiers in b
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']
    
    # Calculate F-statistic and p-value using one-way ANOVA
    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)
    
    # Assuming you have a column and you want to append it based on the p-value
    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)


# In[63]:


print(columns_to_be_kept_numerical)
print(len(columns_to_be_kept_numerical))


# #### Listing all the final features

# In[64]:


features = columns_to_be_kept_numerical + ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']


# In[65]:


df_new = df_new[features + ['Approved_Flag']]


# ### Lable Encoding

# In[66]:


# Print unique values for each categorical column
for col in categorical_columns:
    unique_values = df_new[col].unique()
    print(f"Column '{col}' unique values: {unique_values}")


# **We apply `Lable Encoding` with `EDUCATION` column because we can ranking or ordering by the 1 to 4 for other column we can not apply ranking or ordering**
# * `Others` has to be verified by the business end user

# In[67]:


# Map the 'EDUCATION' column to new values
df_new.loc[df_new['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df_new.loc[df_new['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df_new.loc[df_new['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df_new.loc[df_new['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df_new.loc[df_new['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df_new.loc[df_new['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1                                                                             
df_new.loc[df_new['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3


# In[68]:


df_new['EDUCATION'].value_counts()


# In[69]:


df_new['EDUCATION'] = df_new['EDUCATION'].astype(int)


# In[70]:


df_new.info()


# ### Apply One Hot Encoding (Create Dummies)

# In[71]:


df_new_encoded = pd.get_dummies(df_new, columns = ['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])


# In[72]:


df_new_encoded.head()


# In[73]:


df_new_encoded.info()


# In[74]:


df_new_encoded.describe()


# ### Machine learning model fitting

# In[75]:


y = df_new_encoded['Approved_Flag']
x = df_new_encoded.drop(['Approved_Flag'], axis=1)


# #### Split the data into 80-20

# In[76]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state = 42)


# In[77]:


x_train.head()


# In[78]:


x_test.head()


# ### Random Forest Model

# In[79]:


rf_classifier = RandomForestClassifier(n_estimators = 200, random_state = 42)


# In[80]:


rf_classifier.fit(x_train, y_train)


# In[81]:


y_pred = rf_classifier.predict(x_test)


# In[82]:


accuaracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuaracy}')


# In[83]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


# In[84]:


for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1_Score: {f1_score[i]}")
    print()


# **Here for Class P3 Model not perfrom well so we take another model**

# ### Xgboost Model

# In[85]:


xgb_classifier = xgb.XGBClassifier(objective = 'multi:softmax', num_class = 4)


# In[86]:


y = df_new_encoded['Approved_Flag']
x = df_new_encoded.drop(['Approved_Flag'], axis=1)


# In[87]:


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[88]:


x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2, random_state = 42)


# In[89]:


xgb_classifier.fit(x_train, y_train)


# In[90]:


y_pred = xgb_classifier.predict(x_test)


# In[91]:


accuaracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuaracy}')


# In[92]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


# In[93]:


for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1_Score: {f1_score[i]}")
    print()


# ### Decision Tree

# In[94]:


y = df_new_encoded['Approved_Flag']
x = df_new_encoded.drop(['Approved_Flag'], axis=1)


# In[95]:


x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2, random_state = 42)


# In[96]:


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)


# In[97]:


dt_model.fit(x_train, y_train)


# In[98]:


y_pred = dt_model.predict(x_test)


# In[99]:


accuaracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuaracy}')


# In[100]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


# In[101]:


for i, v in enumerate(['P1','P2','P3','P4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1_Score: {f1_score[i]}")
    print()


# **After applying all three models, we have accuracy as Random Forest Classifier is 0.76, XGBoot Classifier is 0.78, and Decision Tree is 0.71. So our XGBoot Classifier's accuracy is high compared to other models. We take the XGBoot classifier and apply some techniques like hyperparameter tuning, scaling, feature engineering, and so on to make it the best fit model for our problem.**

# **XGBoot is giving the highest accuracy. So we will pick it and futher finetune it**

# ### Finetune The XGBoot Model

# #### Hyperparameter Tuning

# In[108]:


# Define the hyperparameter grid
param_grid = {
    'colsample_bytree' : [0.1, 0.1, 0.5, 0.7, 0.9],
    'learning_rate' : [0.001, 0.01, 0.1, 1],
    'max_depth' : [3, 5, 8, 10],
    'alpha' : [1,10,100],
    'n_estimators' : [10, 50, 100]
}
index = 0
results = []

answer_grid = {
    'combination' : [],
    'train_Accuracy' : [],
    'test_Accuracy' : [],
    'colsample_bytree' : [],
    'learning_rate' : [],
    'max_depth' : [],
    'alpha' : [],
    'n_estimators' : []
}


# In[103]:


y = df_new_encoded['Approved_Flag']
x = df_new_encoded.drop(['Approved_Flag'], axis = 1)


# In[104]:


label_encoder = LabelEncoder()


# In[105]:


y_encoded = label_encoder.fit_transform(y)


# In[106]:


x_train, x_test, y_train, y_test = train_test_split(x,y_encoded, test_size=0.2, random_state = 42)


# In[109]:


# Loop through each combination of hyperparameters
for colsample_bytree in param_grid['colsample_bytree']:
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for alpha in param_grid['alpha']:
                for n_estimators in param_grid['n_estimators']:
                    index = index + 1
                    
                    # Define and Train the XGBoot Model
                    model = xgb.XGBClassifier(objective = 'multi:softmax',
                         num_class = 4,
                         colsample_bytree = colsample_bytree,
                         learning_rate = learning_rate,
                         max_depth = max_depth,
                         alpha = alpha,
                         n_estimators = n_estimators)
                    
                    model.fit(x_train, y_train)
                    
                    # Predict on training and testing sets
                    y_pred_train = model.predict(x_train)
                    y_pred_test = model.predict(x_test)
                    
                    # Calculate train and test results
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    # Store results in a list of dictionaries
                    results.append({
                        'Combination': index,
                        'colsample_bytree': colsample_bytree,
                        'learning_rate': learning_rate,
                        'max_depth': max_depth,
                        'alpha': alpha,
                        'n_estimators': n_estimators,
                        'Train Accuracy': train_accuracy,
                        'Test Accuracy': test_accuracy
                    })
                    
                    # print results for this combination
                    print(f"Combination {index}")
                    print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
                    print(f"Train Accuracy: {train_accuracy: .2f}")
                    print(f"Test Accuracy: {train_accuracy: .2f}")
                    print("-" * 30)


# In[110]:


# Create a DataFrame from the answer grid
results_df = pd.DataFrame(results)


# In[111]:


# Save the results to an Excel file
results_df.to_excel('xgboost_hyperparameter_tuning_results.xlsx', index=False)

print("Results saved to 'xgboost_hyperparameter_tuning_results.xlsx'")


# In[122]:


a3 = pd.read_excel("Unseen_Dataset.xlsx")


# In[123]:


cols_in_df_new = list(df_new.columns)


# In[124]:


cols_in_df_new.pop(42)


# In[125]:


df_unseen = a3 [cols_in_df_new]


# In[116]:


categorical_columns = df_unseen.select_dtypes(include=['object', 'category']).columns


# In[117]:


# Print unique values for each categorical column
for col in categorical_columns:
    unique_values = df_unseen[col].unique()
    print(f"Column '{col}' unique values: {unique_values}")


# In[118]:


# Map the 'EDUCATION' column to new values
df_unseen.loc[df_unseen['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df_unseen.loc[df_unseen['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1                                                                             
df_unseen.loc[df_unseen['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3


# In[119]:


df_unseen['EDUCATION'].value_counts()


# In[120]:


df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)


# In[121]:


df_unseen_encoded = pd.get_dummies(df_unseen, columns = ['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])


# In[126]:


model = xgb.XGBClassifier(objective='multi:softmax',
                         num_class=4,
                         colsample_bytree=0.9,
                         learning_rate=1,
                         max_depth=3,
                         alpha=10,
                         n_estimators=100)


# In[127]:


model.fit(x_train, y_train)


# In[129]:


y_pred_unseen = model.predict(df_unseen_encoded)


# In[130]:


a3 ['Target_variable'] = y_pred_unseen


# In[134]:


a3.to_excel("C:/Users/Hritika Vaishnav/Final_Predication.xlsx", index=False)


# ### Print Runtime

# In[135]:


end_time = time.time()
elapsed_time = end_time - start_time
print("Total run time of the program:" + str (round(elapsed_time,2))+ ' sec')


# In[136]:


input("Press Enter to exit")


# In[ ]:




