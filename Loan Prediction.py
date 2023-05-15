#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np

#for data visualization
import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


train=pd.read_csv(r"C:\Users\komal\Downloads\test_loan.csv")
test=pd.read_csv(r"C:\Users\komal\Downloads\test_loan.csv")


# In[19]:


train.columns #all columns in dataset


# In[20]:


train.sample(5)  #to display any 5 random rows from dataset


# In[21]:


train.info() #to get type of each feature object means that it is categorical data 


# In[22]:


print(train.shape)
print(test.shape)


# In[37]:


train["Loan_Amount_Term"].value_counts().plot.bar(color=["red",'green'])


# In[39]:


(train["Loan_Amount_Term"]=="Y").value_counts()


# In[25]:


plt.figure(1)
plt.subplot(221)
train["Gender"].value_counts(normalize='True').plot.bar(figsize=(8,8),title="Gender",color=["orange","green"])
plt.subplot(222)
train["Married"].value_counts(normalize='True').plot.bar(figsize=(8,8),title="Married",color=["orange","green"])
plt.subplot(223)
train["Self_Employed"].value_counts(normalize='True').plot.bar(figsize=(8,8),title="Self_Employed",color=["orange","green"])
plt.subplot(224)
train["Credit_History"].value_counts(normalize='True').plot.bar(figsize=(8,8),title="Credit_History",color=["orange","green"])
plt.show()


# In[26]:


plt.figure(1)
plt.subplot(131)
train["Dependents"].value_counts(normalize=True).plot.bar(figsize=(12,6),color="pink",title="Dependents")
plt.subplot(132)
train["Education"].value_counts(normalize=True).plot.bar(figsize=(12,6),title="Education")
plt.subplot(133)
train["Property_Area"].value_counts(normalize=True).plot.bar(figsize=(12,6),color="yellow",title="Property_Area")

#DependentsEducation


# In[27]:


import seaborn as sns
plt.figure(1)
plt.subplot(121)
train["ApplicantIncome"].plot.box(figsize=(12,6))
plt.subplot(122)
sns.distplot(train["ApplicantIncome"])
plt.show()


# In[28]:


train.boxplot(column='ApplicantIncome', by = 'Education')


# In[29]:


plt.figure(1)
plt.subplot(121)
train["CoapplicantIncome"].plot.box(figsize=(12,6))
plt.subplot(122)
sns.distplot(train["CoapplicantIncome"])
plt.show()


# In[30]:


plt.figure(1)
plt.subplot(121)
train["LoanAmount"].plot.box()
plt.subplot(122)
sns.distplot(train["LoanAmount"])
plt.show()
#it's also not a normal distrubution 


# In[34]:


train.boxplot(column='ApplicantIncome', by = 'Education')


# In[35]:


plt.figure(1)
plt.subplot(121)
train["CoapplicantIncome"].plot.box(figsize=(12,6))
plt.subplot(122)
sns.distplot(train["CoapplicantIncome"])
plt.show()
#Not a normal distribution


# In[36]:


plt.figure(1)
plt.subplot(121)
train["LoanAmount"].plot.box()
plt.subplot(122)
sns.distplot(train["LoanAmount"])
plt.show()
#it's also not a normal distrubution 


# In[40]:


Gender=pd.crosstab(train['Gender'],train['Loan_Amount_Term']) 
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)


# In[41]:


Married=pd.crosstab(train['Married'],train['Loan_Amount_Term']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[42]:


Employed=pd.crosstab(train['Self_Employed'],train['Loan_Amount_Term']) 
Employed.div(Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[43]:


Dependents=pd.crosstab(train['Dependents'],train['Loan_Amount_Term']) 
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[44]:


property_a = pd.crosstab(train["Property_Area"],train["Loan_Amount_Term"])
property_a.div(property_a.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True)


# In[45]:


#Credit_History
credit_hist = pd.crosstab(train["Credit_History"],train["Loan_Amount_Term"])
credit_hist.div(credit_hist.sum(1).astype(float),axis=0).plot(kind="bar",stacked="True")


# In[46]:


#encoding columns
train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True) 
train['Loan_Amount_Term'].replace('N', 0,inplace=True) 
train['Loan_Amount_Term'].replace('Y', 1,inplace=True)

 #plot corelation matrix
matrix = train.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[47]:


is_na = train.isna().sum()
is_na


# In[48]:


train["Gender"].fillna(train["Gender"].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
print(train.shape)


# In[49]:


#normalise the outlier it gives normalsied distribution
train["LoanAmount"] = np.log(train['LoanAmount'])
test["LoanAmount"] = np.log(test['LoanAmount'])


# In[50]:


train.head(2)
from sklearn import preprocessing


# In[51]:


#drop Loan_ID
train = train.drop("Loan_ID",axis=1)
test = test.drop("Loan_ID",axis=1)
print(train.shape)


# In[52]:


#seprate target colmn
X = train.iloc[:,:-1]
Y = train.iloc[:,-1]
print(X.shape)
print(Y.shape)
print(X.columns)


# In[53]:


X =pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[54]:


train.shape


# In[55]:


#we train our model on train set and make prediction on cross veidation set
#so plit train set into train and cross velidation set


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#split train & cv
x_train,x_cv,y_train,y_cv = train_test_split(X,Y,test_size = 0.3) 

#model initialisation & fit

model = LogisticRegression()
model.fit(x_train,y_train)

#preict
predict_y = model.predict(x_cv)

#accuracy
acc = accuracy_score(y_cv,predict_y)

print("accuracy on cv",acc*100,'%')

#for test set we need to perform all operations on test set also 
#than predict 


# In[56]:


from sklearn.model_selection import StratifiedKFold


# In[57]:


i=1
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
acc=0
for train_index,test_index in kf.split(X,Y):
    print("\n{} of kfold {}".format(i,kf.n_splits))
    xtr,xvl=X.loc[train_index],X.loc[test_index]
    ytr,yvl=Y.loc[train_index],Y.loc[test_index]
    model = LogisticRegression(max_iter=150,random_state=1)
    model.fit(xtr,ytr)
    pred_test = model.predict(xvl)
    score=accuracy_score(yvl,pred_test)
    acc = acc+score
    print('accuracy_score',score)
    i+=1 
    
print("Mean accu",acc/5 *100,'%')


# In[ ]:




