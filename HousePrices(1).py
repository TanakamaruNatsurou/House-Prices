
# coding: utf-8

# In[1]:

#Mission: It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 
#Submissions are evaluated on Root-Mean-Squared-Error (RMSE)between the logarithm of the predicted value and the logarithm of the observed sales price.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

#unprocessed data を取得
df_train = pd.read_csv('unpro_train.csv')


# In[4]:

#表示する列数を変更（デフォルトだと全部表示してくれなかった）
pd.set_option("display.max_columns", 100)
df_train.head(10)


# In[10]:

#カテゴリデータをガン無視してみる
df_num_select = df_train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:]
df_num_select.describe()


# In[8]:

#欠損値の概要把握
total = df_num_select.isnull().sum().sort_values(ascending=False)
percent = (df_num_select.isnull().sum()/df_num_select.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(10)


# In[11]:

#"LotFrontage","GarageYrBlt"は平均値補完,"MasVnrArea"は0補完
df_num_select["LotFrontage"] = df_num_select["LotFrontage"].fillna(df_num_select["LotFrontage"].mean())
df_num_select["GarageYrBlt"] = df_num_select["GarageYrBlt"].fillna(df_num_select["GarageYrBlt"].mean())
df_num_select["MasVnrArea"] = df_num_select["MasVnrArea"].fillna(0)

df_num_select.head(10)


# In[13]:

#相関係数ヒートマップ
plt.figure(figsize=(18, 15))
sns.heatmap(df_num_select.corr(), annot=True, square=True, fmt='.2f')
plt.show()


# In[21]:

#さらに相関ありそうなやつ(|相関係数|>0.4)を取り出して、散布図行列をみる
df_pickup = df_num_select.loc[:,["MasVnrArea","1stFlrSF","GarageArea","GarageCars","GarageYrBlt","GrLivArea","OverallQual","TotRmsAbvGrd","TotalBsmtSF","YearBuilt","YearRemodAdd","SalePrice"]]
sns.pairplot(df_pickup, size=2.0)
plt.show()


# In[36]:

#"OverallQual"で単回帰してみる
X_1 = df_num_select.loc[:,["OverallQual"]].values
y = df_num_select.loc[:,["SalePrice"]].values


# In[37]:

from sklearn.model_selection import train_test_split
X_1_train,X_1_test,y_train,y_test = train_test_split(X_1,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
model_1 = LinearRegression()
model_1.fit(X_1_train,y_train)


# In[38]:

# 自由度調整済み決定係数
# (決定係数, trainまたはtestのサンプル数, 利用した特徴量の数)
def adjusted(score, n_sample, n_features):
    adjusted_score = 1 - (1 - score) * ((n_sample - 1) / (n_sample - n_features - 1))
    return adjusted_score


# In[39]:

print('train: %3f' % adjusted(model_1.score(X_1_train, y_train), len(y_train), 1))
print('test : %3f' % adjusted(model_1.score(X_1_test, y_test), len(y_test), 1))


# In[41]:

#("GrLivArea","OverallQual")で重回帰分析
X_2 = df_num_select.loc[:,["GrLivArea","OverallQual"]].values
y = df_num_select.loc[:,["SalePrice"]].values


# In[42]:

X_2_train,X_2_test,y_train,y_test = train_test_split(X_2,y,test_size=0.3,random_state=0)

model_2 = LinearRegression()
model_2.fit(X_2_train,y_train)


# In[43]:

print('train: %3f' % adjusted(model_2.score(X_2_train, y_train), len(y_train), 2))
print('test : %3f' % adjusted(model_2.score(X_2_test, y_test), len(y_test), 2))


# In[44]:

# 2次関数
from sklearn.preprocessing import PolynomialFeatures
quad = PolynomialFeatures(degree=2) 
X_quad = quad.fit_transform(X_2) 

X_quad_train,X_quad_test,y_train,y_test = train_test_split(X_quad,y,test_size=0.3,random_state=0)
model_2_quad = LinearRegression()
model_2_quad.fit(X_quad_train, y_train)


# In[45]:

print('train: %3f' % adjusted(model_2_quad.score(X_quad_train, y_train), len(y_train), 3))
print('test : %3f' % adjusted(model_2_quad.score(X_quad_test, y_test), len(y_test), 3))


# In[46]:

# 3次関数
cubic = PolynomialFeatures(degree=3) 
X_cubic = cubic.fit_transform(X_2) 

X_cubic_train,X_cubic_test,y_train,y_test = train_test_split(X_cubic,y,test_size=0.3,random_state=0)
model_2_cubic = LinearRegression()
model_2_cubic.fit(X_cubic_train, y_train)


# In[71]:

print('train: %3f' % adjusted(model_2_cubic.score(X_cubic_train, y_train), len(y_train), 4))
print('test : %3f' % adjusted(model_2_cubic.score(X_cubic_test, y_test), len(y_test), 4))


# In[105]:

#交差検証
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model_2_cubic, X_cubic, y, cv=kf)
print(scores)
print(scores.mean())


# In[134]:

# RMSE(SalePrice mean:180921,std:79443)
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

print('quad_train: %.3f' % sqrt(mse(y_train, model_2_quad.predict(X_quad_train))))
print('quad_test : %.3f' % sqrt(mse(y_test,model_2_quad.predict(X_quad_test))))
print("")
print('cubic_train: %.3f' % sqrt(mse(y_train,model_2_cubic.predict(X_cubic_train))))
print('cubic_train: %.3f' % sqrt(mse(y_test,model_2_cubic.predict(X_cubic_test))))


# In[135]:

#RMSLE
print('quad_train: %.3f' % sqrt(mse(np.log(y_train), np.log(model_2_quad.predict(X_quad_train)))))
print('quad_test : %.3f' % sqrt(mse(np.log(y_test),np.log(model_2_quad.predict(X_quad_test)))))
print("")
print('cubic_train: %.3f' % sqrt(mse(np.log(y_train),np.log(model_2_cubic.predict(X_cubic_train)))))


# In[117]:

#RMSEを交差検証
from sklearn.metrics import make_scorer
scorer = make_scorer(mean_squared_error, greater_is_better = False)
def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_cubic_test, y_test, scoring = scorer, cv = 10))
    return(rmse)

rmse_cv_test(model_2_cubic).mean()


# In[101]:

# 残差プロット
def res_plot(y_train, y_train_pred, y_test, y_test_pred):
    res_train = y_train_pred - y_train
    res_test  = y_test_pred - y_test
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_train_pred, res_train, color='blue', marker='o', label='train', alpha=0.5)
    plt.scatter(y_test_pred, res_test, color='green', marker='s', label='test', alpha=0.5)
    
    plt.xlabel('Predicted Values')                 
    plt.ylabel('Residuals')                         
    plt.legend(loc='upper left')                   
    plt.hlines(y=0, xmin=-100000, xmax=600000,color='red')
    plt.show()


# In[102]:

res_plot(y_train, model_2_cubic.predict(X_cubic_train),y_test, model_2_cubic.predict(X_cubic_test))


# In[136]:

df_test = pd.read_csv('unpro_test.csv')
df_test.head(10)


# In[139]:

sub = pd.DataFrame()
sub['Id'] = df_test["Id"]
X_sub = df_test.loc[:,["GrLivArea","OverallQual"]].values
cubic = PolynomialFeatures(degree=3) 
X_cubic_sub = cubic.fit_transform(X_sub) 
sub['SalePrice'] = model_2_cubic.predict(X_cubic_sub)
sub.to_csv('submission.csv',index=False)


# In[ ]:

#1834/2157 でした

