#%% [markdown]
# # <p style="text-align: center;"> FAUD DETECTION PROJECT</p>
# If a fraudolent activity is to happen, customers may be charged for items that they did not purchase. It is thus important for banks to be able to detect such activities as soon as possible in order to protect their customers.
# 
# The dataset used in this analysis has been downloaded from [Keggle](https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/data).
#%% [markdown]
# # Imports

#%%
# Data manipulation 
import numpy as np
import pandas as pd

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns

# Setting plot style
plt.style.use('ggplot')

df = pd.read_csv('data/creditcard.csv')

#%% [markdown]
# ## Exploratory data analysis

#%%
df.head()


#%%
df.describe()


#%%
df.info()


#%%
df.shape


#%%
df.columns

#%% [markdown]
# From the description of the dataset we know that the V1 to V28 features went through a principal component analysis (PCA) and have been anonymized. Also they have been scaled already as a consequence of the PCA. 

#%%
# Let's check if there are any missing data
assert df.notnull().all().all()


#%%
# Let's plot the 'Class' category
df['Class'].value_counts().plot(kind='bar', log=True)
plt.xticks(np.arange(2), ['No Fraud', 'Fraud'], rotation='horizontal')

#%% [markdown]
# As can be seen the dataset is heavilly skewed and many transactions are categorized as not fraudolent. If we were to use the dataset as it is, the model would probably not give reliable predictions and would likely overfit thinking many transactions are not a fraud. 

#%%
# Some more distributions
fig, ax = plt.subplots(1, 2, figsize=(18,4))
df['Time'].plot(kind='box', ax=ax[0])
df['Amount'].plot(kind='box', ax=ax[1])


#%%
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount = df.Amount.values
time = df.Time.values

sns.distplot(amount, ax=ax[0], color='g')
ax[0].set_title('Distribution of Transaction Amount')
ax[0].set_xlim(min(amount), max(amount))

sns.distplot(time, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transactions over Time')
ax[1].set_xlim(min(time), max(time))

#%% [markdown]
# The bulk of the amount of money in each transaction seem to be small amount with only few cases of large amounts. The time variable seems to be recorded in seconds starting from the first transaction of the day, thus the distribution spans over all the transactions occurred over about two days. It is reasonable to assume that the drop in trasactions that occurs approximately after 28 hours since the first transaction occurs during the night.
#%% [markdown]
# ## Scaling
# Since we are informed that most of our dataset is already scaled but we can notice that 'Time' and 'Amount' are not, we need to scale them. Scaling is important since many algorithm use distance to inform them, if we have features with different scales this can trick the algorithm.

#%%
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is more robust when dealing with outlyers and since there might be a few in 'Amount' let's use this one
rob_scaler = RobustScaler()

df['amount_scaled'] = rob_scaler.fit_transform(df.Time.values.reshape(-1, 1))
df['time_scaled'] = rob_scaler.fit_transform(df.Amount.values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)

df.head()

#%% [markdown]
# ## Splitting the original dataset
# 
# We will subsample our dataset later but we will want to test our model on the original dataset. Thus we need to divide into training and testing set now.

#%%
from sklearn.model_selection import StratifiedShuffleSplit

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)

for train_index, test_index in sss.split(X, y):
    
    print("TRAIN: ", train_index, " TEST: ", test_index)
    
    X_train_original, X_test_original = X.iloc[train_index], X.iloc[test_index]
    Y_train_original, Y_test_original = y.iloc[train_index], y.iloc[test_index]

#%% [markdown]
# ## Subsampling
# As mentioned above the dataset as it stands is heavily skewed and would lead to overfitting. We thus want to select a subset of data that is balanced and that we can use to train our model that later will be tested over the original dataset.

#%%
# How many samples we have classified as fraudolent?
n_fraud = df['Class'].value_counts()[1]

fraud = df[df['Class'] == 1]
no_fraud = df[df['Class'] == 0]

# Randomly select a subsample of non fraudolent activities of the same size as the fraudolent one
selected = no_fraud.sample(n_fraud)
selected.head()


#%%
# Concatenate the samples in the final subsample
selected.reset_index(drop=True, inplace=True)
fraud.reset_index(drop=True, inplace=True)

subsample = pd.concat([selected, fraud])

# Let's shuffle the new subsample
subsample = subsample.sample(frac=1).reset_index(drop=True)
subsample.head()


#%%
# Let's check the 'Class' category
subsample['Class'].value_counts().plot(kind='bar', log=True)
plt.xticks(np.arange(2), ['No Fraud', 'Fraud'], rotation='horizontal')

subsample['Class'].value_counts()

#%% [markdown]
# ## Correlation matrix study
# 
# Now that we have a balanced dataset we can study the correlation between the features.

#%%
corr = subsample.corr()

plt.figure(figsize=(12,10))
heat = sns.heatmap(data=corr, cmap='coolwarm_r')
plt.title('Heatmap of Correlation')

#%% [markdown]
# As can be seen from the heat map some of the features are correlated with fraud transactions. In particular V1, V3, V5, V6, V7, V9, V10, V12, V14, V16, V17 and V18 are negatively correlated with the 'Class' feature, thus the lower any these are the more likely it is that the transaction will be classified as a fraud. On the other hand the V2, V4 and V11 are positively correlated with the 'Class' feature meaning that the higher any of these are, the less likely it is that the transaction will be identified as a fraud.
# 
# Let us now see in more details the various correlations.

#%%
f, axes = plt.subplots(ncols=4, nrows=3, figsize=(20,20))

# Negative correlations with 'Class'
neg_cor = ('V1', 'V3', 'V5', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16', 'V17', 'V18')

i = 0
j = 0

for var in neg_cor:

    sns.boxplot(x="Class", y=var, data=subsample, palette='Set1', ax=axes[i, j])
    axes[i, j].set_title('{} vs Class Negative Correlation'.format(var))
    
    j += 1
    
    if j > 3:
        
        i += 1
        j = 0


#%%
f, axes = plt.subplots(ncols=3, figsize=(20,7))

# Positive correlations with 'Class'
pos_cor = ('V2', 'V4', 'V11')

for i, var in enumerate(pos_cor):
    
    sns.boxplot(x="Class", y=var, data=subsample, palette='Set1', ax=axes[i])
    axes[i].set_title('{} vs Class Negative Correlation'.format(var))

#%% [markdown]
# ## Outliers removal
# 
# In [this website](https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/) two methods to spot outliers are illustrated. Here the interquartile range method (IQR) will be used, in this method the IQR is calculated as the difference between the 75th and the 25th percentile. The IQR is then multiplyed by a factor k and data points above the k*IQR are considered outliers. The k factor is usually 1.5 but higher values can be used in case of extreme outliers.  

#%%
features = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18')

# This will store the lower and maximum threshold values for data points removal
kiqr_min_max = {}

# The k factor
k = 2.5

# Let's drop from the database the outliers
for i in features:
    
    Q25 = subsample[i].quantile(0.25)
    Q75 = subsample[i].quantile(0.75)
    
    IQR = Q75 - Q25

    min_val = Q25 - k * IQR
    max_val = Q75 + k * IQR
    
    kiqr_min_max[i] = (min_val, max_val)
    
    subsample_drop = subsample[~((subsample[i] < min_val) | (subsample[i] > max_val))]


#%%
f, axes = plt.subplots(ncols=2, figsize=(20,10))

sns.boxplot(x="Class", y='V1', data=subsample, palette='Set1', ax=axes[0])
axes[0].set_title('V1 vs Class Negative Correlation\nBefore dropping outliers')
    
sns.boxplot(x="Class", y='V1', data=subsample_drop, palette='Set1', ax=axes[1])
axes[1].set_title('V1 vs Class Negative Correlation\nAfter dropping outliers')
    

#%% [markdown]
# ## Dimesionality reduction
# 
# In order to visualize the classes, we can use some dimensionality reduction techniques. Here the t-SNE is used, watch [this video](https://www.youtube.com/watch?v=NEaUSP4YerM) for more information. This method allows to visualize in a lower  dimensional space the clusters present in the original features space. 

#%%
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches

X = subsample_drop.drop('Class', axis=1)
y = subsample_drop['Class']

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

truncatedSVD = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42)
X_svd = truncatedSVD.fit_transform(X)


#%%
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)
ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])

ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()

#%% [markdown]
# ## Classifiers
# 
# Let us now train four different classifiers in order to better distinguish between fraudolent and non fraudolent activities and later decide which one performs better.

#%%
# Let us plit the sample
from sklearn.model_selection import train_test_split

X = subsample_drop.drop('Class', axis=1)
y = subsample_drop['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%%
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# import the classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(),
    KNeighborsClassifier()]

for classifier in classifiers:
    
    pipeline = Pipeline(steps=[('classifier', classifier)])
    
    pipeline.fit(X_train, y_train)
    
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", 
          round(training_score.mean(), 2) * 100, "% accuracy score")


#%%



