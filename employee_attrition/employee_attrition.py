# %%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import itertools
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['ggplot', 'seaborn'])

# %%
in_dir = 'data'

in_data = os.path.join(in_dir, 'employee-attrition.csv')

df = pd.read_csv(in_data)

# %%
df.info()

# %% [markdown]
# there aren't any missing values and out target variable appears to be categorical.

# %%
df.describe().T

# %%
df.sample(10)

# %% [markdown]
# There are only two possible values for the target variable and it is highly imbalanced, will need to balance it before training the model. Let us transform it into numeric.

# %%
data = df['Attrition'].value_counts()

_ = sns.barplot(data.index, data.values, palette='muted')

# %%
df.loc[df['Attrition'] == 'Yes', 'Attrition'] = 1
df.loc[df['Attrition'] == 'No', 'Attrition'] = 0

# %% [markdown]
# Let us check correlation between variables.

# %%
corr = df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', linewidths=.5, ax=ax)

# %%
abs(corr['Attrition']) > 0.5

# %% [markdown]
# There don't seem to be high correlation between any of the variables and the target one but some features are highly correlated with each other and worth investigating more to see if they can be dropped. In particular:
# - JobLevel almost has perfect correlation with MonthlyIncome
# - EmployeeCount and StandardHours have the same number in it and can probably be dropped from the dataset.
# - Age higly correlates with JobLevel, MonthlyIncome and TotalWorkingYears
# - JobLevel highly correlates with TotalWorkingYears and YearsAtCompany
# - MonthlyIncome highly correlates with TotalWorkingYears and YearsAtCompany
# - PercentSalaryHike highly correlates with PerformanceRating

# Let us check the categorical features

# %%
df.describe(include=['O'])

# %% [markdown]
# It appears that Over18 only have one value and can be dropped from the dataset.

# %%
to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']

df.drop(columns=to_drop, inplace=True)

# %%


def PlotDists(feature, position):
    '''
    '''
    g = sns.factorplot(x=feature, y='Attrition',
                       data=df, palette='muted', kind='bar', size=6, ax=position)

    g.despine(left=True)

    g = g.set_ylabels('Attrition probability')

    # This is needed, see: https://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots
    plt.close(g.fig)


# %%
to_plot = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
           'JobRole', 'MaritalStatus', 'OverTime']

fig, ax = plt.subplots(4, 2, figsize=(20, 20), sharex=False, sharey=False)

# Flatten out the axis object
ax = ax.ravel()

for i in range(7):

    PlotDists(to_plot[i], ax[i])

plt.tight_layout()
plt.show()

# %% [markdown]
# - Looks like that people who trave more frequently are more likely to quit compared to those who don't travel or travel rarely.
# - People in the sales department are more likely to quit although HR has a high standard deviation.
# - Male quit more often than women.
# - Sales representatives have the highest probability to quit.
# - Singles are more likely to quit compared to married or divorced employees.
# - People doing overtime have a high probability to quit.
#
# Would be nice to study more the relationship between the features but for time constraints I will come back to it if I have some time left.

# %%
# Get numerical and categorical features
numerical = df.select_dtypes(exclude=['object'])
categorical = df.select_dtypes(['object'])

# %% [markdown]
# Convert BusinessTravel into an ordinal categorical variable since there is intrinsic order between non, rarely and frequently.
# One hot encode the remaining variables.
travel_map = {'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2}

df['BusinessTravel'] = df['BusinessTravel'].map(travel_map)

# One hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

# %% [markdown]
# We saw that the Attrition class is imbalanced. As a consequence accuracy is not a good metric anymore and we should use other metrics when comparing models such as recall, f1-score or AUC. Moreover most algorithms will learn patterns that will be biased towards the majority class.

# We can deal with imbalanced classes by:
# - Assign a bigger penalty to wrong predictions from the minority class.
# - Upsampling the minority class or downsampling the majority one.
# - Simulate more data.

# A penalty for wrong predictios for the minority class can be assigned in some of the algorithms using the `class_weight` parameter.
# We need to split our dataset before oversampling because in this case, the same observation can be repeated in both the training and test sets causing the model to simply learn some specific datapoints and thus overfit.

# %%
# Split the dataset
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y)

# %%
# Oversampling the minority class
X_train_up, y_train_up = resample(X_train[y_train == 1],
                                  y_train[y_train == 1],
                                  replace=True,
                                  n_samples=X_train[y_train == 0].shape[0],
                                  random_state=1)

X_train_up = pd.concat([X_train[y_train == 0], X_train_up])
y_train_up = pd.concat([y_train[y_train == 0], y_train_up])

# %%
# Downsample majority class
X_train_dw, y_train_dw = resample(X_train[y_train == 0],
                                  y_train[y_train == 0],
                                  replace=True,
                                  n_samples=X_train[y_train == 1].shape[0],
                                  random_state=1)

X_train_dw = pd.concat([X_train[y_train == 1], X_train_dw])
y_train_dw = pd.concat([y_train[y_train == 1], y_train_dw])

# %%
# Check the shapes of the classes
print("Original shape:", X_train.shape, y_train.shape)
print("Upsampled shape:", X_train_up.shape, y_train_up.shape)
print("Downsampled shape:", X_train_dw.shape, y_train_dw.shape)

# %%
# Check the principal components

pca = PCA(n_components=None, svd_solver="full")

scaler = StandardScaler()
scaler = scaler.fit_transform(X_train)

pca.fit(scaler)

cum_var_exp = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(12, 6))

n_features = len(cum_var_exp) + 1

plt.bar(range(1, n_features), pca.explained_variance_ratio_, align="center",
        color='magenta', label="Individual explained variance")

plt.step(range(1, n_features), cum_var_exp, where="mid",
         label="Cumulative explained variance", color='blue')

plt.xticks(range(1, n_features))
plt.legend(loc="best")

plt.xlabel("Principal component index", {"fontsize": 14})
plt.ylabel("Explained variance ratio", {"fontsize": 14})
plt.title("PCA on training data", {"fontsize": 16})

# %%
print('We need', np.where(cum_var_exp > 0.90)[0][0], 'features to explain 90% of the variation of the data.')
print('We need', np.where(cum_var_exp > 0.95)[0][0], 'features to explain 95% of the variation of the data.')
print('We need', np.where(cum_var_exp > 0.99)[0][0], 'features to explain 99% of the variation of the data.')

#%% [markdown]
# Since there are some redundant features we can use some algorithm to make a rank of the feature importance and decide which one we should remove.


# %%

lr = LogisticRegression()

lr.fit(X_train, y_train)

# %%
# Predict on test set
lr_pred = lr.predict(X_test)

# %% [markdown]
# Accuracy can be misleading when dealing with imbalanced classes, we can use instead:
# - Confusion Matrix: a table showing correct predictions and types of incorrect predictions.
# - Precision: the number of true positives divided by all positive predictions. It is a measure of a classifier's exactness. Low precision indicates a high number of false positives.
# - Recall or true positive rate: the number of true positives divided by the number of positive values in the test data. It is a measure of a classifier's completeness. Low recall indicates a high number of false negatives.
# - F1 Score: the weighted average of precision and recall.

# Since our main objective with the dataset is to prioritize accuraltely classifying fraud cases the recall score can be considered our main metric to use for evaluating outcomes.

# %%
# Check some metrics
accuracy_score(y_test, lr_pred)

# %%
f1_score(y_test, lr_pred)

# %%
cm_lr = pd.DataFrame(confusion_matrix(y_test, lr_pred), index=[
                     'Attrition', 'No Attrition'], columns=['Attrition', 'No Attrition'])

_ = sns.heatmap(cm_lr, cmap='coolwarm', annot=True,
                fmt='g', linewidths=.5, cbar=False)

# %%
recall_score(y_test, lr_pred)

# %%
lr.feature_importances_

# %% [markdown]
# Next steps:
# - Implement RandomForrest
# - Use eithr random forrest of logistic regression to get a ranking of the variables and exclude redundant ones
# - Scale first just numeric and after numeric plus encoded variables to see performance
# Balance classes: oversampling, undersampling and penalyzing classes with `class_weight`
# PCA to see if we can reduce the dataset
# Train several models and evaluate the best performing ones.
# Use cross validation and GridSearchCV or RandomizedSearchCV.


# %% [markdown]
#
