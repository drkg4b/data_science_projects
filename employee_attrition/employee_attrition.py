# %%
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
#
# Let us now split the data into training and testing set.

# %%
# Split the dataset
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

#%%
numerical = df.select_dtypes(exclude=['object'])
categorical = df.select_dtypes(['object'])

#%%
categorical

# %% [markdown]
# We want to split before attempting any oversampling because when oversampling the same observation can be repeated multiple times and we don't want to test our
