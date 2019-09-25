# %% [markdown]
# #Basic Survival Analysis
# The aim of this analysis is to identify whether there exist subgroups of veterans with lung cancer that differ in survival times and try to predict their survival times. 

# %%
# imports
import seaborn as sns
from sksurv.datasets import load_veterans_lung_cancer
import matplotlib.pyplot as plt
import sksurv
import pandas as pd
import numpy as np

plt.style.use(['ggplot', 'seaborn'])

# %%
# Import dataset

# %%
data_x, data_y = load_veterans_lung_cancer()

# %%
# Get a dataframe out of the data
df = pd.DataFrame(data_y)

df = pd.concat([data_x, df], axis=1)

# %% [markdown]
# EDA

# %%
df.info()

# %% [markdown]
# - There aren't any missing values.
# - We have a few categorical features (Celltype, Prior_therapy, Treatment)

# %%
# Inferential statistics
df.describe()

# %% [markdown]
# - Most, ~75%, veterans are relatively young with less than 66 years.
# - The [Karnofsky Performance Score (KPS)](https://en.wikipedia.org/wiki/Performance_status#Karnofsky_scoring) ranking runs from 100 to 0, where 100 is "perfect" health and 0 is death. It appears that very little veterans, ~25%, had a KPS greater than 70 and were thus able to take care of themselves.
# - Only less than 25% of the veterans remained into the study for more than 11 moths.
# On average people in the study survived for about 4 months.

# %%
df.describe(include=['category'])

# %% [markdown]
# - Celltype has 4 possible values of which the 'smallcell' is the most frequent.
# - About 70% of the patients didn't undertake any previous therapy.
# - About half of the veterans had a standard treatment drug, the remaining ones had the test drug.

# %% [markdown]
# ####Age
# How does age affect the survival of the veterans with cancer? Does youger patients have better chances? Is treatment more effective depending on age? Let's explore a bit!

# %%
g = sns.distplot(df.loc[df['Status'], 'Age_in_years'], color='red')
g = sns.distplot(df.loc[~df['Status'], 'Age_in_years'], color='blue')

g.set_ylabel('Frequency')

g = g.legend(['Death Event', 'Censored'])

# %%
# Let's create a categorical variable for age
df = df.assign(Age_in_decades=pd.cut(df.Age_in_years,
                                     bins=[30, 40, 50, 60, 70, 80, 90],
                                     labels=['30', '40', '50', '60', '70', '80']))

# %%
g = sns.barplot(x='Age_in_decades', y='Survival_in_days',
                data=df, palette='muted', hue='Status')

h, l = g.get_legend_handles_labels()

g = g.legend(h, ['No', 'Yes'], title='Experienced Death Event')

# %% [markdown]
# - It appears that older veterans are more likely to experience the death event while younger ones to be censored. 
# - Veterans between 40 and 60 years of age tend to live longer and also stay longer into the study.

#%%
g = sns.distplot(df.loc[df['Treatment'] == 'standard', 'Age_in_years'], color='blue')
g = sns.distplot(df.loc[df['Treatment'] == 'test', 'Age_in_years'], color='red')

g.set_ylabel('Frequency')

g = g.legend(['Standard', 'Test'], title='Treatment')

#%%
# Let us consider only the veterans who experienced the death event.
_ = sns.barplot(x='Age_in_decades', y='Survival_in_days',
                data=df[df['Status']], palette='muted', hue='Treatment')

#%% [markdown]
# - Seems like there is no difference in age distributions for the different treatments.
# - Veterans in their 50s and 70s seems to benefit the most from the test drug while for those in their 30s the standard one seems to be more effective. In the other cases there doesn't appear to be a significant advantage of one treatment above the other.

#%%
