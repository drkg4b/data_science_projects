# %% [markdown]
# #Basic Survival Analysis
# The aim of this analysis is to identify whether there exist subgroups of veterans with lung cancer that differ in survival times and try to predict their survival times.

# %%
# imports
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
import seaborn as sns
import matplotlib.pyplot as plt
import sksurv
import pandas as pd
import numpy as np

plt.style.use(['ggplot', 'seaborn'])

# %%
# Import dataset
df = pd.read_csv('data/lung_cancer.csv')

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
df.describe(include=['O'])

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

# %%
g = sns.distplot(df.loc[df['Treatment'] == 'standard',
                        'Age_in_years'], color='blue')
g = sns.distplot(df.loc[df['Treatment'] == 'test',
                        'Age_in_years'], color='red')

g.set_ylabel('Frequency')

g = g.legend(['Standard', 'Test'], title='Treatment')

# %%
# Let us consider only the veterans who experienced the death event.
_ = sns.barplot(x='Age_in_decades', y='Survival_in_days',
                data=df[df['Status']], palette='muted', hue='Treatment')

# %% [markdown]
# - Seems like there is no difference in age distributions for the different treatments.
# - Veterans in their 50s and 70s seems to benefit the most from the test drug while for those in their 30s the standard one seems to be more effective. In the other cases there doesn't appear to be a significant advantage of one treatment above the other.

# %% [markdown]
# Does the celltype affects the survival time? Or what if they had previous therapy?

# *note to self*: Get back here with some more visualizations.

# %% [markdown]
# ####Survival analysis with Kaplan-Meier estimator

# %%
kmf = KaplanMeierFitter()

T = df['Survival_in_days']
E = df['Status']

kmf.fit(T, event_observed=E)

# %%
# Plot the K-M survival function
kmf.plot()

_ = plt.ylabel("est. probability of survival $\hat{S}(t)$")
_ = plt.xlabel("time $t$")

# %%
print(kmf.median_)

# %% [markdown]
# Looks like only 20% of the veterans survives after 200 days while 50% of them die after 80 days.
#
#  ####Stratification by treatment
# Let us now see how the survival curves vary according to treatment.

# %%
therapy = df['Treatment'] == 'standard'

ax = plt.subplot(111)

kmf.fit(T[therapy], event_observed=E[therapy], label='Standard Treatment')

_ = kmf.plot(ax=ax)

print('Median survival time for standard treatment:', kmf.median_)

kmf.fit(T[~therapy], event_observed=E[~therapy], label='Test Treatment')

_ = kmf.plot(ax=ax, color='red')

print('Median survival time for test treatment:', kmf.median_)

_ = plt.ylim(0, 1)

_ = plt.ylabel("est. probability of survival $\hat{S}(t)$")
_ = plt.xlabel("time $t$")

# %% [markdown]
# Even given the huge gap between median survival times, at first sight the two treatments don't look to be different but we need to perform a log rank test to be sure.

# %%
# Let us perform a logrank test:
# H_0 : There is no difference in treatment
results = logrank_test(T[therapy], T[~therapy],
                       E[therapy], E[~therapy], alpha=.95)

results.print_summary()                    

# %% [markdown]
# The p-value is not significant thus we cannot reject the null hypothesis.
