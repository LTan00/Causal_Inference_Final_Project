# Causal Inference Final Project

**ADSP32029 Final Project: Lalonde (1986) Training Program's Effect on Earnings Potential**  

This project analyzes the impact of a job training program studied by Robert LaLonde (1986) on participants' subsequent earnings. We employ several **causal inference techniques** to estimate the **Average Treatment Effect (ATE)**.  

The analysis uses the **LaLonde dataset**, which combines experimental (NSW) and non-experimental (PSID) control group data.  

**Dataset:** `lalonde.csv` (must be present in the working directory)

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dependencies](#dependencies)  
3. [Reproduction Steps](#reproduction-steps)  
    - [Load Data](#load-data)  
    - [Summary Statistics](#summary-statistics)  
    - [Visualizations](#visualizations)  
    - [Regression Adjustment](#regression-adjustment)  
    - [Propensity Score Matching (PSM)](#propensity-score-matching-psm)  
    - [Inverse Probability Weighting (IPW)](#inverse-probability-weighting-ipw)  
    - [Difference-in-Differences (DiD)](#difference-in-differences-did)  
4. [Contact](#contact)  

---

## Project Overview

We explore the causal effect of a **job training program** on real earnings in 1978 (`re78`). Techniques applied include:

- Regression Adjustment  
- Propensity Score Matching  
- Inverse Probability Weighting  
- Difference-in-Differences  

This allows for robust estimation of treatment effects using both experimental and observational data.

---

## Dependencies

The code is written in **Python** and requires the following libraries:

```python
import warnings
import pandas as pd
import numpy as np
import graphviz as gr
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

lalonde = pd.read_csv('lalonde.csv')
lalonde = lalonde.drop('ID', axis=1)

control_df = lalonde[lalonde.treat == 0]
treatment_df = lalonde[lalonde.treat == 1]

# Summary of earnings
lalonde.groupby('treat')['re78'].agg(['count', 'min', 'max', 'median','mean'])

# Covariate means and variances
X = ['age', 'educ', 're74', 're75']
mu = lalonde.groupby('treat')[X].mean()
var = lalonde.groupby('treat')[X].var()

# Normalized differences
norm_diff = (mu - mu.loc[0]) / np.sqrt((var + var.loc[0]) / 2)

sns.set_theme(style="whitegrid", context="talk")

plt.figure(figsize=(10, 6))
sns.histplot(lalonde["age"], kde=True, bins=15, edgecolor="black")
plt.title("Distribution of Age", fontsize=20, weight="bold")
plt.xlabel("Age", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(lalonde["educ"], kde=True, bins=15, edgecolor="black")
plt.title("Distribution of Education Level (Years)", fontsize=20, weight="bold")
plt.xlabel("Years of Education", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.show()

sns.histplot(data=lalonde, x="age", hue="treat", kde=True, bins=15, edgecolor="black", palette="Set1", alpha=0.6)
plt.title("Distribution of Age by Treatment Group", fontsize=20, weight="bold")
plt.show()

sns.histplot(data=lalonde, x="educ", hue="treat", kde=True, bins=15, edgecolor="black", palette="Set1", alpha=0.6)
plt.title("Distribution of Education Level by Treatment Group", fontsize=20, weight="bold")
plt.show()

data["treatment_group"] = data["treat"].map({1: "Treated", 0: "Control"})
sns.boxplot(data=data, x="treatment_group", y="re78", width=0.25)
plt.title("Earnings in 1978 by Treatment Group", fontsize=20, weight="bold")
plt.show()

# Simple regression
model = smf.ols('re78 ~ treat', data=lalonde).fit()

# Multiple regression with covariates
model = smf.ols('re78 ~ treat + age + educ + black + hispan + married + nodegree + re74 + re75', data=lalonde).fit()

FWL_lalonde = lalonde.copy()
debiasing_model = smf.ols('treat ~ age + educ + black + hispan + married + nodegree + re74 + re75', data=FWL_lalonde).fit()
FWL_lalonde['treat_res'] = debiasing_model.resid
denoising_model = smf.ols('re78 ~ age + educ + black + hispan + married + nodegree + re74 + re75', data=FWL_lalonde).fit()
FWL_lalonde['re78_res'] = denoising_model.resid

final_model = smf.ols('re78_res ~ treat_res', data=FWL_lalonde).fit()

pre_treatment_vars = ['age', 'educ', 'married', 'nodegree', 'hispan', 'black', 're74', 're75']
X = lalonde[pre_treatment_vars]
y = lalonde['treat']

log_reg = LogisticRegression()
log_reg.fit(X, y)

lalonde['propensity_score_logistic'] = log_reg.predict_proba(X)[:, 1]

def calculate_matched_treatment_effect(df):
    treated = df[df['treat'] == 1]
    control = df[df['treat'] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score_logistic']])
    distances, indices = nn.kneighbors(treated[['propensity_score_logistic']])

    matched_control = control.iloc[indices.flatten()]
    matched_data = pd.concat([treated, matched_control])

    return matched_data[matched_data['treat']==1]['re78'].mean() - matched_data[matched_data['treat']==0]['re78'].mean()

def compute_ipw_ate(df):
    df = df.copy()
    df['ipw'] = np.where(df['treat']==1, 1/df['propensity_score'], 1/(1-df['propensity_score']))
    treated_mean = np.sum(df[df['treat']==1]['re78']*df[df['treat']==1]['ipw']) / np.sum(df[df['treat']==1]['ipw'])
    control_mean = np.sum(df[df['treat']==0]['re78']*df[df['treat']==0]['ipw']) / np.sum(df[df['treat']==0]['ipw'])
    return treated_mean - control_mean

df_long = pd.DataFrame({
    "id": np.repeat(df.index, 2),
    "treat": np.repeat(df["treat"].to_numpy(), 2),
    "time": np.tile([0, 1], n),
    "earnings": np.concatenate([df[X].to_numpy(), df["re78"].to_numpy()])
})

model = smf.ols("earnings ~ treat * time", data=df_long).fit()
did_ate = model.params["treat:time"]
conf_int = model.conf_int().loc["treat:time"]

Contact
Email: ltan1@uchicago.edu

Phone: (347)-616-3902


