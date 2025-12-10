# Causal_Inference_Final_Project

ADSP32029_Final Project: Lalonde (1986) Training Program's Effect on Earnings PotentialThis project analyzes the impact of a job training program studied by Robert LaLonde (1986) on the subsequent earnings of participants. It employs several causal inference techniques to estimate the Average Treatment Effect (ATE).

The analysis uses the classic LaLonde dataset, which combines experimental (NSW) and non-experimental (PSID) control group data.
File: lalonde.csv (must be present in the working directory)

The code is written in Python and requires the following libraries:

import warnings
import pandas as pd
import numpy as np
import graphviz as gr
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

2. Reproduction Steps
(Load Data)
lalonde = pd.read_csv('lalonde.csv')
lalonde = lalonde.drop('ID', axis= 1)

(Statistical Data)
##Summary Statistics

control_df = lalonde[lalonde.treat==0]
treatment_df = lalonde[lalonde.treat==1]

lalonde.groupby('treat')['re78'].agg(['count', 'min', 'max', 'median','mean'])

X = ['age', 'educ','re74', 're75']
mu = lalonde.groupby('treat')[X].mean()
var = lalonde.groupby('treat')[X].var()

norm_diff = (mu - mu.loc[0]) / np.sqrt((var + var.loc[0]) / 2)

(Visualizations)

sns.set_theme(style="whitegrid", context="talk")

# --- Age Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(lalonde["age"], kde=True, bins=15, edgecolor="black")
plt.title("Distribution of Age", fontsize=20, weight="bold")
plt.xlabel("Age", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.tight_layout()
plt.show()

# --- Education Distribution ---
plt.figure(figsize=(10, 6))
sns.histplot(lalonde["educ"], kde=True, bins=15, edgecolor="black")
plt.title("Distribution of Education Level (Years)", fontsize=20, weight="bold")
plt.xlabel("Years of Education", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.tight_layout()
plt.show()

# --- Age Distribution by Treatment Group ---
plt.figure(figsize=(10, 6))
sns.histplot(data=lalonde, x="age", hue="treat", kde=True, bins=15,
             edgecolor="black", palette="Set1", alpha=0.6)
plt.title("Distribution of Age by Treatment Group", fontsize=20, weight="bold")
plt.xlabel("Age", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.tight_layout()
plt.show()

# --- Education Distribution by Treatment Group ---
plt.figure(figsize=(10, 6))
sns.histplot(data=lalonde, x="educ", hue="treat", kde=True, bins=15,
             edgecolor="black", palette="Set1", alpha=0.6)
plt.title("Distribution of Education Level (Years) by Treatment Group", fontsize=20, weight="bold")
plt.xlabel("Years of Education", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.tight_layout()
plt.show()

# --- Boxplot of re78(Control vs Treated) ---

data["treatment_group"] = data["treat"].map({1: "Treated", 0: "Control"})
plt.figure(figsize=(10, 6))
sns.boxplot(
    data=data,
    x="treatment_group",
    y="re78",
    ##palette="pastel",
    width=0.25
)
plt.title("Real Earnings in 1978 (re78) by Treatment Group", fontsize=20, weight="bold")
plt.xlabel("Group", fontsize=16)
plt.ylabel("Earnings in 1978 (re78)", fontsize=16)

plt.ticklabel_format(axis='y', style='plain')
plt.tight_layout()
plt.show()

# --- Distribution of real income between treated and control ---

group_labels = {0: "Control", 1: "Treated"}
data["group"] = data["treat"].map(group_labels)

variables = ["re78", "re75", "re74"]
titles = {
    "re78": "Earnings in 1978 (re78)",
    "re75": "Earnings in 1975 (re75)",
    "re74": "Earnings in 1974 (re74)"
}
fig, axes = plt.subplots(1, 3, figsize=(28, 8), sharey=False)

for ax, var in zip(axes, variables):
    for group, subset in data.groupby("group"):
        ax.hist(
            subset[var],
            bins=20,
            alpha=0.6,
            label=group,
            edgecolor="black"
        )
    ax.set_title(titles[var], fontsize=20, weight="bold")
    ax.set_xlabel("Earnings", fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.ticklabel_format(axis='x', style='plain')
    ax.legend(title="Group")

plt.tight_layout()
plt.show()

(Regression Adjustment)

model = smf.ols('re78 ~ treat', data=lalonde).fit()

model = smf.ols('re78 ~ treat + age + educ + black + hispan + married + nodegree + re74 + re75', data=lalonde).fit()

model = smf.ols('re78 ~ treat + age + C(educ) + C(black) + C(hispan) + C(married) + C(nodegree) + re74 + re75', data=lalonde).fit()

FWL_lalonde = lalonde.copy()
debiasing_model = smf.ols(
    'treat ~ age + educ + black + hispan + married + nodegree + re74 + re75',
    data = FWL_lalonde
).fit()

FWL_lalonde['treat_res'] = debiasing_model.resid

denoising_model = smf.ols(
    're78 ~ age + educ + black + hispan + married + nodegree + re74 + re75',
    data = FWL_lalonde
).fit()

FWL_lalonde['re78_res'] = denoising_model.resid

final_model = smf.ols(
    're78_res ~ treat_res',
    data = FWL_lalonde
).fit()

(Propensity Score)

pre_treatment_vars = ['age', 'educ', 'married', 'nodegree', 'hispan', 'black', 're74', 're75']

X = lalonde[pre_treatment_vars]
y = lalonde['treat']

log_reg = LogisticRegression()
log_reg.fit(X, y)

lalonde_ps = lalonde

lalonde_ps['propensity_score_logistic'] = log_reg.predict_proba(X)[:, 1]

def calculate_matched_treatment_effect(df):
    treated = df[df['treat'] == 1].copy()
    control = df[df['treat'] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score_logistic']])
    distances, indices = nn.kneighbors(treated[['propensity_score_logistic']])

    matched_control_indices = indices.flatten()
    new_control = control.iloc[matched_control_indices]

    matched_data = pd.concat([treated, new_control])

    treated_mean = matched_data[matched_data['treat'] == 1]['re78'].mean()
    matched_contol_mean = matched_data[matched_data['treat'] == 0]['re78'].mean()

    treatment_effect = treated_mean - matched_contol_mean

    return treatment_effect


def bootstrap_ci(df, B=1000, alpha=0.05):
    treatment_effects = []
    N = len(df)

    for i in range(B):
        bootstrap_sample = df.sample(n=N, replace=True)

        try:
            effect = calculate_matched_treatment_effect(bootstrap_sample)
            treatment_effects.append(effect)
        except:
            pass

    effects_array = np.array(treatment_effects)

    lower_bound = np.percentile(effects_array, 100 * (alpha / 2))
    upper_bound = np.percentile(effects_array, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound

original_estimate = calculate_matched_treatment_effect(lalonde_ps)
lower, upper = bootstrap_ci(lalonde_ps, B=1000)

def compute_ipw_ate(df):
    df = df.copy()

    df['ipw'] = np.where(df['treat'] == 1,
                         1 / df['propensity_score'],
                         1 / (1 - df['propensity_score']))

    treated = df.query("treat == 1")
    control = df.query("treat == 0")

    weighted_mean_treated = np.sum(treated['re78'] * treated['ipw']) / np.sum(treated['ipw'])
    weighted_mean_control = np.sum(control['re78'] * control['ipw']) / np.sum(control['ipw'])

    return weighted_mean_treated - weighted_mean_control

B = 1000
boot_ates = []

n = data_ps.shape[0]

for b in range(B):
    sample_df = data_ps.sample(n, replace=True)
    ate_b = compute_ipw_ate(sample_df)
    boot_ates.append(ate_b)

boot_ates = np.array(boot_ates)

lower = np.percentile(boot_ates, 2.5)
upper = np.percentile(boot_ates, 97.5)

ipw_ate = compute_ipw_ate(data_ps)

(Diff in Diff)

df = lalonde_ps.copy()
n = len(df)
## X is the pretreatment("re74", "re75", (df["re74"] + df["re75"]) / 2)

df_long = pd.DataFrame({
    "id": np.repeat(df.index, 2),
    "treat": np.repeat(df["treat"].to_numpy(), 2),
    "time": np.tile([0, 1], n),
    "earnings": np.concatenate([df[X].to_numpy(),
                                df["re78"].to_numpy()])
})

group_means = (
    df_long.groupby(["treat", "time"])["earnings"]
           .mean()
           .reset_index()
)

Y11 = group_means[(group_means.treat == 1) & (group_means.time == 1)]["earnings"].values[0]
Y10 = group_means[(group_means.treat == 1) & (group_means.time == 0)]["earnings"].values[0]
Y01 = group_means[(group_means.treat == 0) & (group_means.time == 1)]["earnings"].values[0]
Y00 = group_means[(group_means.treat == 0) & (group_means.time == 0)]["earnings"].values[0]

did_formula = (Y11 - Y10) - (Y01 - Y00)

model = smf.ols("earnings ~ treat * time", data=df_long).fit()
did_ols = model.params["treat:time"]
conf_int = model.conf_int().loc["treat:time"]
lower_model, upper_model = conf_int[0], conf_int[1]

Contacts: ltan1@uchicago.edu  (347)-616-3902


