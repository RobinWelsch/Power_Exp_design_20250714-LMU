# Simulation-Based Statistical Inference for HCI – Python Tutorial with Visualizations
# -------------------------------------------------------------------
# This tutorial demonstrates how statistical power
# behaves for different statistical tests under different conditions.
# We visualize both the raw data and the p-value distributions.
# -------------------------------------------------------------------
# Key Concepts:
# - Type I error (α): False positive rate when H0 is true.
# - Power (1 - β): True positive rate when there is an effect.
# - Liberal test: Rejects too often (inflated α).
# - Conservative test: Rejects too rarely (low power).
# -------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, kruskal, mannwhitneyu
from statsmodels.stats.power import TTestIndPower

# Set pandas option for full DataFrame display
pd.set_option('display.max_columns', None)
# Set seed for reproducibility
np.random.seed(123)
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------------
# 1. Visualize raw simulated dataset from one simulation -----------------
# -------------------------------------------------------------------------

n = 12
x_example = np.random.normal(0, 1, size=n)
y_example = np.random.normal(0.5, 1, size=n)

df_example = pd.DataFrame({
    'value': np.concatenate([x_example, y_example]),
    'group': ['G1'] * n + ['G2'] * n
})

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_example, x='group', y='value', showfliers=False)
sns.stripplot(data=df_example, x='group', y='value', color='gray', alpha=0.7, size=4, jitter=True)
plt.title('Raw data: effect size ~.5')
plt.ylabel('Value')
plt.xlabel('')
plt.show()

# -------------------------------------------------------------------------
# 2. Power analysis for two-sample t-test ---------------------------------
# -------------------------------------------------------------------------

d = 0.5       # Cohen's d
alpha = 0.05  # significance level
nsim = 100    # number of simulations

ns = np.arange(5, 85, 5)

# 2.1 Simulation-based power
sim_powers = []
for n_i in ns:
    pvals = []
    for _ in range(nsim):
        x = np.random.normal(0, 1, size=n_i)
        y = np.random.normal(d, 1, size=n_i)
        _, p = ttest_ind(x, y, equal_var=True)
        pvals.append(p)
    sim_powers.append(np.mean(np.array(pvals) < alpha))

# 2.2 Analytical power
test_power = TTestIndPower()
ana_powers = test_power.solve_power(effect_size=d, nobs1=ns, alpha=alpha, ratio=1.0, alternative='two-sided')

power_df = pd.DataFrame({
    'n': np.concatenate([ns, ns]),
    'power': np.concatenate([sim_powers, ana_powers]),
    'method': ['Simulation'] * len(ns) + ['Analytical'] * len(ns)
})

plt.figure(figsize=(8, 5))
sns.lineplot(data=power_df, x='n', y='power', hue='method', style='method', markers=True)
plt.axhline(0.8, linestyle='--')
plt.title(f'Power vs. Sample Size (d = {d}, α = {alpha})')
plt.xlabel('Sample Size per Group (n)')
plt.ylabel('Power')
plt.legend(title='Method', loc='lower right')
plt.show()

# -------------------------------------------------------------------------
# 3. Non-parametric data: ordinal scale simulation ------------------------
# -------------------------------------------------------------------------

n_per_group = 12
shift_AB = 0.9
shift_C  = 0.45
cut_points = [-np.inf, -0.8, -0.25, 0.25, 0.8, np.inf]

def generate_ordinal(shift):
    latent = np.random.normal(loc=shift, scale=1, size=n_per_group)
    return np.digitize(latent, cut_points)

val_A = generate_ordinal(0)
val_B = generate_ordinal(shift_AB)
val_C = generate_ordinal(shift_C)

raw_df = pd.DataFrame({
    'group': ['A']*n_per_group + ['B']*n_per_group + ['C']*n_per_group,
    'value': np.concatenate([val_A, val_B, val_C])
})
raw_df['value'] = raw_df['value'].astype('category')

# 3a. Frequency counts
freq_table = raw_df.groupby(['group', 'value']).size().unstack(fill_value=0)
print("Frequency table of observed levels in each group:")
print(freq_table.to_string())

# 3b. Summary statistics
raw_df['value_int'] = raw_df['value'].cat.codes + 1
summary = raw_df.groupby('group')['value_int'].agg(median='median', Q1=lambda x: np.percentile(x, 25), Q3=lambda x: np.percentile(x, 75))
print("\nSummary statistics by group:")
print(summary.to_string())

# -------------------------------------------------------------------------
# 4. Distribution of ordinal responses ------------------------------------
# -------------------------------------------------------------------------

prop_df = raw_df.groupby(['group', 'value']).size().reset_index(name='count')
prop_df['proportion'] = prop_df.groupby('group')['count'].transform(lambda x: x / x.sum())

plt.figure(figsize=(6, 4))
sns.barplot(data=prop_df, x='group', y='proportion', hue='value')
plt.ylabel('Proportion of Responses')
plt.xlabel('Group')
plt.title('Distribution of Ordinal Responses by Group')
plt.legend(title='Response Level')
plt.show()

# -------------------------------------------------------------------------
# 5. Non-parametric tests on the single dataset ---------------------------
# -------------------------------------------------------------------------

a_vals = raw_df[raw_df['group']=='A']['value_int']
b_vals = raw_df[raw_df['group']=='B']['value_int']
c_vals = raw_df[raw_df['group']=='C']['value_int']

kw_stat, kw_p = kruskal(a_vals, b_vals, c_vals)
print(f"Kruskal-Wallis test (A, B, C): p-value = {kw_p:.3f}")

w_stat, w_p = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
print(f"Wilcoxon rank-sum test (A vs B): p-value = {w_p:.3f}")

# -------------------------------------------------------------------------
# 6. Simulation study: estimate power via repeated sampling ----------------
# -------------------------------------------------------------------------

n_sims = 1000
pvals_wilcox = []
pvals_kruskal = []

#Your For Loop goes here

alpha = 0.05
power_wilcox = np.mean(np.array(pvals_wilcox) < alpha)
power_kruskal = np.mean(np.array(pvals_kruskal) < alpha)

power_results = pd.DataFrame({
    'test': ['Wilcoxon rank-sum', 'Kruskal-Wallis'],
    'empirical_power': [power_wilcox, power_kruskal]
})
print("\nEstimated statistical power (α = 0.05):")
print(power_results.to_string())

# -------------------------------------------------------------------------
# 7. Simulation-based power curves for non-parametric tests over n --------
# -------------------------------------------------------------------------

np.random.seed(456)

# 7a. Define range of sample sizes and number of sims
n_sims     = 1000
ns_vec     = np.arange(5, 85, 5)
alpha      = 0.05

# 7b. Re-use your generate_ordinal() function, but now vectorized for arbitrary n:
def generate_ordinal(shift, n):
    latent = np.random.normal(loc=shift, scale=1, size=n)
    return np.digitize(latent, cut_points)

# 7c. Loop over sample sizes and collect empirical power
# You code here
# 7d. Plot the power curves
# You code here


