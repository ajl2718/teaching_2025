import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import interact, fixed

from datetime import timedelta

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score, precision_score, recall_score, confusion_matrix

from scipy.stats import norm, bernoulli, chi2_contingency, fisher_exact, mannwhitneyu, pearsonr as calculate_pearson_r
import scipy
from scipy import stats

from ipywidgets import interact, widgets
from ipywidgets import interactive, widgets

from IPython.display import display
from IPython.display import display_html 
import warnings


cols = ['icustay_los', 'age', 'time_elapsed_seconds', 'elixhauser_score', 'mean_Heart Rate',
       'mean_Respiratory Rate', 'mean_FiO2 Set', 'FiO2_Set',
       'Tidal Volume (Obser)', 'Arterial BP Mean', 'Albumin (>3.2)',
       'Arterial pH', 'Creatinine (0-1.3)', 'hgb', 'PaO2', 'PaCO2', 'WBC',
       'Respiratory SOFA Score', 'Overall SOFA Score', 'first_heart_rate', 'first_resp_rate']

# for module 11 activity 2
best_n_estimators = 100
best_max_depth = 5

def remove_outliers_iqr(df, column):
    """
    Given a dataframe, remove all values that are outside of 1.5*IQR
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Function to remove outliers using IQR method
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def make_pvalue_string(p_value):
    """
    Produce a range for the p-value
    """
    if p_value < 0.01:
        p_value_string = 'p-value < 0.01'
    elif p_value < 0.05:
        p_value_string = 'p-value < 0.05'
    else:
        p_value_string = 'p_value > 0.05'
    return p_value_string

def make_scatter_plot(df, x_variable, y_variable, correlation_coefficient, p_value):
    """
    Produce a scatterplot of x_variable against y_variable and show the correlation coefficient
    and p-value in the title of the chart
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x_variable], df[y_variable], alpha=0.5)
    plt.title(f'Scatter Plot of {x_variable} vs {y_variable}\n correlation = {correlation_coefficient}\n{p_value}')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)
    plt.show()

def plot_linear_regression(df, 
                           x_variable, 
                           y_variable,
                           x_min,
                           x_max,
                           y_min,
                           y_max,
                           num_points,
                           intercept, 
                           slope):
    """
    Make a scatter plot with a linear regression overlaid
    """
    x = np.linspace(x_min, x_max, num_points)

    x_vals, y_vals = df[x_variable], df[y_variable]

    # scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals, alpha=0.5)
    plt.title(f'Scatter Plot of {x_variable} vs {y_variable}')
    plt.xlabel(x_variable)
    plt.ylabel(y_variable)

    # y predicted from line of best fit
    y_preds = x_vals * slope + intercept
    
    # Plot the line based on the slope and intercept
    y_pred_vis = slope * x + intercept

    # calculate the R^2 and MSE
    mse = np.round(mean_squared_error(y_vals, y_preds), 3)
    r2 = np.round(r2_score(y_vals, y_preds), 3)
    
    plt.plot(x, y_pred_vis, color='red', label=f'Line: y = {slope:.2f}x + {intercept:.2f}\nMSE = {mse}\n$R^2$ = {r2}')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
                                  
# create a scatterplot
def scatter_plot(dataset, x_var, y_var):
    xvals, yvals = zip(*dataset.loc[:, [x_var, y_var]].dropna().values)
    
    # calculate Pearson correlation coefficient
    corr_coef, p_value = calculate_pearson_r(xvals, yvals)
    corr_coef = np.round(corr_coef, 2)
    r2 = np.round(corr_coef**2, 2)
    p_value_string = make_pvalue_string(p_value)

    sns.set_style('white')
    # produce the plot
    plt.figure(figsize=(6, 4))
    plt.scatter(xvals, yvals, color='b', alpha=0.7)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f'{x_var} vs {y_var}\n Pearson correlation coefficient = {corr_coef}\n$r^2$ = {r2}\n{p_value_string}')
    plt.show()
       
# Function to calculate correlations
def calculate_correlations(df, key_var, variables, method):
    correlations = []
    for var in variables:
        if var in df.columns:
            # Remove rows with NaN values for the current pair of variables
            valid_data = df[[key_var, var]].dropna()
            if method == 'pearson':
                corr, p_value = scipy.stats.pearsonr(valid_data[key_var], valid_data[var])
            elif method == 'spearman':
                corr, p_value = scipy.stats.spearmanr(valid_data[key_var], valid_data[var])
            correlations.append((var, corr, p_value))
        else:
            print(f"Warning: Variable '{var}' not found in the dataset.")
    return pd.DataFrame(correlations, columns=['Variable', 'Correlation', 'P-value'])

# Function to format p-values
def format_pvalue(pvalue):
    if pvalue < 0.001:
        return "< 0.001"
    else:
        return f"{pvalue:.3f}"
def calculate_ci(estimates, p):
    """
    Given a set of bootstrap estimates, calculate the p confidence interval

    Args
    ----
    estimates (list): list of estimates
    p (float): fraction between 0 and 1

    Returns
    -------
    results (3-tuple of floats): point estimate, left CI, right CI
    """
    
    num_estimates = len(estimates)
    left_ci, right_ci = int(num_estimates * ((1 - p) / 2)), int(num_estimates * (1 - (1 - p) / 2))
    estimates.sort()
    point_estimate = np.mean(estimates)
    estimates_l, estimates_r = estimates[left_ci], estimates[right_ci]
    results = (point_estimate, estimates_l, estimates_r)
    
    return results

def calculate_performance_stats(dataset, clf, num_iterations):
    """
    Calculate performance stats for a classifier

    Args
    ----
    dataset (pd.DataFrame): dataset used to extract predictors and outcome variables
    clf (sklearn Classifier): trained classifier
    num_iterations (int): number of bootstrap iterations to run

    Returns
    -------
    results (list of 3-tuples): all performance stats including point estimates and 95% CI
    """
    num_points = dataset.shape[0] 
    ppvs, npvs, sens, specs, aucs = [], [], [], [], []
    indices = np.arange(num_points)

    # bootstrap estimates
    for n in range(num_iterations):
        bootstrap_sample = np.random.choice(indices, num_points)
        dataset_temp = dataset.copy().iloc[bootstrap_sample, :]
        
        X_temp = dataset_temp.loc[:, dataset_temp.columns != 'fail_ext_coded'].values
        y_temp = dataset_temp.fail_ext_coded.values
        
        y_pred_prob_temp = clf.predict_proba(X_temp)
        y_pred_temp = clf.predict(X_temp)
        
        tn, fp, fn, tp = confusion_matrix(y_temp, y_pred_temp).ravel()
        
        ppv_temp, npv_temp = tp / (tp + fp), tn / (tn + fn)
        ppvs.append(ppv_temp)
        npvs.append(npv_temp)
        
        sens_temp, spec_temp = tp / (tp + fn), tn / (tn + fp)
        sens.append(sens_temp)
        specs.append(spec_temp)
        
        auc_temp = roc_auc_score(y_temp, y_pred_prob_temp[:,1])
        aucs.append(auc_temp)
    
    # 95% CIs
    ppv_estims, npv_estims = calculate_ci(ppvs, 0.95), calculate_ci(npvs, 0.95)
    sens_estims, spec_estims = calculate_ci(sens, 0.95), calculate_ci(specs, 0.95)
    auc_estims = calculate_ci(aucs, 0.95)
    
    results = [ppv_estims, npv_estims, sens_estims, spec_estims, auc_estims]
    return results


# This function will close a specific figure
def close_figure(i):
    plt.close(i)
    print(f"Figure {i+1} closed.")


############ Functions for BAR CHARTS ##############
# Function to create a bar chart for a categorical variable
def create_bar_chart(data, variable):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=variable, hue='fail_ext_status_bin')
    plt.title(f'Distribution of {variable} by Extubation Status')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Extubation Status', labels=['Successful', 'Failed'])
    plt.tight_layout()
    plt.show()


############### Function to create CROSS-TABS #############################3
# Function to create and format a cross-tabulation for both categorical and binary variables
def create_crosstab(data, var, target='fail_ext_status_bin'):
    ct = pd.crosstab(data[var], data[target], dropna=True)
    ct_pct = ct.div(ct.sum(axis=0), axis=1).round(4) * 100
    
    ct_combined = pd.DataFrame()
    for column in ct.columns:
        ct_combined[column] = ct[column].astype(str) + ' (' + ct_pct[column].map("{:.1f}%".format) + ')'
    
    # Add total row
    total = ct.sum()
    total_pct = total / total * 100
    total_row = total.astype(str) + ' (100.0%)'
    total_row.name = 'Total'
    ct_combined = pd.concat([ct_combined, total_row.to_frame().T])
    
    # Add total column
    ct_combined['Total'] = ct.sum(axis=1).astype(str) + ' (' + (ct.sum(axis=1) / ct.sum().sum() * 100).map("{:.1f}%".format) + ')'
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(ct)    
    
    return ct_combined, chi2, p_value


# Function to calculate statistical tests
def perform_statistical_tests(data1, data2):
    # T-test
    t_stat, t_p_value = stats.ttest_ind(data1, data2, equal_var=False)  # Using Welch's t-test
    
    # Mann-Whitney U test
    u_stat, u_p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    
    return {
        't_stat': t_stat,
        't_p_value': t_p_value,
        'u_stat': u_stat,
        'u_p_value': u_p_value
    }


def calculate_95_ci(data):
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)
    
    # Add median and IQR calculations
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    
    return pd.Series({
        'ci_lower': ci[0], 
        'ci_upper': ci[1],
        'median': median,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    })


def format_summary(summary):
    formatted = summary.copy()
    formatted.loc['count'] = formatted.loc['count'].astype(int)
    
    # Format numeric values
    for col in formatted.columns:
        for idx in formatted.index:
            if isinstance(formatted.loc[idx, col], (int, float)):
                formatted.loc[idx, col] = f"{float(formatted.loc[idx, col]):.2f}"
    
    # Combine CI into a single string
    formatted.loc['95% CI'] = (
        '(' + 
        formatted.loc['ci_lower'].astype(float).round(2).astype(str) + 
        ', ' + 
        formatted.loc['ci_upper'].astype(float).round(2).astype(str) + 
        ')'
    )
    
    # Format median and IQR
    formatted.loc['Median (IQR)'] = (
        formatted.loc['median'].astype(float).round(2).astype(str) + 
        ' (' + 
        formatted.loc['iqr'].astype(float).round(2).astype(str) + 
        ')'
    )
    
    # Remove separate rows
    formatted = formatted.drop(['ci_lower', 'ci_upper', 'median', 'q1', 'q3', 'iqr'])
    
    return formatted


# Create stylized tables
def style_table(df, caption):
    return df.style.set_properties(**{'text-align': 'center'}).set_table_styles([
        {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]},
        {'selector': 'caption', 'props': [('caption-side', 'top')]}
    ]).set_caption(caption)
    

############### Functions for HISTOGRAMS ###################
# Function to calculate Freedman-Diaconis bin width
def freedman_diaconis_bins(data):
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return 1  # fallback to 1 if IQR is 0
    bin_width = 2 * iqr * len(data) ** (-1/3)
    return bin_width

# Function to calculate SE and CI
def calc_se_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, se, ci

# # Function to remove outliers using IQR method
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]


def create_histo(df, variable_to_analyze):
    # Ensure the variable is numeric
    df[variable_to_analyze] = pd.to_numeric(df[variable_to_analyze], errors='coerce')
    
    # Remove any rows with NaN values
    df_clean = df.dropna(subset=[variable_to_analyze, 'ext_status_bin_num']).copy()
    
    # Group the data and calculate statistics
    groups = df_clean['ext_status_bin_num'].unique()
    group_stats = {}
    
    # Store group data for statistical tests
    group_data = {group: df_clean[df_clean['ext_status_bin_num'] == group][variable_to_analyze] 
                 for group in groups}
    
    # Perform statistical tests if exactly two groups
    if len(groups) == 2:
        test_results = perform_statistical_tests(group_data[groups[0]], group_data[groups[1]])
    
    for group in groups:
        data = group_data[group]
        mean, se, ci = calc_se_ci(data)
        median = np.median(data)
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        
        group_stats[group] = {
            'mean': mean,
            'count': len(data),
            'std': np.std(data, ddof=1),
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'median': median,
            'iqr': iqr
        }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Boxplot
    boxplot_data = [group_data[group] for group in groups]
    ax1.boxplot(boxplot_data, tick_labels=[f'Group{group}' for group in groups])
    ax1.set_title(f'Boxplot of {variable_to_analyze} by ext_status_bin_num')
    ax1.set_xlabel('Group')
    ax1.set_ylabel(variable_to_analyze)
    
    # Histogram
    for group in groups:
        data = group_data[group]
        ax2.hist(data, bins='auto', alpha=0.5, label=f'Group {group}', density=True)
    ax2.set_title(f'Histogram of {variable_to_analyze} by ext_status_bin_num')
    ax2.set_xlabel(variable_to_analyze)
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print group statistics and test results
    for group, stats in group_stats.items():
        print(f"\nGroup {group}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  IQR: {stats['iqr']:.2f}")
        print(f"  Count: {stats['count']}")
        print(f"  Standard Deviation: {stats['std']:.2f}")
        print(f"  Standard Error: {stats['se']:.2f}")
        print(f"  95% CI: ({stats['ci_lower']:.2f}, {stats['ci_upper']:.2f})")
    
    if len(groups) == 2:
        print("\nStatistical Tests:")
        print(f"Independent t-test:")
        print(f"  t-statistic: {test_results['t_stat']:.3f}")
        print(f"  p-value: {test_results['t_p_value']:.4e}")
        print("\nMann-Whitney U test:")
        print(f"  U-statistic: {test_results['u_stat']:.3f}")
        print(f"  p-value: {test_results['u_p_value']:.4e}")

####### FUNCTION FOR SCATTERPLOT
def create_scatter(df, x_var, y_var):
    # Create a copy of the data with selected variables
    df = data[[x_var, y_var]].copy()
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(df[x_var], df[y_var], alpha=0.5)
    
    plt.title(f'Scatter Plot of {x_var} vs {y_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Calculate correlation coefficient and p-value
    corr_coef, p_value = stats.pearsonr(df[x_var], df[y_var])
    print(f"\nPearson Correlation Coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4e}")


################### FUNCTION FOR SCATTERPLOT BY EXT STATUS
def create_grp_scatter(df, x_var, y_var):
    # Create a copy of the data with selected variables
    df = data[['ext_status_bin_num', x_var, y_var]].copy()
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    for status in df['ext_status_bin_num'].unique():
        subset = df[df['ext_status_bin_num'] == status]
        plt.scatter(subset[x_var], subset[y_var], alpha=0.5, label=f'Group {status}')
    
    plt.title(f'Scatter Plot of {x_var} vs {y_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df[[x_var, y_var]].describe())
    
    # Calculate correlation coefficient and p-value
    corr_coef, p_value = stats.pearsonr(df[x_var], df[y_var])
    print(f"\nPearson Correlation Coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4e}")