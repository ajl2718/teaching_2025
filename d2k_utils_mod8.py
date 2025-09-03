import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu, fisher_exact
from typing import List, Dict
from ipywidgets import interact, widgets
from ipywidgets import interactive, widgets

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# Global variable for DataFrame
df = None

def load_data(filepath: str) -> None:
    """
    Load data from CSV file and set it as global DataFrame
    
    Args:
        filepath (str): Path to the CSV file
    """
    global df
    df = pd.read_csv(filepath)
    return df

def perform_statistical_tests(data1, data2):
    """
    Perform t-test and Mann-Whitney U test on two datasets, calculate mean difference and CI
    
    Args:
        data1: First dataset (Successful extubation)
        data2: Second dataset (Failed extubation)
        
    Returns:
        dict: Dictionary containing test statistics, p-values, mean difference, and CI
    """
    # Calculate mean difference
    mean_diff = np.mean(data1) - np.mean(data2)
    
    # Calculate standard error of the difference
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)  # ddof=1 for sample variance
    se_diff = np.sqrt(var1/n1 + var2/n2)
    
    # Calculate 95% CI for the difference
    # Using Welch's t-test degrees of freedom
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    t_crit = stats.t.ppf(0.975, df)  # 97.5th percentile for 95% CI
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # Perform statistical tests
    t_stat, t_p_value = stats.ttest_ind(data1, data2, equal_var=False)
    u_stat, u_p_value = mannwhitneyu(data1, data2, alternative='two-sided')
    
    return {
        't_stat': t_stat,
        't_p_value': t_p_value,
        'u_stat': u_stat,
        'u_p_value': u_p_value,
        'mean_difference': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def freedman_diaconis_bins(data):
    """Calculate optimal bin width using Freedman-Diaconis rule"""
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    if iqr == 0:
        return 1
    bin_width = 2 * iqr * len(data) ** (-1/3)
    return bin_width

def calc_se_ci(data, confidence=0.95):
    """Calculate standard error and confidence interval"""
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
    return mean, se, ci


def format_pvalue(p_value):
    """Format p-value with appropriate notation"""
    if p_value < 0.001:
        return "p < .001"
    else:
        return f"p = {p_value:.3f}"

def print_statistics(group_stats, test_results=None):
    """Print formatted statistics for each group"""
    for group, stats in group_stats.items():
        print(f"\nGroup {group}:")
        print(f"  N = {stats['count']}")
        print(f"  Mean ± SD: {stats['mean']:.2f} ± {stats['std']:.2f}")
        print(f"  Median (IQR): {stats['median']:.2f} ({stats['iqr']:.2f})")
        print(f"  95% CI: ({stats['ci_lower']:.2f}, {stats['ci_upper']:.2f})")
        
    if test_results:
        print("\nStatistical Tests:")
        print("Mean Difference (Successful - Failed):")
        print(f"  {test_results['mean_difference']:.2f} (95% CI: {test_results['ci_lower']:.2f} to {test_results['ci_upper']:.2f})")
        print("\nIndependent t-test:")
        print(f"  t-statistic: {test_results['t_stat']:.3f}")
        print(f"  {format_pvalue(test_results['t_p_value'])}")
        print("\nMann-Whitney U test:")
        print(f"  U-statistic: {test_results['u_stat']:.3f}")
        print(f"  {format_pvalue(test_results['u_p_value'])}")


def remove_outliers(data):
    """
    Remove outliers using IQR method
    
    Args:
        data: numpy array or pandas Series
        
    Returns:
        Data with outliers removed
    """
    if isinstance(data, pd.Series):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
    else:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
    
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    if isinstance(data, pd.Series):
        return data[(data >= lower_bound) & (data <= upper_bound)]
    else:
        return data[(data >= lower_bound) & (data <= upper_bound)]

# Main visualization functions
def create_bar_chart(variable):
    """
    Create a bar chart for a categorical variable
    
    Args:
        variable (str): Name of the categorical variable to plot
    """
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x=variable, hue='fail_ext_status_bin')
    plt.title(f'Distribution of {variable} by Extubation Status')
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Extubation Status', labels=['Successful', 'Failed'])
    plt.tight_layout()
    plt.show()


def create_crosstab(var, target='fail_ext_status_bin'):
    """
    Create cross-tabulation with both row and column totals.
    Shows row percentages within brackets.
    Shows column totals with column percentages within brackets.
    
    Args:
        var: variable name to cross-tabulate
        target: target variable name (default: 'fail_ext_status_bin')
        
    Returns:
        DataFrame: formatted crosstab
        float: odds ratio
        float: p-value from Fisher's exact test
    """
    from scipy.stats import fisher_exact  # Import inside the function
    from IPython.display import display, HTML
    
    # Create the crosstab using the global df
    ct = pd.crosstab(df[var], df[target], dropna=True)
    
    # Ensure columns are in correct order
    if 'Successful extubation' in ct.columns and 'Failed extubation' in ct.columns:
        ct = ct[['Successful extubation', 'Failed extubation']]
    
    # Calculate row percentages
    ct_row_pct = ct.div(ct.sum(axis=1), axis=0).round(4) * 100
    
    # Calculate column percentages
    ct_col_pct = ct.div(ct.sum(axis=0), axis=1).round(4) * 100
    
    # Create combined DataFrame with row percentages
    ct_combined = pd.DataFrame()
    for column in ct.columns:
        ct_combined[column] = ct[column].astype(str) + ' (' + ct_row_pct[column].map("{:.1f}%".format) + ')'
    
    # Add row totals
    row_sums = ct.sum(axis=1)
    ct_combined['Total'] = row_sums.astype(str) + ' (100.0%)'
    
    # Add column totals
    col_sums = ct.sum(axis=0)
    col_pct = (col_sums / col_sums.sum() * 100).round(1)
    total_row = pd.Series({
        col: f"{col_sums[col]} ({col_pct[col]:.1f}%)" 
        for col in ct.columns
    }, name='Total')
    # Add the grand total to the total row
    total_row['Total'] = f"{col_sums.sum()} (100.0%)"
    
    # Combine with the total row
    ct_combined = pd.concat([ct_combined, total_row.to_frame().T])
    
    # Calculate odds ratio and p-value
    if ct.shape == (2, 2):
        # Convert string column names to 0/1 for fisher exact test
        ct_numeric = ct.copy()
        ct_numeric.columns = [0, 1]
        odds_ratio, p_value = fisher_exact(ct_numeric)
    else:
        odds_ratio = None
        _, p_value, _, _ = chi2_contingency(ct)
    
    # Format the table with controlled width
    styled_table = ct_combined.style.set_properties(**{
        'max-width': '100px',
        'width': 'auto',
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('padding', '8px'),
            ('white-space', 'normal'),
            ('max-width', '100px')
        ]},
        {'selector': 'td', 'props': [
            ('text-align', 'center'),
            ('padding', '8px'),
            ('white-space', 'normal'),
            ('max-width', '100px')
        ]},
        {'selector': 'table', 'props': [
            ('width', 'auto'),
            ('margin', '0 auto'),
            ('border-collapse', 'collapse')
        ]},
        # Highlight the totals row with a light gray background
        {'selector': 'tr:last-child', 'props': [
            ('background-color', '#f0f0f0')
        ]}
    ])
    
    return styled_table, odds_ratio, p_value

def create_histo(variable_to_analyze):
    """
    Create histograms and boxplots with descriptive group labels
    
    Args:
        variable_to_analyze (str): Name of the variable to analyze
    """
    # Ensure the variable is numeric
    df[variable_to_analyze] = pd.to_numeric(df[variable_to_analyze], errors='coerce')
    
    # Remove any rows with NaN values
    df_clean = df.dropna(subset=[variable_to_analyze, 'ext_status_bin_num']).copy()
    
    # Group the data
    groups = df_clean['ext_status_bin_num'].unique()
    group_stats = {}
    
    # Create mapping for group labels
    group_labels = {
        0: 'Successful extubation',
        1: 'Failed extubation'
    }
    
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
        
        group_stats[group_labels[group]] = {
            'mean': mean,
            'count': len(data),
            'std': np.std(data, ddof=1),
            'se': se,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'median': median,
            'iqr': iqr
        }
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Boxplot
    boxplot_data = [group_data[group] for group in groups]
    ax1.boxplot(boxplot_data, labels=[group_labels[group] for group in groups])
    ax1.set_title(f'Boxplot of {variable_to_analyze} by Extubation Status')
    ax1.set_xlabel('Extubation Status')
    ax1.set_ylabel(variable_to_analyze)
    
    # Histogram
    for group in groups:
        data = group_data[group]
        ax2.hist(data, bins='auto', alpha=0.5, label=group_labels[group], density=True)
    ax2.set_title(f'Histogram of {variable_to_analyze} by Extubation Status')
    ax2.set_xlabel(variable_to_analyze)
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print_statistics(group_stats, test_results if len(groups) == 2 else None)

def create_scatter(x_var, y_var):
    """
    Create a basic scatter plot
    
    Args:
        x_var (str): Name of the variable for x-axis
        y_var (str): Name of the variable for y-axis
    """
    # Create a copy of the data with selected variables
    plot_df = df[[x_var, y_var]].copy()
    
    # Remove any rows with NaN values
    plot_df = plot_df.dropna()
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    plt.scatter(plot_df[x_var], plot_df[y_var], alpha=0.5)
    
    plt.title(f'Scatter Plot of {x_var} vs {y_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(plot_df.describe())
    
    # Calculate correlation coefficient and p-value
    corr_coef, p_value = stats.pearsonr(plot_df[x_var], plot_df[y_var])
    print(f"\nPearson Correlation Coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4e}")

def create_grp_scatter(x_var, y_var):
    """
    Create a scatter plot with groups
    
    Args:
        x_var (str): Name of the variable for x-axis
        y_var (str): Name of the variable for y-axis
    """
    # Create a copy with selected variables
    plot_df = df[['ext_status_bin_num', x_var, y_var]].copy()
    
    # Remove NaN values
    plot_df = plot_df.dropna()
    
    # Create mapping for group labels
    group_labels = {
        0: 'Successful extubation',
        1: 'Failed extubation'
    }
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    for status in plot_df['ext_status_bin_num'].unique():
        subset = plot_df[plot_df['ext_status_bin_num'] == status]
        plt.scatter(subset[x_var], subset[y_var], alpha=0.5, label=group_labels[status])
    
    plt.title(f'Scatter Plot of {x_var} vs {y_var} by Extubation Status')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    
    # Print statistics
    print("\nSummary Statistics:")
    print(plot_df[[x_var, y_var]].describe())
    
    corr_coef, p_value = stats.pearsonr(plot_df[x_var], plot_df[y_var])
    print(f"\nPearson Correlation Coefficient: {corr_coef:.4f}")
    print(f"P-value: {p_value:.4e}")
