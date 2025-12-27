from turtle import pd
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def relate_MLI_to_pc(MLI_rates, purkinje_rates, clusters=None):
    """
    Relate MLI Cells activity to Purkinje Cells activity based on clusters.

    Args:
        MLI_rates: numpy array of MLI Cells rates per neuron
        purkinje_spikes: numpy array of Purkinje Cells rates per neuron
        clusters: cluster labels for each Purkinje neuron

    Returns:
        corr_matrix: correlation matrix between MLI and Purkinje neurons
        neurons_boundary: index separating ON and OFF neurons (for plotting)
        neuronsOFF: indices of OFF neurons
        neuronsON: indices of ON neurons
    """
    num_neurons_MLI = MLI_rates.shape[0]
    centered_ratesPC = purkinje_rates - np.mean(purkinje_rates, axis=1, keepdims=True)
    standardized_ratesPC = centered_ratesPC / np.std(centered_ratesPC, axis=1, keepdims=True)
    centered_ratesMLI = MLI_rates - np.mean(MLI_rates, axis=1, keepdims=True)
    standardized_ratesMLI = centered_ratesMLI / np.std(centered_ratesMLI, axis=1, keepdims=True)

    if clusters is not None:
        neuronsOFF = np.where(clusters == 0)[0]
        neuronsON  = np.where(clusters == 1)[0]
        rates_neuron_ON = standardized_ratesPC[neuronsON]
        rates_neuron_OFF = standardized_ratesPC[neuronsOFF]
        corr_matrixON = np.zeros((num_neurons_MLI, neuronsON.shape[0]))
        corr_matrixOFF = np.zeros((num_neurons_MLI, neuronsOFF.shape[0]))

        for i in range(num_neurons_MLI):
            for j in range(neuronsON.shape[0]):
                corr_matrixON[i, j], _ = spearmanr(standardized_ratesMLI[i], rates_neuron_ON[j])
            for j in range(neuronsOFF.shape[0]):
                corr_matrixOFF[i, j], _ = spearmanr(standardized_ratesMLI[i], rates_neuron_OFF[j])

        print(f"Shape of correlation matrix ON: {corr_matrixON.shape}")
        print(f"Shape of correlation matrix OFF: {corr_matrixOFF.shape}")

        corr_matrix = np.concatenate((corr_matrixON, corr_matrixOFF), axis=1)
        neurons_boundary = neuronsON.shape[0]
        print(f"Shape of combined correlation matrix: {corr_matrix.shape}")
    else:
        corr_matrix = np.zeros((num_neurons_MLI, purkinje_rates.shape[0]))
        for i in range(num_neurons_MLI):
            for j in range(purkinje_rates.shape[0]):
                corr_matrix[i, j], _ = spearmanr(standardized_ratesMLI[i], standardized_ratesPC[j])
        neurons_boundary = purkinje_rates.shape[0]
        neuronsOFF = np.array([neuron for neuron in range(purkinje_rates.shape[0])])
        neuronsON = np.array([])

    return corr_matrix, neurons_boundary, neuronsOFF, neuronsON

def compute_pvalue(corr_matrix, dof):
    basket_neurons = corr_matrix.shape[0]
    purkinje_neurons = corr_matrix.shape[1]

    p_values = np.zeros((basket_neurons, purkinje_neurons))
    for i in range(basket_neurons):
        for j in range(purkinje_neurons):
            r = corr_matrix[i, j]
            t_stat = r * np.sqrt((dof - 2) / (1 - r**2))
            p_value = 2 * (1 - scipy.stats.t.cdf(np.abs(t_stat), df=dof - 2))
            p_values[i, j] = p_value
    
    return p_values

def plot_pvalue_corr(corr, p_values, neuronsOFF, neuronsON, title, neurons_boundary, alpha=0.05):
    """
    Plot the correlation matrix with p-values, with significance markers.
    Args:
        corr: correlation matrix
        p_values: p-value matrix for the correlations
        neuronsOFF: indices of OFF neurons
        neuronsON: indices of ON neurons
        title: title for the plot
        neurons_boundary: index separating ON and OFF neurons
        alpha: significance level for p-values
    """
    x_labels = neuronsON.tolist() + neuronsOFF.tolist()
    print(f"X labels: {x_labels}")

    fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
    ax.axvline(x=neurons_boundary, color='k', linestyle='--', alpha=0.5, linewidth=2)
    ax.set_xticks(ticks=[label for label in range(len(x_labels))], labels=x_labels)

    min_val = np.min(p_values)
    max_val = np.max(p_values)
    max_corr = np.max(corr)
    min_corr = np.min(corr)
    print(f"Min p-value: {min_val}, Max p-value: {max_val}")
    
    sns.heatmap(corr, cmap='bwr', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, 
                cbar=True, vmin=min_corr, vmax=max_corr, center=0, ax=ax)
    
    ax.set_title(title, fontsize=8, pad=20)
    ax.set_xlabel('Purkinje Neurons', fontsize=8)
    ax.set_ylabel('MLI Neurons', fontsize=8)
    sig_mask = p_values < alpha
    print(f"Significant p-values (alpha={alpha}): {np.sum(sig_mask)} out of {p_values.size}")

    # significance markers for significant correlations
    for i in range(p_values.shape[0]):
        for j in range(p_values.shape[1]):
            p_value = p_values[i, j]
            if p_value < alpha and not np.isnan(p_value):
                correlation_value = corr[i, j]
                text_color = 'white' if abs(correlation_value) > 0.5 else 'black'
                if p_value <= 0.001:
                    marker = '***'
                elif p_value <= 0.01:
                    marker = '**'
                elif p_value <= 0.05:
                    marker = '*'
                else:
                    continue
                ax.text(j + 0.5, i + 0.5, marker,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=2,
                        color=text_color,
                        weight='bold')
    ax.grid(False)
    plt.show()

def MLI_pc_exponent_analysis(foof_features_df, MLI_rates, purkinje_rates, clusters=None):
    """
    PC-MLI Cells correlation analysis correlating it also to the aperiodic exponent.
    Args:
        foof_features_df: DataFrame containing aperiodic exponent for each Purkinje neuron
        MLI_rates: numpy array of MLI Cells rates per neuron
        purkinje_rates: numpy array of Purkinje Cells rates per neuron
        clusters: cluster labels for each Purkinje neuron (optional)
    Returns:
        results: dictionary containing correlation results
    """
    exponents = foof_features_df['Aperiodic_Exponent'].values
    
    # Population-level correlation
    pop_MLI_activity = np.mean(MLI_rates, axis=0)
    pop_pc_activity = np.mean(purkinje_rates, axis=0)

    print(f"Shape of population MLI activity: {pop_MLI_activity.shape}") # (900, )
    print(f"Shape of population PC activity: {pop_pc_activity.shape}") # (900, )
    
    # Correlate population activities
    pop_correlation, pval_pop = spearmanr(pop_MLI_activity, pop_pc_activity)

    # Individual neuron-level analysis
    mean_pc_rates = np.mean(purkinje_rates, axis=1)  # Mean firing rate per PC neuron
    print(f"Shape of mean PC rates: {mean_pc_rates.shape}")  # (68, )
    # Correlate PC mean rates with their aperiodic exponents
    pc_rate_exp_corr, pval_pc_exp = spearmanr(mean_pc_rates, exponents)
    
    # For each PC, correlate its activity with mean basket activity
    pc_MLI_correlations = np.zeros(len(mean_pc_rates))
    for i in range(len(mean_pc_rates)):
        pc_MLI_correlations[i], _ = spearmanr(purkinje_rates[i], pop_MLI_activity)

    # Correlate these PC-MLI correlations with aperiodic exponents
    corr_strength_exp_relation, pval_exp = spearmanr(pc_MLI_correlations, exponents)

    print(f"Population MLI-PC correlation: {pop_correlation:.3f} and p-value: {pval_pop:.4f}")
    print(f"PC firing rate vs aperiodic exponent correlation: {pc_rate_exp_corr:.3f} and p-value: {pval_pc_exp:.4f}")
    print(f"PC-MLI correlation strength vs aperiodic exponent: {corr_strength_exp_relation:.3f}, p:{pval_exp:.4f}")

    results = {
        'pop_correlation': pop_correlation,
        'pc_rate_exp_corr': pc_rate_exp_corr,
        'corr_strength_exp_relation': corr_strength_exp_relation,
        'pc_MLI_correlations': pc_MLI_correlations,
        'exponents': exponents
    }

    if clusters is not None:
        neuronsOFF = np.where(clusters == 0)[0]
        neuronsON = np.where(clusters == 1)[0]
    
        exp_ON = exponents[neuronsON]
        exp_OFF = exponents[neuronsOFF]
        corr_ON = pc_MLI_correlations[neuronsON]
        corr_OFF = pc_MLI_correlations[neuronsOFF]

        corr_exp_ON, pval_expON = spearmanr(corr_ON, exp_ON) if len(neuronsON) > 1 else (np.nan, np.nan)
        corr_exp_OFF, pval_expOFF = spearmanr(corr_OFF, exp_OFF) if len(neuronsOFF) > 1 else (np.nan, np.nan)

        print(f"ON neurons - PC-MLI corr vs aperiodic exp: {corr_exp_ON:.3f}, p:{pval_expON:.4f}")
        print(f"OFF neurons - PC-MLI corr vs aperiodic exp: {corr_exp_OFF:.3f}, p:{pval_expOFF:.4f}")

    results = pd.DataFrame(results)
    return results

def plot_pc_MLI_vs_exponent(results, clusters_pc=None, regression=False, ax=None):
    """
    Plot the relationship between PC-MLI correlations and aperiodic exponents
    """
    pc_MLI_correlations = results['pc_MLI_correlations']
    exponents = results['exponents']

    if ax is not None:
        axes = ax
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Scatter plot of all neurons
    axes[0].scatter(exponents, pc_MLI_correlations, alpha=0.5, s=15, color='purple')

    if regression:
        # correlation line and stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(exponents, pc_MLI_correlations)
        line = slope * exponents + intercept
        axes[0].plot(exponents, line, 'r--', alpha=0.8, linewidth=2)
    else:
        r_value, p_value = spearmanr(exponents, pc_MLI_correlations) if len(exponents) > 1 else (np.nan, np.nan)

    axes[0].set_xlabel('Aperiodic Exponent')
    axes[0].set_ylabel('PC-Basket Correlation Strength')
    axes[0].set_title(f'PC-Basket Correlation vs Aperiodic Exponent')
    axes[0].grid(True, alpha=0.3)
    
    # Add text with correlation info
    axes[0].text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.4f}\n', 
                transform=axes[0].transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Separate by ON/OFF clusters if available
    if clusters_pc is not None:
        neuronsOFF = np.where(clusters_pc == 0)[0]
        neuronsON = np.where(clusters_pc == 1)[0]
        
        # ON neurons
        axes[1].scatter(exponents[neuronsON], pc_MLI_correlations[neuronsON], 
                    alpha=0.5, s=15, color='red', label=f'ON neurons (n={len(neuronsON)})')
        
        # OFF neurons  
        axes[1].scatter(exponents[neuronsOFF], pc_MLI_correlations[neuronsOFF], 
                    alpha=0.5, s=15, color='blue', label=f'OFF neurons (n={len(neuronsOFF)})')
        if regression:
            # separate regression lines
            if len(neuronsON) > 1:
                slope_on, intercept_on, r_on, p_on, _ = stats.linregress(
                    exponents[neuronsON], pc_MLI_correlations[neuronsON])
                line_on = slope_on * exponents[neuronsON] + intercept_on
                axes[1].plot(exponents[neuronsON], line_on, 'r--', alpha=0.8, linewidth=2)
            
            if len(neuronsOFF) > 1:
                slope_off, intercept_off, r_off, p_off, _ = stats.linregress(
                    exponents[neuronsOFF], pc_MLI_correlations[neuronsOFF])
                line_off = slope_off * exponents[neuronsOFF] + intercept_off
                axes[1].plot(exponents[neuronsOFF], line_off, 'b--', alpha=0.8, linewidth=2)
        else:
            r_on, p_on = spearmanr(exponents[neuronsON], pc_MLI_correlations[neuronsON]) if len(neuronsON) > 1 else (np.nan, np.nan)
            r_off, p_off = spearmanr(exponents[neuronsOFF], pc_MLI_correlations[neuronsOFF]) if len(neuronsOFF) > 1 else (np.nan, np.nan)
        
        axes[1].set_xlabel('Aperiodic Exponent')
        axes[1].set_ylabel('PC-MLI Correlation Strength')
        axes[1].set_title('PC-MLI Correlation vs Aperiodic Exponent by ON/OFF State')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        # Add correlation stats for each group
        if len(neuronsON) > 1 and len(neuronsOFF) > 1:
            stats_text = f'ON: r={r_on:.3f}, p={p_on:.4f}\nOFF: r={r_off:.3f}, p={p_off:.4f}'
            axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axes[1].set_visible(False)
    
    print(f"Overall correlation between PC-basket strength and aperiodic exponent: {r_value:.3f} (p={p_value:.4f})")


def mutual_info(rates_per_neuronPC, rates_per_neuronBK, regression=False):
    """
    Compute Mutual Information between Purkinje Cells and Basket Cells activity.
    Args:
        rates_per_neuronPC: numpy array of Purkinje Cells rates per neuron
        rates_per_neuronBK: numpy array of Basket Cells rates per neuron
        regression: whether to perform regression analysis
    Returns:
        est_MI: estimated Mutual Information
    """
    centrered_ratesPC = rates_per_neuronPC - np.mean(rates_per_neuronPC, axis=1, keepdims=True)
    print(f"Shape of centered ratesPC: {centrered_ratesPC.shape}")

    # aggregate BK activity for each time bin
    mean_BK = np.zeros((rates_per_neuronBK.shape[1],))
    for i in range(rates_per_neuronBK.shape[1]):
        mean_BK[i] = np.mean(rates_per_neuronBK[:, i])

    print(f"Shape of mean_BK: {mean_BK.shape}") # debugging

    est_MI = mutual_info_regression(centrered_ratesPC.T, mean_BK, discrete_features=True)
    print(f"Estimated Mutual Information: {est_MI}")

    return est_MI

def plot_pc_basketMI_vs_exponent(est_MI, foof_features_df, clusters_pc=None, regression=False):
    """
    Plot the relationship between PC-basket Mutual Information and aperiodic exponents
    """
    pc_basket_MI = est_MI
    exponents = foof_features_df['Aperiodic_Exponent'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Scatter plot of all neurons
    axes[0].scatter(exponents, pc_basket_MI, alpha=0.7, s=15, color='orange')
    
    if regression:
        slope, intercept, r_value, p_value, std_err = stats.linregress(exponents, pc_basket_MI)
        line = slope * exponents + intercept
        axes[0].plot(exponents, line, 'r--', alpha=0.8, linewidth=2)
    else:
        r_value, p_value = spearmanr(exponents, pc_basket_MI)

    axes[0].set_xlabel('Aperiodic Exponent')
    axes[0].set_ylabel('PC-Basket Mutual Information')
    axes[0].set_title(f'PC-Basket Mutual Information vs Aperiodic Exponent\nr = {r_value:.3f}, p = {p_value:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    axes[0].text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.4f}\n', 
                transform=axes[0].transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if clusters_pc is not None:
        neuronsOFF = np.where(clusters_pc == 0)[0]
        neuronsON = np.where(clusters_pc == 1)[0]
        
        # ON neurons
        axes[1].scatter(exponents[neuronsON], pc_basket_MI[neuronsON], 
                       alpha=0.7, s=15, color='red', label=f'ON neurons (n={len(neuronsON)})')
        
        # OFF neurons  
        axes[1].scatter(exponents[neuronsOFF], pc_basket_MI[neuronsOFF], 
                       alpha=0.7, s=15, color='blue', label=f'OFF neurons (n={len(neuronsOFF)})')
        
        if regression: 
            if len(neuronsON) > 1:
                slope_on, intercept_on, r_on, p_on, _ = stats.linregress(
                    exponents[neuronsON], pc_basket_MI[neuronsON])
                line_on = slope_on * exponents[neuronsON] + intercept_on
                axes[1].plot(exponents[neuronsON], line_on, 'r--', alpha=0.8, linewidth=2)
            
            if len(neuronsOFF) > 1:
                slope_off, intercept_off, r_off, p_off, _ = stats.linregress(
                    exponents[neuronsOFF], pc_basket_MI[neuronsOFF])
                line_off = slope_off * exponents[neuronsOFF] + intercept_off
                axes[1].plot(exponents[neuronsOFF], line_off, 'b--', alpha=0.8, linewidth=2)
        else:
            r_on, p_on = spearmanr(exponents[neuronsON], pc_basket_MI[neuronsON]) if len(neuronsON) > 1 else (np.nan, np.nan)
            r_off, p_off = spearmanr(exponents[neuronsOFF], pc_basket_MI[neuronsOFF]) if len(neuronsOFF) > 1 else (np.nan, np.nan)
        
        axes[1].set_xlabel('Aperiodic Exponent')
        axes[1].set_ylabel('PC-Basket Mutual Information')
        axes[1].set_title('PC-Basket Mutual Information vs Aperiodic Exponent by ON/OFF State')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        if len(neuronsON) > 1 and len(neuronsOFF) > 1:
            stats_text = f'ON: r={r_on:.3f}, p={p_on:.4f}\nOFF: r={r_off:.3f}, p={p_off:.4f}'
            axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    else:
        axes[1].set_visible(False)
    
    plt.tight_layout()
    plt.show()

    print(f"Overall correlation between PC-basket MI and aperiodic exponent: {r_value:.3f} (p={p_value:.4f})")






