import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import pandas as pd
from scipy.stats import ttest_ind, spearmanr

def heatmap_firing_rates(rates_per_neuron):
    """
    Create a heatmap of frequency counts for each neuron across frequency bins.
    Args:
        rates_per_neuron: (n_neurons, n_timepoints) - firing rates for each neuron
    Returns:
        counts_per_neuron: DataFrame with frequency counts for each neuron
    """
    freqs = np.linspace(1, 200, 200) # Frequency bins from 1Hz to 200Hz
    counts_per_neuron = np.zeros((rates_per_neuron.shape[0], len(freqs) - 1)) 
    for idx, neuron_rates in enumerate(rates_per_neuron):
        counts, _ = np.histogram(neuron_rates, bins=freqs)
        counts_per_neuron[idx] = counts

    counts_per_neuron = pd.DataFrame(counts_per_neuron, columns=freqs[:-1])
    
    return counts_per_neuron

def plot_count_heatmap(heatmap_data, title='Firing Rate Heatmap', ax=None):
    """
    Plot a heatmap of firing rates for each neuron across frequency bins.
    Args:
        heatmap_data: DataFrame with frequency counts for each neuron
        title: title for the heatmap plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    
    sns.heatmap(heatmap_data.T[:100], ax=ax, cmap='plasma', cbar_kws={'label': 'Counts Frequencies'})
    ax.set_title(title)

    tick_positions = np.arange(0, 100, 5)  
    tick_labels = np.arange(1, 101, 5)      
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel('Neurons')
    ax.set_ylabel('Frequency (Hz)')

    if ax is None:
        plt.tight_layout()
        plt.show()

def PCA_heatmap(heatmap, n_components=None):
    """
    Perform PCA on the heatmap data.
    Args:
        heatmap: (n_neurons, n_freqs) - frequency counts heatmap
        n_components: number of PCA components to keep (default is min(n_neurons, n _freqs))
    Returns:
        pca_result: PCA transformed data
        pca: fitted PCA object
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(heatmap)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_result, pca

def plot_pca_analysis(heatmap, pca_obj, ax=None):
    """
    PCA plotting: components and neuron contributions
    Args:
        pca_obj: fitted PCA object
        title: plot title
    """
    n_neurons, n_freqs = heatmap.shape
    
    # PCA scores (neuron contributions to components)
    pca_scores = pca_obj.transform(heatmap)
    
    components = pca_obj.components_
    
    if ax is None:
        fig, axes = plt.subplots(1, 1, figsize=(18, 5), dpi=300)
    else:
        axes = ax

    # Plot 1: PC1 vs PC2
    freq_axis = np.arange(n_freqs)
    axes.plot(freq_axis, components[0, :], 'b-', linewidth=2, label='PC1', alpha=0.8)
    axes.plot(freq_axis, components[1, :], 'r-', linewidth=2, label='PC2', alpha=0.8)
    axes.set_xlabel('Frequency Bins')
    axes.set_ylabel('Component Weight')
    axes.set_title('PCA Components (Temporal Patterns)')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    # some stats
    print(f"PCA Summary:")
    print(f"PC1 explains {pca_obj.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PC2 explains {pca_obj.explained_variance_ratio_[1]:.1%} of variance")
    
    return pca_scores

def kmeans_cluster(heatmap, n_clusters=2):
    """
    Perform k-means clustering on the heatmap data.
    Args:
        heatmap: (n_neurons, n_freqs) - firing rates heatmap
        n_clusters: number of clusters to form
    Returns:
        labels: cluster labels for each neuron
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(heatmap)
    
    # Print clustering metrics
    print(f"K-Means Clustering: {n_clusters} clusters")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(heatmap, labels):.3f}")
    
    return labels

def hierarchical_cluster(heatmap, n_clusters=2):
    """
    Perform hierarchical clustering on the heatmap data.
    Args:
        heatmap: (n_neurons, n_freqs) - firing rates heatmap
        n_clusters: number of clusters to form
    Returns:
        labels: cluster labels for each neuron
    """

    Z = linkage(heatmap, method='complete', metric='cosine')
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')
    fig = plt.figure(figsize=(10, 5), dpi=300)
    dn = dendrogram(Z)
    plt.show()

    # Print clustering metrics
    print(f"Hierarchical Clustering: {n_clusters} clusters")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(heatmap, labels):.3f}")
    
    return labels

def plot_fr_clusters(rates_per_neuron, clusters, cell_ids=None, ax=None):
    """
    Plot firing rate clusters and perform statistical analysis.
    Args:
        rates_per_neuron: (n_neurons, n_timepoints) - firing rates for each neuron
        clusters: cluster labels for each neuron
        cell_ids: actual cell IDs for each neuron (optional, defaults to 1,2,3...)
    Returns:
        neuronsON: indices of neurons in the ON cluster
        neuronsOFF: indices of neurons in the OFF cluster
    """
    neuronsOFF = np.where(clusters == 0)[0]
    neuronsON  = np.where(clusters == 1)[0]

    fr_mean = rates_per_neuron.mean(axis=1)
    fr_mean_OFF = fr_mean[neuronsOFF]
    fr_mean_ON = fr_mean[neuronsON]

    t_stat, pval = ttest_ind(fr_mean_ON, fr_mean_OFF)

    if cell_ids is None:
        # Create sequential cell IDs starting from 1
        cell_ids = np.arange(1, len(clusters) + 1)
    
    all_data = pd.DataFrame({
        'Cell_ID': cell_ids,
        'Frequency': fr_mean,
        'Label': ['ON' if clusters[i] == 1 else 'OFF' for i in range(len(clusters))],
        'Original_Index': range(len(clusters))
    })
    
    all_data_sorted = all_data.sort_values('Cell_ID').reset_index(drop=True)
    
    on_data = all_data.query("Label == 'ON'")
    off_data = all_data.query("Label == 'OFF'")
    
    print("First 5 neurons (sorted by Cell ID):")
    print(all_data_sorted.head())
    
    if ax is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6), dpi=300)
    else:
        axs = ax
    colors = ['red', 'blue']
    
    # First plot: neurons ordered by Cell ID
    on_mask = all_data_sorted['Label'] == 'ON'
    off_mask = all_data_sorted['Label'] == 'OFF'
    
    # Plot all neurons in Cell ID order, colored by cluster
    axs[0].scatter(range(len(all_data_sorted)), all_data_sorted['Frequency'], 
                   c=['red' if label == 'ON' else 'blue' for label in all_data_sorted['Label']], 
                   alpha=0.7, s=15)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label=f'ON Neurons (n={len(on_data)})'),
                      Patch(facecolor='blue', label=f'OFF Neurons (n={len(off_data)})')]
    axs[0].legend(handles=legend_elements, title='Cluster', loc='upper right')
    
    tick_positions = range(len(all_data_sorted))
    axs[0].set_xticks(tick_positions[::2])  
    axs[0].set_xticklabels([all_data_sorted.iloc[i]['Cell_ID'] for i in range(0, len(all_data_sorted), 2)], 
                           rotation=90, ha='right')
    
    axs[0].set_title('Firing Rate Clusters by Cell ID')
    axs[0].set_xlabel('Cell ID')
    axs[0].set_ylabel('Firing Rate (Hz)')
    axs[0].grid(True, alpha=0.3)
    
    # Box plot 
    data_for_stats = pd.DataFrame({
        'Frequency': np.concatenate([on_data['Frequency'], off_data['Frequency']]),
        'Label': ['ON'] * len(on_data) + ['OFF'] * len(off_data)
    })
    
    sns.boxplot(data=data_for_stats, x='Label', y='Frequency', hue='Label', palette=colors, ax=axs[1])
    axs[1].set_title(f'Firing Rate Distribution by Cluster\np-value: {pval:.2e}')
    axs[1].set_xlabel('Cluster')
    axs[1].set_ylabel('Firing Rate (Hz)')

    if ax is None:
        plt.tight_layout()
        plt.show()

    return neuronsON, neuronsOFF

def corr_pvalue(feature1, feature2):
    """
    Calculate the p-value for the correlation between two features.
    
    Args:
        feature1: first feature array
        feature2: second feature array
    Returns:
        p_value: p-value for the correlation
        r: correlation coefficient
    """
    r, p_value = spearmanr(feature1, feature2)
    return p_value, r