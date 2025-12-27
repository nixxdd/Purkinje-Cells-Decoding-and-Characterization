import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt 
import seaborn as sns
import pandas as pd
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

def create_3d_multi_panel_figure(pc_feature_dfs, MLI_positions, color_by='Mean_Phase', 
                                palette=None, titles=None, heatmap_labels_data=None, 
                                heatmap_labels_display=None, heatmap_colors=None, 
                                add_annotations=True):
    """
    Create a multi-panel figure with 3D plots and colorbar(s)
    
    Parameters:
    - pc_feature_dfs: List of 3 DataFrames with Purkinje cell positions + features
    - MLI_positions: DataFrame with all cell positions  
    - color_by: Feature to color by
    - palette: Color palette dictionary OR list of 3 dictionaries (one per condition)
    - titles: List of subplot titles
    - heatmap_labels_data: Labels data OR list of 3 label arrays
    - heatmap_labels_display: Display labels OR list of 3 display label arrays
    - heatmap_colors: Colors OR list of 3 color arrays
    - add_annotations: Whether to add cell ID annotations
    """
    # Check if we have individual color schemes for each condition
    individual_colormaps = (isinstance(palette, list) and 
                        isinstance(heatmap_labels_data, list) and 
                        isinstance(heatmap_colors, list))
    
    if individual_colormaps:
        # 3 plots + 3 individual colorbars
        fig = plt.figure(figsize=(24, 8), dpi=300)
        gs = GridSpec(1, 6, figure=fig,
                    width_ratios=[2, 0.2, 2, 0.2, 2, 0.2],
                    left=0.05, right=0.98, top=0.85, bottom=0.15, 
                    wspace=0.15)
        
        axes_3d = []
        axes_heatmap = []
        for i in range(3):
            ax_3d = fig.add_subplot(gs[0, i*2], projection='3d')
            ax_heatmap = fig.add_subplot(gs[0, i*2 + 1])
            axes_3d.append(ax_3d)
            axes_heatmap.append(ax_heatmap)
    else:
        # 3 plots + 1 shared colorbar
        fig = plt.figure(figsize=(20, 8), dpi=300)
        gs = GridSpec(1, 4, figure=fig,
                    width_ratios=[2, 2, 2, 0.3],
                    left=0.1, right=0.95, top=0.85, bottom=0.15, 
                    wspace=0.3)
        
        axes_3d = []
        for i in range(3):
            ax = fig.add_subplot(gs[0, i], projection='3d')
            axes_3d.append(ax)
        
        ax_heatmap = fig.add_subplot(gs[0, 3])
    
    x_size, y_size, z_size = 300.0, 200.0, 295.0
    positionsBK = MLI_positions[MLI_positions['cell_type'] == 'basket_cell'][['x', 'y', 'z']].values
    positionsSC = MLI_positions[MLI_positions['cell_type'] == 'stellate_cell'][['x', 'y', 'z']].values
    
    if titles is None:
        titles = ['Baseline', 'Step', 'Burst']
    
    for i, (ax, pc_data, title) in enumerate(zip(axes_3d, pc_feature_dfs, titles)):
        # Purkinje positions
        positionsPC = pc_data[['x', 'y', 'z']].values
        
        # Plot MLI cells (background)
        ax.scatter(positionsSC[:, 0], positionsSC[:, 1], positionsSC[:, 2],
                c='lightgreen', s=15, marker='o', alpha=0.2, 
                label='Stellate cells' if i == 0 else '', edgecolors='darkgreen', linewidth=0.5)
        
        ax.scatter(positionsBK[:, 0], positionsBK[:, 1], positionsBK[:, 2],
                c='lightcoral', s=25, marker='^', alpha=0.2,
                label='Basket cells' if i == 0 else '', edgecolors='darkred', linewidth=0.5)
        
        # Plot Purkinje cells with condition-specific or shared coloring
        if color_by in pc_data.columns and palette is not None:
            feature_values = pc_data[color_by].values
            
            if individual_colormaps:
                # Use condition-specific palette
                current_palette = palette[i]
                colors = [current_palette.get(val, 'gray') for val in feature_values]
            else:
                # Use shared palette
                colors = [palette.get(val, 'gray') for val in feature_values]
                
            scatter = ax.scatter(positionsPC[:, 0], positionsPC[:, 1], positionsPC[:, 2],
                            c=colors, s=100, alpha=0.9, edgecolors='black', 
                            linewidth=0.5, marker='s',
                            label='Purkinje cells' if i == 0 else '')
        else:
            scatter = ax.scatter(positionsPC[:, 0], positionsPC[:, 1], positionsPC[:, 2],
                            c='red', s=100, alpha=0.9, edgecolors='black',
                            linewidth=0.5, marker='s',
                            label='Purkinje cells' if i == 0 else '')
        
        # Add cell ID annotations
        if add_annotations and 'cell_id' in pc_data.columns:
            for idx, row in pc_data.iterrows():
                ax.text(row['x'], row['y'], row['z'],
                    str(row['cell_id']), fontsize=5, color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add layer boundaries
        granular_layer = 130.0
        purkinje_layer = 15.0
        b_molecular_layer = 50.0
        t_molecular_layer = 100.0
        
        z_layers = [
            granular_layer,
            granular_layer + purkinje_layer,
            granular_layer + purkinje_layer + b_molecular_layer,
            granular_layer + purkinje_layer + b_molecular_layer + t_molecular_layer
        ]
        
        xx, yy = np.meshgrid(np.linspace(0, x_size, 8), np.linspace(0, y_size, 8))
        for z_pos in z_layers:
            zz = np.ones_like(xx) * z_pos
            ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')
        
        # Set 3D plot properties
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size) 
        ax.set_zlim(0, z_size)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('X (μm)', fontsize=10)
        ax.set_ylabel('Y (μm)', fontsize=10)
        ax.set_zlabel('Z (μm)', fontsize=10)
        ax.grid(False)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
    
    # Create colorbars
    if individual_colormaps:
        # Individual colorbars for each condition
        for i, ax_heatmap in enumerate(axes_heatmap):
            if (heatmap_labels_data is not None and 
                heatmap_colors is not None and 
                heatmap_labels_display is not None):
                
                sns.heatmap(data=heatmap_labels_data[i], ax=ax_heatmap, 
                        cmap=mcolors.ListedColormap(heatmap_colors[i]), 
                        cbar=False, yticklabels=heatmap_labels_display[i], 
                        xticklabels=False)
                ax_heatmap.set_title(f"{titles[i]}\n{color_by}", fontsize=9)
                ax_heatmap.tick_params(axis='y', labelsize=7)
    else:
        # Shared colorbar
        if heatmap_labels_data is not None and heatmap_colors is not None:
            sns.heatmap(data=heatmap_labels_data, ax=ax_heatmap, 
                    cmap=mcolors.ListedColormap(heatmap_colors), 
                    cbar=False, yticklabels=heatmap_labels_display, 
                    xticklabels=False)
            ax_heatmap.set_title(f"{color_by}\nValues", fontsize=10)
            ax_heatmap.tick_params(axis='y', labelsize=8)
    
    # Set figure title
    fig.suptitle(f"3D Cerebellar Network: Purkinje Cells Colored by {color_by}",
                fontsize=16)
    
    plt.show()

    return fig, axes_3d, axes_heatmap if individual_colormaps else ax_heatmap

def create_3d_ica_comparison(pc_feature_df, MLI_positions, ica1_palette, ica2_palette,
                        ica1_labels_data, ica1_labels_display, ica1_colors,
                        ica2_labels_data, ica2_labels_display, ica2_colors,
                        condition_name='', add_annotations=True,
                        ax_ica1=None, ax_heatmap1=None, ax_ica2=None, ax_heatmap2=None):
    """
    Create a 2-panel figure comparing ICA Component 1 and 2 in 3D - vertically stacked
    
    Additional Parameters:
    - ax_ica1, ax_heatmap1, ax_ica2, ax_heatmap2: Optional axes objects for integration
    """
    # Check if we need to create a new figure
    create_new_figure = any(ax is None for ax in [ax_ica1, ax_heatmap1, ax_ica2, ax_heatmap2])
    
    if create_new_figure:
        # Create figure with 2 3D plots vertically stacked + 2 colorbars
        fig = plt.figure(figsize=(14, 12), dpi=300)
        gs = GridSpec(2, 2, figure=fig,
                    width_ratios=[1, 0.2],
                    height_ratios=[1, 1],
                    left=0.08, right=0.58, top=0.90, bottom=0.07, 
                    wspace=0.01, hspace=0.35)
        
        # Create axes only if not provided
        if ax_ica1 is None:
            ax_ica1 = fig.add_subplot(gs[0, 0], projection='3d')
        if ax_heatmap1 is None:
            ax_heatmap1 = fig.add_subplot(gs[0, 1])
        if ax_ica2 is None:
            ax_ica2 = fig.add_subplot(gs[1, 0], projection='3d')
        if ax_heatmap2 is None:
            ax_heatmap2 = fig.add_subplot(gs[1, 1])

    
    x_size, y_size, z_size = 300.0, 200.0, 295.0
    positionsBK = MLI_positions[MLI_positions['cell_type'] == 'basket_cell'][['x', 'y', 'z']].values
    positionsSC = MLI_positions[MLI_positions['cell_type'] == 'stellate_cell'][['x', 'y', 'z']].values
    positionsPC = pc_feature_df[['x', 'y', 'z']].values
    
    # Plot both ICA components
    axes_3d = [ax_ica1, ax_ica2]
    ica_components = ['ICA_Component_1', 'ICA_Component_2']
    palettes = [ica1_palette, ica2_palette]
    titles = [f'{condition_name} ICA Component 1', f'{condition_name} ICA Component 2']
    
    for i, (ax, component, palette, title) in enumerate(zip(axes_3d, ica_components, palettes, titles)):
        ax.scatter(positionsSC[:, 0], positionsSC[:, 1], positionsSC[:, 2],
                c='lightgreen', s=15, marker='o', alpha=0.2, 
                label='Stellate cells' if i == 0 else '', edgecolors='darkgreen', linewidth=0.5)
        
        ax.scatter(positionsBK[:, 0], positionsBK[:, 1], positionsBK[:, 2],
                c='lightcoral', s=25, marker='^', alpha=0.2,
                label='Basket cells' if i == 0 else '', edgecolors='darkred', linewidth=0.5)
        
        # Plot Purkinje cells colored by ICA component
        if component in pc_feature_df.columns and palette is not None:
            feature_values = pc_feature_df[component].values
            colors = [palette.get(val, 'gray') for val in feature_values]
        else:
            colors = 'red'  # fallback color
            
        scatter = ax.scatter(positionsPC[:, 0], positionsPC[:, 1], positionsPC[:, 2],
                        c=colors, s=100, alpha=0.9, edgecolors='black', 
                        linewidth=0.5, marker='s',
                        label='Purkinje cells' if i == 0 else '')
        
        # Add cell ID annotations
        if add_annotations and 'cell_id' in pc_feature_df.columns:
            for idx, row in pc_feature_df.iterrows():
                ax.text(row['x'], row['y'], row['z'],
                    str(row['cell_id']), fontsize=5, color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add layer boundaries
        granular_layer = 130.0
        purkinje_layer = 15.0
        b_molecular_layer = 50.0
        t_molecular_layer = 100.0
        
        z_layers = [
            granular_layer,
            granular_layer + purkinje_layer,
            granular_layer + purkinje_layer + b_molecular_layer,
            granular_layer + purkinje_layer + b_molecular_layer + t_molecular_layer
        ]
        
        xx, yy = np.meshgrid(np.linspace(0, x_size, 8), np.linspace(0, y_size, 8))
        for z_pos in z_layers:
            zz = np.ones_like(xx) * z_pos
            ax.plot_surface(xx, yy, zz, alpha=0.05, color='gray')
        
        # Set 3D plot properties
        ax.set_xlim(0, x_size)
        ax.set_ylim(0, y_size) 
        ax.set_zlim(0, z_size)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('X (μm)', fontsize=11)
        ax.set_ylabel('Y (μm)', fontsize=11)
        ax.set_zlabel('Z (μm)', fontsize=11)
        ax.grid(False)
        
        # Only show legend on the first plot
        if i == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
    
    # Create colorbars
    heatmap_data = [ica1_labels_data, ica2_labels_data]
    heatmap_displays = [ica1_labels_display, ica2_labels_display]
    heatmap_colors_list = [ica1_colors, ica2_colors]
    heatmap_axes = [ax_heatmap1, ax_heatmap2]
    
    for i, (ax_heatmap, data, display, colors) in enumerate(zip(heatmap_axes, heatmap_data, heatmap_displays, heatmap_colors_list)):
        sns.heatmap(data=data, ax=ax_heatmap, 
                cmap=mcolors.ListedColormap(colors), 
                cbar=False, yticklabels=display, 
                xticklabels=False)
        ax_heatmap.set_title(f"ICA {i+1}\nValues", fontsize=10)
        ax_heatmap.tick_params(axis='y', labelsize=8)
    
    if create_new_figure:
        fig.suptitle(f"3D Cerebellar Network:\nICA Component Comparison",
                    fontsize=16, y=1.0, x=0.34)
        plt.show()
    

def create_single_correlation_matrix(data, condition_name, variables=None, variable_labels=None, figsize=(8, 8), show_triangle='upper'):
    """
    Create a single Spearman correlation matrix with p-values for any given dataset
    
    Parameters:
    - data: DataFrame with the variables to correlate
    - condition_name: String name for the condition (used in title)
    - variables: List of variable names to correlate (default: predefined set)
    - variable_labels: List of display labels for variables (default: predefined set)
    - figsize: Tuple for figure size
    - show_triangle: 'upper', 'lower', or 'both' to control which triangle to display
    """
    if variables is None:
        variables = ['Mean_Firing_Rate', 'Aperiodic_Exponent', 'Mean_Phase', 'Total_Energy', 'pc_MLI_correlations']
    
    if variable_labels is None:
        variable_labels = ['Firing Rate', 'Aperiodic Exponent', 'Mean Phase', 'Total Energy', 'PC-MLI Correlation']
    
    n_vars = len(variables)
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))
    
    # Calculate pairwise Spearman correlations
    for j in range(n_vars):
        for k in range(n_vars):
            if j == k:
                corr_matrix[j, k] = 1.0
                pval_matrix[j, k] = 0.0
            else:
                var1_data = data[variables[j]].values
                var2_data = data[variables[k]].values
                
                mask = ~(np.isnan(var1_data) | np.isnan(var2_data))
                var1_clean = var1_data[mask]
                var2_clean = var2_data[mask]
                
                if len(var1_clean) > 1:
                    corr, pval = spearmanr(var1_clean, var2_clean)
                    corr_matrix[j, k] = corr
                    pval_matrix[j, k] = pval
                else:
                    corr_matrix[j, k] = np.nan
                    pval_matrix[j, k] = np.nan
    
    if show_triangle == 'upper':
        mask_matrix = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
        corr_matrix = np.ma.masked_where(mask_matrix, corr_matrix)
        pval_matrix = np.ma.masked_where(mask_matrix, pval_matrix)
    elif show_triangle == 'lower':
        mask_matrix = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_matrix = np.ma.masked_where(mask_matrix, corr_matrix)
        pval_matrix = np.ma.masked_where(mask_matrix, pval_matrix)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # Set ticks and remove grid lines
    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))
    ax.grid(False)
    
    # Add correlation values and p-values as text
    for j in range(len(variables)):
        for k in range(len(variables)):
            if j == k:
                continue
            if show_triangle == 'upper' and j > k:
                continue
            elif show_triangle == 'lower' and j < k:
                continue
                
            if not np.isnan(corr_matrix.data[j, k] if hasattr(corr_matrix, 'data') else corr_matrix[j, k]):
                # Correlation coefficient
                corr_val = corr_matrix.data[j, k] if hasattr(corr_matrix, 'data') else corr_matrix[j, k]
                pval_val = pval_matrix.data[j, k] if hasattr(pval_matrix, 'data') else pval_matrix[j, k]
                
                corr_text = f'{corr_val:.3f}'
                
                # P-value formatting
                if j != k:  # Don't show p-value for diagonal
                    if pval_val < 0.001:
                        pval_text = 'p<0.001'
                    elif pval_val < 0.01:
                        pval_text = f'p<0.01'
                    elif pval_val < 0.05:
                        pval_text = f'p<0.05'
                    else:
                        pval_text = f'p={pval_val:.3f}'
                    
                    combined_text = f'{corr_text}\n{pval_text}'
                else:
                    combined_text = corr_text
                
                # Text color based on correlation strength
                text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                
                # Add significance styling
                if j != k and pval_val < 0.05:
                    fontweight = 'bold'
                else:
                    fontweight = 'normal'
                
                ax.text(k, j, combined_text, 
                    ha='center', va='center', color=text_color, 
                    fontsize=10, fontweight=fontweight)
    
    ax.set_xticklabels(variable_labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(variable_labels, fontsize=12)
    ax.set_title(f'{condition_name}\nSpearman Correlations', fontsize=14)
    
    # Ensure square aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Remove tick marks
    ax.tick_params(which='both', length=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation Coefficient', fontsize=12, rotation=90, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of significant correlations
    print(f"Summary of Significant Correlations for {condition_name} (p < 0.05):")
    print("=" * 60)
    
    significant_pairs = []
    for j in range(len(variables)):
        for k in range(j+1, len(variables)):  # Only upper triangle for summary
            corr_val = corr_matrix.data[j, k] if hasattr(corr_matrix, 'data') else corr_matrix[j, k]
            pval_val = pval_matrix.data[j, k] if hasattr(pval_matrix, 'data') else pval_matrix[j, k]
            
            if not np.isnan(pval_val) and pval_val < 0.05:
                significant_pairs.append((
                    variable_labels[j], 
                    variable_labels[k], 
                    corr_val, 
                    pval_val
                ))
    
    if significant_pairs:
        for var1, var2, corr, pval in significant_pairs:
            significance = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
            print(f"  {var1} ↔ {var2}: r = {corr:.3f}, p = {pval:.4f} {significance}")
    else:
        print("  No significant correlations found")
    
    return corr_matrix, pval_matrix, fig, ax