from neo import io
import neo
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt 
import seaborn as sns
import plotly.express as px
import pandas as pd
from pathlib import Path
from scipy.signal import convolve
from scipy.signal.windows import triang
from sklearn.decomposition import PCA
from scipy import stats


def compute_correlation_matrix(rates_array):
    """
    Standardize data (zero mean, unit std) per neuron, then compute covariance matrix between time bins.

    Args:
        rates_array: numpy array of shape (n_neurons, n_time_bins)

    Returns:
        cov_matrix: covariance matrix of shape (n_time_bins, n_time_bins)
    """
    # Center the data by subtracting mean firing rate per neuron (across time bins)
    #centered_rates = rates_array - np.mean(rates_array, axis=1, keepdims=True)  # axis=1 for time bins
    z_scored_rates = (rates_array - np.mean(rates_array, axis=1, keepdims=True)) / np.std(rates_array, axis=1, keepdims=True)

    # Number of neurons
    n_neurons = z_scored_rates.shape[0]

    # Compute covariance matrix between time bins: (n_time_bins x n_time_bins)
    # This is done by multiplying the transpose of centered_rates by centered_rates,
    # then normalizing by number of neurons
    # cov_matrix = np.dot(centered_rates.T, centered_rates) / n_neurons
    correlation_matrix = np.corrcoef(z_scored_rates.T)


    return correlation_matrix



