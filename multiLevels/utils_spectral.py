from scipy.signal import welch, hilbert
import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF
import pandas as pd
import scipy
from scipy.stats import circmean
import seaborn as sns

def compute_power_spectrum(rates_array, dt=0.1, nperseg=1024):
    """
    Compute power spectrum using Welch's method.
    
    Args:
        rates_array: 2D numpy array of shape (n_neurons, n_time_bins)
        dt: time step in ms (from your firing rate computation)
        nperseg: length of each segment for Welch's method
    Returns:
        freqs: frequencies corresponding to the power spectrum (in Hz)
        psd: power spectral density values
    """
    dt_s = dt / 1000.0
    fs =  1 / dt_s  
    print(fs)
    
    psd = []
    
    for neuron_rates in rates_array:
        f, p = welch(neuron_rates, fs=fs, nperseg=nperseg, detrend='constant')
        psd.append(p)

    freqs = np.array(f)
    freqs = freqs[1:]
    psd = np.array(psd)[:, 1:]
    print(f"Computed power spectrum for {rates_array.shape[0]} neurons with {len(freqs)} frequency bins.")
    
    return freqs, psd

def plot_power_spectrum(freqs, psd, pop, n_neurons_to_plot=8, log_plot=False):
    """
    Plot power spectrum for a given population.
    
    Args:
        freqs: frequencies corresponding to the power spectrum
        psd: power spectral density values (shape: n_neurons x n_freqs)
        pop: name of the population
        n_neurons_to_plot: number of neurons to plot
        log_plot: if True, shows linear, log-y, and log-log plots
    """
    # neurons with peaks in the range 50-100Hz
    range_mask = (freqs >= 50) & (freqs <= 100)
    neurons_max_in_range = np.argsort(np.sum(psd[:, range_mask], axis=1))[-n_neurons_to_plot:]
    colors_neurons = dict(zip(neurons_max_in_range, sns.color_palette('hsv', n_colors=len(neurons_max_in_range))))
    colors = [colors_neurons[i] for i in neurons_max_in_range]
    
    if log_plot:
        fig, axs = plt.subplots(1, 3, figsize=(20, 5), dpi=300)
        plot_titles = [
            f'Linear Power Spectrum for {pop}',
            f'Log-Y Power Spectrum for {pop}', 
            f'Log-Log Power Spectrum for {pop}'
        ]
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 5), dpi=300)
        axs = [axs]
        plot_titles = [f'Power Spectrum for {pop}']
    
    for i in range(n_neurons_to_plot):
        neuron_id = neurons_max_in_range[i]
        color = colors[i]
        label = f'Neuron {neuron_id}'
        
        if log_plot:
            axs[0].plot(freqs, psd[neuron_id], label=label, alpha=0.7, 
                       linewidth=0.5, color=color)
    
            axs[1].plot(freqs, psd[neuron_id], label=label, alpha=0.5, 
                       linewidth=0.5, color=color)
            
            axs[2].plot(freqs, psd[neuron_id], label=label, alpha=0.5, 
                       linewidth=0.5, color=color)
        else:
            axs[0].plot(freqs, psd[neuron_id], label=label, alpha=0.7, 
                       linewidth=0.5, color=color)

    for idx, ax in enumerate(axs):
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title(plot_titles[idx])
        ax.axvline(x=50, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=100, color='k', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(8, 150)
        
        if log_plot:
            if idx == 0:  # Linear plot
                ax.set_ylabel('Power Spectral Density')
            elif idx == 1:  # Log-Y plot
                ax.set_ylabel('Log₁₀(Power Spectral Density)')
                ax.set_yscale('log')
                ax.set_xlim(8, 300)
            elif idx == 2:  # Log-Log plot
                ax.set_ylabel('Log₁₀(Power Spectral Density)')
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(8, 300) 
        else:
            ax.set_ylabel('Power Spectral Density')

        if n_neurons_to_plot <= 10:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def psd_mask(freqs, psd, range_mask):
    """
    Mask the power spectral density based on the frequency mask.
    
    Args:
        freqs: frequencies corresponding to the power spectrum
        psd: power spectral density values (shape: n_neurons x n_freqs)

    Returns:
        masked_psd: masked power spectral density
    """
    print(f"Frequencies spacing in psd: {freqs[1] - freqs[0]:.2f} Hz")
    
    freqs_mask = (freqs >= range_mask[0]) & (freqs <= range_mask[1])

    print(f"number of frequencies in the range {range_mask[0]}-{range_mask[1]} Hz: {np.sum(freqs_mask)}")
    extracted_freqs = freqs[freqs_mask]
    print(f"Extracted frequencies in the range: {extracted_freqs}")

    col_names = [f"{freq:.2f}Hz" for freq in extracted_freqs]

    psd_masked = pd.DataFrame(psd[:, freqs_mask], columns=col_names)
    print(f"Shape of psd_masked: {psd_masked.shape}")

    return psd_masked

def extract_fooof_features(psd_data, freqs):
    """
    Extract comprehensive FOOOF features for each Purkinje cell
    """
    n_peaks = 1
    all_features = []
    low_w_limit = np.min(freqs) * 2
    print(f"Low peak width limit: {low_w_limit:.2f} Hz")
    for cell, psd in enumerate(psd_data):
        cell_features = np.zeros(n_peaks * 3 + 2 + 1)
        peak_params = []
        fm = FOOOF(peak_width_limits=(low_w_limit, 3000))
        fm.fit(freqs, psd)
        print(f"For cell {cell}")
        print(f"error of the model: {fm.error_:.4f}")
        print(f"R^2 of the model: {fm.r_squared_:.4f}\n")

        # aperiodic parameters
        aperiodic_params = fm.get_params('aperiodic_params')
        offset = aperiodic_params[0]
        exponent = aperiodic_params[1]
        
        # peak parameters
        peak_ = fm.get_params('peak_params')
        if len(peak_) > 0:
            arg_peak = peak_[peak_[:, 1].argsort()[::-1]]
            sorted_peaks = arg_peak[:n_peaks, :]

            for i , peak in enumerate(sorted_peaks):
                cell_features[i*3:(i+1)*3] = peak
        
        cell_features[-3] = offset
        cell_features[-2] = exponent
        cell_features[-1] = np.sum(psd)  # total energy
        all_features.append(cell_features)

    all_features = np.array(all_features)
    print(f"Extracted FOOOF features shape: {all_features.shape}")

    foof_features_df = pd.DataFrame(all_features, columns=[
    'Peak1_Freq', 'Peak1_Power', 'Peak1_Bandwidth',
    'Aperiodic_Offset', 'Aperiodic_Exponent', 'Total_Energy_Entire_PSD'])
    
    return foof_features_df

def hilbert_phase(rates_per_neuron):
    """
    Apply Hilbert transform to a signal and compute the mean and standard deviation of the instantaneous phase for each neuron.
    Args:
        rates_per_neuron: 2D numpy array of shape (n_neurons, n_time_bins)
    Returns:
        mean_phase_per_neuron: 1D numpy array of mean instantaneous phase for each neuron
        std_phase_per_neuron: 1D numpy array of standard deviation of instantaneous phase for each neuron
    """
    mean_phase_per_neuron = np.zeros((rates_per_neuron.shape[0],))
    std_phase_per_neuron = np.zeros((rates_per_neuron.shape[0],))
    for i, rate in enumerate(rates_per_neuron):
        signal = rate - np.mean(rate)
        analytic_signal = hilbert(signal)
        phase_wrapped =  np.mod(np.angle(analytic_signal), 2*np.pi)
        # print(f"shape of instantaneous_phase for neuron {i}: {instantaneous_phase.shape}")
        mean_phase_per_neuron[i] = circmean(phase_wrapped, high=2*np.pi, low=0)
        std_phase_per_neuron[i] = np.std(phase_wrapped)
    
    print("Statistics:")
    print(f"Mean phase per neuron shape: {mean_phase_per_neuron.shape}")
    print(f"Std phase per neuron shape: {std_phase_per_neuron.shape}")
    print(f"Mean phase per neuron: {mean_phase_per_neuron[:5]}")
    print(f"Std phase per neuron: {std_phase_per_neuron[:5]}")
    print(f"Mean phase per neuron max: {np.max(mean_phase_per_neuron)}")
    print(f"Mean phase per neuron min: {np.min(mean_phase_per_neuron)}")

    return mean_phase_per_neuron, std_phase_per_neuron

def psd_without_aperiodic(psd, freqs, foof_features):
    """
    Remove the aperiodic component from the power spectrum.
    
    Args:
        psd: Power spectral density (2D array).
        freqs: Frequencies corresponding to the PSD.
        foof_features: DataFrame containing FOOOF features.

    Returns:
        psd_no_aperiodic: Power spectral density without the aperiodic component.
    """
    psd_no_aperiodic = np.zeros_like(psd)
    for i in range(psd.shape[0]):
        offset, exponent = foof_features.iloc[i][['Aperiodic_Offset', 'Aperiodic_Exponent']].values
        aperiodic_comp = offset - exponent * np.log10(freqs)
        log_residual = np.log10(psd[i]) - aperiodic_comp
        psd_no_aperiodic[i] = 10 ** log_residual
    return psd_no_aperiodic


def psd_mask(freqs, psd, range_mask):
    """
    Mask the power spectral density based on the frequency mask.
    Args:
        freqs: frequencies corresponding to the power spectrum
        psd: power spectral density values (shape: n_neurons x n_freqs)
        range_mask: frequency range to mask (min, max)
    Returns:
        masked_psd: masked power spectral density
    """
    print(f"Frequencies spacing in psd: {freqs[1] - freqs[0]:.2f} Hz")
    freqs_mask = (freqs >= range_mask[0]) & (freqs <= range_mask[1])
    print(f"number of frequencies in the range {range_mask[0]}-{range_mask[1]} Hz: {np.sum(freqs_mask)}")
    extracted_freqs = freqs[freqs_mask]
    print(f"Extracted frequencies in the range: {extracted_freqs}")
    col_names = [f"{freq:.2f}Hz" for freq in extracted_freqs]
    
    psd_masked = pd.DataFrame(psd[:, freqs_mask], columns=col_names)
    print(f"Shape of psd_masked: {psd_masked.shape}")
    
    return psd_masked, freqs_mask



