from neo import io
import neo
import numpy as np
from pathlib import Path
import matplotlib.pylab as plt 
import pandas as pd
from pathlib import Path
from scipy.signal import convolve
from scipy.signal.windows import triang
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
from scipy.stats import spearmanr

def compute_firing_rate(spiking_data: np.array, t_start=0, t_end=None, dt=0.1, sigma=100, compute_stats=True, mask_flag=True):
    """
    Args:
        - Spiking_data: 2D numpy array with shape (N, 2), where N is the number of spikes
        and the first column is spike times and the second column is neuron IDs.
        - t_start : start time in ms
        - t_end : end time in ms, will be defaulted to the maximum spike time + 1
        - dt : time step in ms
        - sigma : standard deviation of the triangular kernel in ms
    Returns:
        - times_vector : The time points at which firing rates are calculated (in ms)
        - rate : The average firing rate across active neurons at each time point (in Hz)
        - rates_array : 2D numpy array with shape (num_neurons, num_bins), with firing rates for each neuron at each time point
        - stats_dict : dictionary with mean, std, max, min firing rates (if compute_stats is True)
    """

    if isinstance(spiking_data, np.ndarray) and len(spiking_data.shape) == 2:
        times = spiking_data[:, 0]
        ids = spiking_data[:, 1]
    else:
        print("Invalid input data format. Expected a 2D numpy array.")
        return None

    # Filtering unique neurons
    neurons_ids = np.unique(ids)
    num_neurons = neurons_ids.shape[0]
    print("Number of neurons:", num_neurons)
    
    # Creating the time vector
    if t_end is None:
        t_end = times.max() + 1 

    print("Time vector:", t_start, t_end)
    
    times_vector = np.arange(t_start, t_end, dt)
    num_bins = times_vector.shape[0]
    print("Number of bins:", num_bins)

    # Creating the kernel
    kernel_size = int(2 * sigma / dt)
    kernel_size = max(kernel_size, 3) 
    kernel = triang(kernel_size)
    kernel = kernel / np.sum(kernel)  # Normalizing the kernel
    print("Kernel size:", kernel_size)
    

    rates_per_neuron = []
    active_neurons = 0
    
    for i, neuron_id in enumerate(neurons_ids):
        neuron_spikes = times[ids == neuron_id]
        
        if len(neuron_spikes) == 0:
            continue 
        
        active_neurons += 1
        if i < 3: 
            print(f"Neuron ID: {neuron_id}")
            print(f"First few spikes: {neuron_spikes[:5]}")
        
        single_neuron_train = np.zeros_like(times_vector)
        positions = (neuron_spikes / dt).astype(int)
        positions = positions[positions < num_bins]
        single_neuron_train[positions] = 1
        
        single_rate = convolve(single_neuron_train, kernel, mode='same') * (1000.0 / dt)
        rates_per_neuron.append(single_rate)
    
    if not rates_per_neuron:
        print("No active neurons found")
        return None
        
    rates_array = np.vstack(rates_per_neuron) # Shape: (num_neurons, num_bins)
    print("Rates array shape:", rates_array.shape)
    print("Rates array first few values:", rates_array[:5])
    print("Rates array first few neurons:", rates_array[:, :5])
    rate = np.mean(rates_array, axis=0)
    print("Rate shape:", rate.shape)
    print("Rate first few values:", rate[:5])

    if mask_flag:
        window_start = 250
        window_end = times_vector[-1] - 250
        mask = (times_vector >= window_start) & (times_vector <= window_end)
        rate = rate[mask]
        rates_array = rates_array[:, mask]
        times_vector = times_vector[mask]
    
    print("---"*20)
    print(f"Active neurons: {active_neurons} out of {num_neurons}")
    print("Firing rate shape:", rate.shape)
    print("Firing rate first few values:", rate[:5])
    
    if compute_stats:
        mean_rate = np.mean(rate)
        std_rate = np.std(rate)
        max_rate = np.max(rate)
        min_rate = np.min(rate)
        print(f"Mean firing rate: {mean_rate:.4f}+/-{std_rate:.4f}")
        print(f"Max firing rate: {max_rate:.4f}")
        print(f"Min firing rate: {min_rate:.4f}")
        stats_dict = {
            'mean': mean_rate,
            'std': std_rate,
            'max': max_rate,
            'min': min_rate
        } 

    return times_vector, rate, rates_array, stats_dict if compute_stats else None

def plot_firing_rate(times_vector, rate, ax, pop, color, stats_dict=None, xlabel="Time (ms)", ylabel="Firing Rate (Hz)", save_path=None):
    """
    Plot the firing rate over time.
    """

    if stats_dict is None:
        stats_dict = {
            'mean': np.mean(rate),
            'std': np.std(rate),
            'max': np.max(rate),
            'min': np.min(rate)
        }

    mean_rate = stats_dict['mean']
    std_rate = stats_dict['std']
    max_rate = stats_dict['max']
    min_rate = stats_dict['min']
    print("---"*20)
    print(f"📊 Statistics for population {pop}:")
    print(f"Mean firing rate: {mean_rate:.4f}+/-{std_rate:.4f}")
    print(f"Max firing rate: {max_rate:.4f}")
    print(f"Min firing rate: {min_rate:.4f}")
    
    title = f"{pop} Firing Rate"
    ax.plot(times_vector, rate, label=f'{pop} mean FR', color=color)
    ax.fill_between(times_vector, rate - std_rate, rate + std_rate, alpha=0.3, label=f'±1 SD', color=color)
    ax.axhline(np.mean(mean_rate), color=color, linestyle='--', label=f'Mean = {np.mean(mean_rate):.2f} Hz')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path)


# function for computing the interspike interval
def isi(spiking_data : np.array):
    """
    Args:
        - Spiking_data: 2D numpy array with shape (N, 2), where N is the number of spikes
        and the first column is spike times and the second column is neuron IDs.
    """
    if isinstance(spiking_data, np.ndarray) and len(spiking_data.shape) == 2:
        times = spiking_data[:, 0]
        ids = spiking_data[:, 1]
    else:
        print("Invalid input data format. Expected a 2D numpy array.")
        return None
    
    unique_ids = np.unique(ids)
    num_neurons = len(unique_ids)

    isi_dict = {}
    all_isis = []
    
    for i, n_id in enumerate(unique_ids):
        neuron_spike = times[ids == n_id]
        neuron_spike_sorted = np.sort(neuron_spike)
        neuron_isi = np.diff(neuron_spike_sorted) if len(neuron_spike_sorted) > 1 else np.array([0])
        isi_dict[n_id] = neuron_isi
        all_isis.extend(neuron_isi)

    all_isis = np.array(all_isis)
    if len(all_isis) > 0:
        mean_isi = np.mean(all_isis)
        std_isi = np.std(all_isis)
        min_isi = np.min(all_isis)
        max_isi = np.max(all_isis)
        cv = std_isi / mean_isi if mean_isi > 0 else np.nan
    else:
        mean_isi, std_isi, min_isi, max_isi, cv = np.nan, np.nan, np.nan, np.nan, np.nan
    
    print(f"Coefficient of Variation: {cv:.4f}")
    print(f"Mean ISI: {mean_isi:.4f} +- {std_isi:.4f} ms")
    print(f"Min ISI: {min_isi:.4f} ms, Max ISI: {max_isi:.4f} ms")
    
    stats_dict = {
        "mean": mean_isi,
        "std": std_isi,
        "min": min_isi,
        "max": max_isi,
        "cv": cv,
        "total_count": len(all_isis)
    }
    
    return isi_dict, stats_dict


def plot_isi(isi_dict, stats_dict, ax, pop, color, xlabel="ISI (ms)", ylabel="Count", save_path=None):
    """
    Plot the interspike interval histogram.
    Args:
        - isi_dict: dictionary with neuron IDs as keys and numpy arrays of ISIs as values
        - stats_dict: dictionary with mean, std, min, max, cv of ISIs
        - ax: matplotlib axis to plot on
        - pop: name of the population (for title)
        - color: color for the histogram
        - xlabel: label for the x-axis
        - ylabel: label for the y-axis
        - save_path: path to save the figure (if None, the figure is not saved
    """
    all_isis = np.concatenate(list(isi_dict.values()))
    mean_isi = stats_dict['mean']
    std_isi = stats_dict['std']
    min_isi = stats_dict['min']
    max_isi = stats_dict['max']
    cv = stats_dict['cv']

    print("---"*20)
    print(f"📊 Statistics of Interspike intervals for population {pop}:")
    print(f"Mean ISI: {mean_isi:.4f} +- {std_isi:.4f} ms")
    print(f"Min ISI: {min_isi:.4f} ms, Max ISI: {max_isi:.4f} ms")
    print(f"Coefficient of Variation: {cv:.4f}")
    
    sns.histplot(all_isis, bins=50, kde=True, ax=ax, color=color)
    ax.axvline(mean_isi, color='green', linestyle='--', label=f'Mean ISI = {mean_isi:.2f} ms')
    ax.axvline(mean_isi + std_isi, color='purple', linestyle='--', label=f'Standard Deviation = {std_isi:.2f} ms')
    ax.axvline(mean_isi - std_isi, color='purple', linestyle='--')
    
    ax.set_title(f"{pop} Interspike Interval")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)


def plot_spikes(spiking_data, ax, pop, color, xlabel="Time (ms)", ylabel="Neuron ID", save_path=None):
    """
    Scatter plot of spikes over time.
    Args:
        - Spiking_data: 2D numpy array with shape (N, 2), where N is the number of spikes
        and the first column is spike times and the second column is neuron IDs.
    """
    if isinstance(spiking_data, np.ndarray) and len(spiking_data.shape) == 2:
        times = spiking_data[:, 0]
        ids = spiking_data[:, 1]
    else:
        print("Invalid input data format. Expected a 2D numpy array.")
        return None
    
    title = f"{pop} Spikes"
    sns.scatterplot(x=times, y=ids, ax=ax, s=2.5, color=color, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def process_pop(df_spikes, pop_name, color, mask_flag=True, dt=0.1, sigma=100, save_path=None, plot=True, plot_fr=False, ax=None):
    # getting the data for the population
    df_pop = df_spikes[df_spikes['pop']==pop_name][['time_ms', 'sender_id']]

    # dataframe -> numpy array
    pop_spikes = np.array(df_pop)
    print("Shape:", pop_spikes.shape)
    print(pop_spikes)

    # estimation of the firing rate
    print(f"Computing firing rates of {pop_name}")
    times_vector, mean_rate, rates_per_neuron, stats_pop = compute_firing_rate(pop_spikes, mask_flag=mask_flag, dt=dt, sigma=sigma)

    print(f"Computing interspike intervals")
    isi_dict, stats_dict = isi(pop_spikes)

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
        plot_spikes(pop_spikes, ax=axs[0, 0], pop=pop_name, color=color)
        plot_firing_rate(times_vector, mean_rate, axs[0, 1], pop_name, color=color, stats_dict=stats_pop)
        plot_isi(isi_dict, stats_dict, axs[1, 0], pop_name, color=color)
        plt.delaxes(axs[1, 1])
    else:
        if plot_fr and ax is not None:
            plot_firing_rate(times_vector, mean_rate, ax, pop_name, color=color, stats_dict=stats_pop)

    return times_vector, mean_rate, rates_per_neuron, stats_pop, isi_dict, stats_dict, pop_spikes

def extract_fr(spiking_data, times_vector, rates_per_neuron, mean_rate, 
                       time_interval=(0, 5000), pop='Purkinje Cells', color='blue', 
                       plot=True):
    """
    Extract firing rate, ISI, and spike data for a given time interval.
    
    Args:
        spiking_data: 2D numpy array with shape (N, 2) - raw spike data
        times_vector: time vector in ms from compute_firing_rate
        rates_per_neuron: 2D numpy array of firing rates for each neuron
        mean_rate: mean firing rate across all neurons
        time_interval: tuple of (start_time, end_time) in ms
        pop: name of the population (for plotting)
        color: color for the plot
        plot: whether to plot the results
        
    Returns:
        rates_per_neuron_masked: firing rates for the time interval
        time_mask: boolean mask for the time interval
        interval_spikes: raw spike data for the time interval
        isi_dict: ISI dictionary for the interval
        isi_stats: ISI statistics for the interval
    """
    # Extract firing rates for the interval
    time_mask = (times_vector >= time_interval[0]) & (times_vector <= time_interval[1])
    rates_per_neuron_masked = rates_per_neuron[:, time_mask]
    
    # Extract raw spikes for the interval
    if isinstance(spiking_data, np.ndarray) and len(spiking_data.shape) == 2:
        spike_times = spiking_data[:, 0]
        spike_ids = spiking_data[:, 1]
        
        spike_mask = (spike_times >= time_interval[0]) & (spike_times <= time_interval[1])
        interval_spikes = spiking_data[spike_mask]
        
        # Compute ISI for the interval
        isi_dict, isi_stats = isi(interval_spikes)
        
        duration = time_interval[1] - time_interval[0]
        print(f"Duration of the interval: {duration} ms")
        print(f"Spikes in interval: {len(interval_spikes)}")
        
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
            
            plot_firing_rate(times_vector[time_mask], mean_rate[time_mask], 
                           ax=axs[0, 0], pop=pop, color=color)
            
            plot_spikes(interval_spikes, ax=axs[0, 1], pop=pop, color=color)
            
            plot_isi(isi_dict, isi_stats, ax=axs[1, 0], pop=pop, color=color)
            
            plt.delaxes(axs[1, 1])
            
            fig.suptitle(f'{pop} Analysis: {time_interval[0]}-{time_interval[1]} ms')
            plt.tight_layout()
        
        return rates_per_neuron_masked, time_mask, interval_spikes, isi_dict, isi_stats
    else:
        print("Invalid spiking data format")
        return rates_per_neuron_masked, time_mask, None, None, None