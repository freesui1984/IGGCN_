import numpy as np
import pandas as pd
import mne

from eeg_TUSZ_19 import standardize_sensors, downsample, highpass, remove_line_noise, get_brain_waves_power
from tqdm import tqdm  # Import tqdm for progress bars


log_amplitude_feature_matrix = []  # Features used to store logarithmic amplitudes

SAMPLING_FREQ = 200.0  # Sampling frequency
"""
Group recordings by where they are stored. 
You do not want to open recordings repeatedly for different Windows
"""
index_df = pd.read_csv("F:\\IGECN\\data\\tusz_eeg\\4_class\\4_class_window_index.csv", dtype={"patient_ID": str})
grouped_df = index_df.groupby("raw_file_path")
adjacency_feature_matrix = np.zeros((index_df.shape[0], 19*19))
# Define a progress bar for the outer loop
outer_loop_bar = tqdm(grouped_df, desc="Processing Files", total=len(grouped_df))


for raw_file_path, group_df in outer_loop_bar:
    print(f"FILE NAME: {raw_file_path}")
    channel_config = str(group_df["channel_config"].unique()[0])

    # Preprocessing = Open file, select channel, apply montage, downsample to 200
    raw_data = mne.io.read_raw_edf(raw_file_path, verbose=True, preload=True)
    raw_data = standardize_sensors(raw_data, channel_config, return_montage=False)
    raw_data, sfreq = downsample(raw_data, SAMPLING_FREQ)

    for window_idx in group_df.index.tolist():
        # Get the raw data of the window
        start_sample = group_df.loc[window_idx]['start_sample_index']
        stop_sample = group_df.loc[window_idx]['end_sample_index']
        window_data = raw_data.get_data(start=start_sample, stop=stop_sample)

        # Fourier transform the window data
        window_data_fft = np.fft.fft(window_data, n=int(SAMPLING_FREQ), axis=-1)

        # only take the positive freq part
        idx_pos = int(np.floor(window_data_fft.shape[1] / 2))
        window_data_fft = window_data_fft[:, :idx_pos]
        amp = np.abs(window_data_fft)
        amp[amp == 0.0] = 1e-8  # avoid log of 0

        # Calculate logarithmic amplitude characteristics
        log_amplitude = np.log(amp)
        log_amplitude_feature_matrix.append(log_amplitude)

        transf_window_data = np.expand_dims(window_data, axis=0)

        # Compute the characteristics of the adjacency matrix (spectral correlation)
        from mne_connectivity import spectral_connectivity_epochs

        for ch_idx in range(19):
            spec_conn = spectral_connectivity_epochs(data=transf_window_data,
                                                     method='coh',
                                                     indices=([ch_idx] * 19, range(19)),
                                                     sfreq=SAMPLING_FREQ,
                                                     fmin=1.0, fmax=40.0,  # 适当设置频率范围
                                                     faverage=True, verbose=False)

            spec_conn_value = spec_conn.get_data()
            assert spec_conn_value.shape[0] == 19
            spec_conn_values = np.squeeze(spec_conn_value)
            start_edge_idx = ch_idx * 19
            end_edge_idx = start_edge_idx + 19
            adjacency_feature_matrix[window_idx, start_edge_idx:end_edge_idx] = spec_conn_values
    # Update the outer loop progress bar
    outer_loop_bar.update(1)
    print(f"\nfile: '{raw_file_path}' [RECORDING] ALL WINDOWS DONE! FILE DONE!...\n")

# Converts a feature matrix to a NumPy array
log_amplitude_feature_matrix = np.array(log_amplitude_feature_matrix)
adjacency_feature_matrix = np.array(adjacency_feature_matrix)

from sklearn.preprocessing import MinMaxScaler
# Normalize the maximum and minimum
min_max_scaler = MinMaxScaler()
log_amplitude_feature_matrix_scaled = min_max_scaler.fit_transform(log_amplitude_feature_matrix.reshape(-1, log_amplitude_feature_matrix.shape[-1])).reshape(log_amplitude_feature_matrix.shape)

adjacency_feature_matrix_scaled = min_max_scaler.fit_transform(adjacency_feature_matrix.reshape(-1, adjacency_feature_matrix.shape[-1])).reshape(adjacency_feature_matrix.shape)

# Save
np.save("F:\\IGECN\\data\\tusz_eeg\\4_class\\feature_matrix.npy", log_amplitude_feature_matrix_scaled)
np.save("F:\\IGECN\\data\\tusz_eeg\\4_class\\adjacency_feature_matrix.npy", adjacency_feature_matrix_scaled)
np.save("F:\\IGECN\\data\\tusz_eeg\\4_class\\label_y.npy", index_df["label"].to_numpy())


