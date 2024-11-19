import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import os
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import stats

# Set the layout to wide (this must be placed at the very top of the script)
st.set_page_config(layout="wide")

# Load the necessary .npy files from the 'numpy_files' folder
all_wavelet_features = np.load('numpy_files/wavelet_features.npy')
all_labels = np.load('numpy_files/wavelet_labels.npy')

accuracy = np.load('numpy_files/accuracy.npy')[0]
classification_report_data = np.load('numpy_files/classification_report.npy', allow_pickle=True).item()
cv_scores = np.load('numpy_files/cv_scores.npy')
confusion_matrix = np.load('numpy_files/confusion_matrix.npy')

# XGBoost results
xgb_accuracy = np.load('numpy_files/xgb_accuracy.npy')[0]
xgb_classification_report_data = np.load('numpy_files/xgb_classification_report.npy', allow_pickle=True).item()
xgb_cv_scores = np.load('numpy_files/xgb_cv_scores.npy')
xgb_confusion_matrix = np.load('numpy_files/xgb_confusion_matrix.npy')

# KNN results
knn_accuracy = np.load('numpy_files/knn_accuracy.npy')[0]
knn_classification_report_data = np.load('numpy_files/knn_classification_report.npy', allow_pickle=True).item()
knn_cv_scores = np.load('numpy_files/knn_cv_scores.npy')
knn_confusion_matrix = np.load('numpy_files/knn_confusion_matrix.npy')

# Sleep stage data
sleep_stage_durations = np.load('numpy_files/sleep_stage_durations_full.npy', allow_pickle=True).item()
chart_image = Image.open('numpy_files/sleep_stage_durations_chart.npy')
sleep_transition_image = Image.open('transition_matrix_heatmap.png')  # Changed the file path

# Streamlit layout
st.title("EEG Sleep Stage Classification")

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio(
    "Select a report to view:",
    ("Classification Report", "Confusion Matrix", "T-Test", "Sleep Stage Distribution", "Sleep Stage Transition",
    "EEG Signal Visualization", "Dataset Visualization", "Frequency Bands", "Power Spectral Density (PSD)")
)

# Classification Report Section
if selection == "Classification Report":
    st.header("Classification Reports")
    st.subheader("Random Forest Classification Report")
    st.json(classification_report_data)

    st.subheader("XGBoost Classification Report")
    st.json(xgb_classification_report_data)

    st.subheader("KNN Classification Report")
    st.json(knn_classification_report_data)

# Confusion Matrix Section
elif selection == "Confusion Matrix":
    st.header("Confusion Matrix Visualizations")

    st.subheader("Random Forest Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(all_labels)).plot(ax=ax)
    st.pyplot(fig)

    st.subheader("XGBoost Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=xgb_confusion_matrix, display_labels=np.unique(all_labels)).plot(ax=ax)
    st.pyplot(fig)

    st.subheader("KNN Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=knn_confusion_matrix, display_labels=np.unique(all_labels)).plot(ax=ax)
    st.pyplot(fig)

# T-Test Section (Instead of Z-Test)
elif selection == "T-Test":
    st.header("T-Test Results Between Models")
    
    cv_scores_rf = np.load('numpy_files/cv_scores.npy')
    cv_scores_xgb = np.load('numpy_files/xgb_cv_scores.npy')
    cv_scores_knn = np.load('numpy_files/knn_cv_scores.npy')

    # Paired t-test between Random Forest and XGBoost
    t_stat_rf_xgb, p_value_rf_xgb = stats.ttest_rel(cv_scores_rf, cv_scores_xgb)
    st.write(f"Random Forest vs XGBoost: t = {t_stat_rf_xgb:.3f}, p = {p_value_rf_xgb:.3f}")

    # Paired t-test between Random Forest and KNN
    t_stat_rf_knn, p_value_rf_knn = stats.ttest_rel(cv_scores_rf, cv_scores_knn)
    st.write(f"Random Forest vs KNN: t = {t_stat_rf_knn:.3f}, p = {p_value_rf_knn:.3f}")

    # Paired t-test between XGBoost and KNN
    t_stat_xgb_knn, p_value_xgb_knn = stats.ttest_rel(cv_scores_xgb, cv_scores_knn)
    st.write(f"XGBoost vs KNN: t = {t_stat_xgb_knn:.3f}, p = {p_value_xgb_knn:.3f}")

# Sleep Stage Distribution Section (Pie Chart)
elif selection == "Sleep Stage Distribution":
    st.header("Sleep Stage Distribution (Pie Chart)")

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.pie(sleep_stage_durations.values(), labels=sleep_stage_durations.keys(), autopct='%1.1f%%', startangle=90, colors=['blue', 'orange', 'green', 'red', 'purple'])
    plt.title('Sleep Stage Distribution (Percentage)')
    st.pyplot(fig)

# Sleep Stage Transition Section (Transition Chart and Graph)
elif selection == "Sleep Stage Transition":
    st.header("Sleep Stage Transition Heatmap")

    # Display the heatmap image with updated parameter
    st.image(sleep_transition_image, use_container_width=True)

    # Additional Graph for Sleep Stage Transition
    st.subheader("Sleep Stage Transition Graph")
    
    # Load and display the transition graph
    graph_path = "sleep_stage_transition_graph.png"  # Ensure correct path
    graph_image = Image.open(graph_path)
    st.image(graph_image, caption="Sleep Stage Transition Graph", use_container_width=True)  # Display in Streamlit


# Dataset Visualization Section
elif selection == "Dataset Visualization":
    st.header("Dataset Visualization")

    # Take user input for the file number
    file_number = st.text_input("Enter the file number (e.g., 002 for SN002):", "")

    if file_number:
        # Base directory for EDF files
        edf_dir = r'physionet.org/files/hmc-sleep-staging/1.0.0/recordings'

        # Construct the EDF file path
        edf_file = os.path.join(edf_dir, f'SN{file_number}.edf')

        # Check if the file exists (to handle missing files like SN138)
        if not os.path.exists(edf_file):
            st.error(f"File SN{file_number}.edf not found. Please enter a valid file number.")
        else:
            # Load the EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True)

            # Display channel selection dropdown
            available_channels = raw.info['ch_names']  # Get available channels
            selected_channel = st.selectbox("Select a channel to visualize:", available_channels)

            # Specify the duration of data to display (e.g., 10 seconds)
            display_duration = 10.0  # in seconds

            # Get the total duration of the recording
            total_duration = raw.times[-1]

            # Set a slider to select the start time
            start_time = st.slider(
                "Select the start time (seconds):",
                min_value=0.0,
                max_value=total_duration - display_duration,
                step=0.1
            )

            # Check if the selected channel is valid
            if selected_channel:
                st.subheader(f"Signal for {selected_channel} (10-second segment starting at {start_time:.1f} seconds)")

                # Copy raw data and pick the selected channel
                channel_data = raw.copy().pick([selected_channel])

                # Crop the data to the selected 10-second segment
                channel_data.crop(tmin=start_time, tmax=start_time + display_duration)

                # Extract the data and time for the selected segment
                signal_data = channel_data.get_data()[0]  # Extract the data for the selected channel
                time = channel_data.times  # Time vector corresponding to the cropped segment

                # Plot the data
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(time, signal_data)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.set_title(f"{selected_channel} Signal (10-second segment)")
                st.pyplot(fig)

elif selection == "Frequency Bands":
    st.header("Frequency Bands")

    # Take user input for the file number
    file_number = st.text_input("Enter the file number (e.g., 002 for SN002):", "")

    if file_number:
        # Base directory for EDF files
        edf_dir = r'physionet.org/files/hmc-sleep-staging/1.0.0/recordings'

        # Construct the EDF file path
        edf_file = os.path.join(edf_dir, f'SN{file_number}.edf')

        # Check if the file exists (to handle missing files like SN138)
        if not os.path.exists(edf_file):
            st.error(f"File SN{file_number}.edf not found. Please enter a valid file number.")
        else:
            # Load the EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True)

            # Pick a specific EEG channel (e.g., 'EEG C4-M1')
            eeg_m1 = raw.copy().pick_channels(['EEG C4-M1'])

            # Filtering the EEG data into frequency bands
            delta_band = eeg_m1.copy().filter(0.5, 4, fir_design='firwin')
            theta_band = eeg_m1.copy().filter(4, 8, fir_design='firwin')
            alpha_band = eeg_m1.copy().filter(8, 13, fir_design='firwin')
            beta_band = eeg_m1.copy().filter(13, 30, fir_design='firwin')

            # Get duration of the recording in seconds
            duration = int(raw.times[-1])

            # Select frequency band
            band_option = st.selectbox(
                "Select Frequency Band to Visualize:",
                ["Original Signal", "Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-12 Hz)", "Beta (12-30 Hz)"]
            )

            # Select a time interval (start time in seconds)
            start_time = st.slider(
                "Select Start Time for Visualization (seconds):",
                min_value=0,
                max_value=max(0, duration - 10),
                value=0
            )

            # Calculate time range for the selected interval
            sfreq = int(eeg_m1.info['sfreq'])  # Sampling frequency
            times = eeg_m1.times[start_time * sfreq:(start_time + 10) * sfreq]

            # Get data for the selected band
            if band_option == "Original Signal":
                selected_band = eeg_m1
                label = "Original Signal"
                color = "black"
            elif band_option == "Delta (0.5-4 Hz)":
                selected_band = delta_band
                label = "Delta (0.5-4 Hz)"
                color = "blue"
            elif band_option == "Theta (4-8 Hz)":
                selected_band = theta_band
                label = "Theta (4-8 Hz)"
                color = "green"
            elif band_option == "Alpha (8-12 Hz)":
                selected_band = alpha_band
                label = "Alpha (8-12 Hz)"
                color = "orange"
            elif band_option == "Beta (12-30 Hz)":
                selected_band = beta_band
                label = "Beta (12-30 Hz)"
                color = "red"

            # Plot the selected band
            st.subheader(f"{label} Signal (10 seconds from {start_time} to {start_time + 10} seconds)")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(times, selected_band.get_data()[0][:len(times)], label=label, color=color, alpha=0.7)
            ax.set_title(f"{label} Signal")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude (µV)")
            ax.grid(True)
            st.pyplot(fig)

elif selection == "Power Spectral Density (PSD)":
    st.header("Power Spectral Density (PSD)")

    # Take user input for the file number
    file_number = st.text_input("Enter the file number (e.g., 002 for SN002):", "")

    if file_number:
        # Base directory for EDF files
        edf_dir = r'physionet.org/files/hmc-sleep-staging/1.0.0/recordings'

        # Construct the EDF file path
        edf_file = os.path.join(edf_dir, f'SN{file_number}.edf')

        # Check if the file exists (to handle missing files like SN138)
        if not os.path.exists(edf_file):
            st.error(f"File SN{file_number}.edf not found. Please enter a valid file number.")
        else:
            # Load the EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True)

            # Select a specific EEG channel
            channel_name = "EEG C4-M1"

            if channel_name:
                # Create a copy of the selected channel
                eeg_m1 = raw.copy().pick_channels([channel_name])

                # Apply band-pass filter from 0.5 to 30 Hz
                eeg_m1.filter(l_freq=0.5, h_freq=30.0, fir_design="firwin")

                # Resample data to 100 Hz
                eeg_m1.resample(100)

                # Compute the Power Spectral Density (PSD)
                psd = eeg_m1.compute_psd(fmin=0.5, fmax=30.0)

                # Plot the PSD
                st.subheader(f"Power Spectral Density (PSD) for {channel_name}")
                fig_psd = psd.plot()  # Directly obtain the figure
                st.pyplot(fig_psd)

                # Display PSD values in a table (optional)
                st.write("PSD values for each frequency band:")
                psd_table = pd.DataFrame({
                    "Frequency (Hz)": psd.freqs,
                    "PSD (dB)": psd.get_data()[0]
                })
                st.dataframe(psd_table)