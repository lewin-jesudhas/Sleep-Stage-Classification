import mne
import os
import numpy as np
#! pip install skikit-learn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
%matplotlib inline
raw = mne.io.read_raw_edf(r'physionet.org/files/hmc-sleep-staging/1.0.0/recordings/SN023.edf', preload=True)
print(raw.info)
channel_names = raw.ch_names
print("Channels:", channel_names)
raw.plot(title='Signals');
total_duration = raw.times[-1] 
#print(raw.times, total_duration) # Get the total duration of the signal
raw.crop(tmin=150, tmax=total_duration - 150)
raw.plot(scalings='auto', title='EEG Data (After Cropping 150s)', show=True);
eeg_m1 = raw.copy().pick(['EEG C4-M1'])
print(eeg_m1)
eeg_m1.plot(title='EEG C4-M1');
eeg_m1.filter(0.5, 30, fir_design='firwin')
eeg_m1.resample(100)
eeg_m1.plot(title='EEG C4-M1');
# ! pip install pandas
import pandas as pd
annotations_df = pd.read_csv(r'C:\Users\lewin\OneDrive\Desktop\College Semesters\Sleep Stage Classification\physionet.org\files\hmc-sleep-staging\1.0.0\recordings\SN023_sleepscoring.txt')  # Replace with the path to your file
annotations_df.columns = ["Date", "Time", "Recording onset", "Duration", "Annotation", "Linked channel"]
# Convert onset and duration to numeric values (in seconds) if needed
annotations_df['Recording onset'] = pd.to_numeric(annotations_df['Recording onset'], errors='coerce')
annotations_df['Duration'] = pd.to_numeric(annotations_df['Duration'], errors='coerce')
# Check the data
print(annotations_df.head())
# Total duration of the recording after cropping
total_duration = raw.times[-1] - 150 * 2  # After removing 150 seconds from start and end
# Filter annotations to fit within the cropped data range
filtered_annotations_df = annotations_df[
    (annotations_df['Recording onset'] >= 150) &
    (annotations_df['Recording onset'] <= total_duration)
].copy()
# Adjust onset to account for cropping the initial 150 seconds
filtered_annotations_df['Recording onset'] -= 150
print(filtered_annotations_df.head())
# Convert to MNE annotations
annotations = mne.Annotations(
    onset=filtered_annotations_df['Recording onset'].values,
    duration=filtered_annotations_df['Duration'].values,
    description=filtered_annotations_df['Annotation'].values
)

# Set the annotations to the raw object
raw.set_annotations(annotations)
# Bandpass filter and resample
raw.filter(0.5, 30, fir_design='firwin')
raw.resample(100)
# Create events from annotations
events, event_id = mne.events_from_annotations(raw)

# Epoch the data based on events
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=30, baseline=None, preload=True)

# Encode the labels for machine learning
labels = epochs.events[:, -1]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
import numpy as np
import pywt

# Define the wavelet transform function
def compute_wavelet_features(data, wavelet='db4', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Extract statistics from each wavelet level (mean, std deviation)
    features = [np.mean(c) for c in coeffs] + [np.std(c) for c in coeffs]
    return features

# Get the EEG data from the epochs object
eeg_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_timepoints)

# Apply wavelet transform on each epoch and extract features
wavelet_features = []

for epoch in eeg_data:
    # Flatten the data for each channel and extract features
    for channel_data in epoch:  # Iterate over each channel in the epoch
        features = compute_wavelet_features(channel_data)  # Process each channel separately
        wavelet_features.append(features)

# Convert to NumPy array for ML
wavelet_features = np.array(wavelet_features)
print(wavelet_features)
# Reshape wavelet features if needed
n_epochs = eeg_data.shape[0]  # Number of epochs
n_channels = eeg_data.shape[1]  # Number of channels
n_features_per_channel = len(wavelet_features[0])  # Number of wavelet features per channel

# Reshape to (n_epochs, n_channels * n_features_per_channel)
wavelet_features = wavelet_features.reshape(n_epochs, n_channels * n_features_per_channel)
print("Reshaped wavelet features:", wavelet_features.shape)
# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(wavelet_features, labels, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Model training completed.")
# Make predictions
y_pred = clf.predict(X_test)

# Display accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Display the confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_)
plt.title("Confusion Matrix of Sleep Stage Classification")
plt.show()
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(clf, wavelet_features, labels, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Average CV Score:", np.mean(cv_scores))


