# streamlit run seizure_detection_app_v4.py

import streamlit as st
import mne
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tempfile import NamedTemporaryFile
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import roc_curve

TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", 
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", 
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", 
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2", 
    "FZ-CZ", "CZ-PZ"
]

@st.cache_resource
def load_model():
    from tensorflow.keras.models import load_model
    import tensorflow.keras.layers as layers

    def custom_batch_norm_from_config(config):
        if isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return layers.BatchNormalization(**config)

    layers.BatchNormalization.from_config = custom_batch_norm_from_config

    return load_model("./models/seizure_detection_inceptionv3_11.h5")

model = load_model()

def process_raw_data(raw):
    summary = model.summary()
    print(summary)
    mapping = {ch: ch.replace('-0', '') for ch in raw.ch_names if '-0' in ch}
    raw.rename_channels(mapping)
    missing_channels = set(TARGET_CHANNELS) - set(raw.ch_names)
    if missing_channels:
        raise ValueError(f"File is missing required channels: {missing_channels}")

    raw.pick_channels(TARGET_CHANNELS)
    raw.filter(l_freq=1., h_freq=40.)
    return raw

def analyze_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw = process_raw_data(raw)

    epoch_duration = 3.0
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(raw, events, preload=True)
    epochs_data = epochs.get_data()

    epochs_data = epochs_data * 1e6

    print(f"Raw Data Shape: {epochs_data.shape}")
    print(f"Raw Data Min: {epochs_data.min()}, Max: {epochs_data.max()}, Mean: {epochs_data.mean()}")

    resized_data = np.array([
        tf.image.resize(
            np.repeat(np.expand_dims(epoch, axis=-1), 3, axis=-1), (299, 299)
        ).numpy()
        for epoch in epochs_data
    ])

    scaled_data = (resized_data - resized_data.min()) / (resized_data.max() - resized_data.min()) * 255.0
    normalized_data = preprocess_input(scaled_data)

    predictions = model.predict(normalized_data).flatten()

    window_size = 5 
    smoothed_predictions = np.convolve(predictions, np.ones(window_size) / window_size, mode='same')

    seizure_indices = np.where(smoothed_predictions >= 0.98)[0]

    sfreq = raw.info['sfreq']
    raw_start_time = raw.first_samp / sfreq
    seizure_timeframes = [
        (raw_start_time + (events[idx, 0] / sfreq), raw_start_time + (events[idx, 0] / sfreq) + epoch_duration)
        for idx in seizure_indices
    ]

    for idx in seizure_indices:
        start = raw_start_time + (events[idx, 0] / sfreq)
        end = start + epoch_duration
        print(f"Epoch {idx}: Start={start:.2f}s, End={end:.2f}s")

    return smoothed_predictions, seizure_timeframes, seizure_indices, raw

def merge_timeframes(timeframes):
    if not timeframes:
        return []

    merged = [timeframes[0]]
    for current in timeframes[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            merged.append(current)
    return merged

def plot_eeg(raw, start_time, duration):
    try:
        sfreq = raw.info['sfreq']
        start_sample = int(start_time * sfreq)
        end_sample = int((start_time + duration) * sfreq)
        data, times = raw[:, start_sample:end_sample]
        data = data * 1e6

        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(data.shape[0]):
            ax.plot(times, data[i] + i * 100, label=f"Channel {i + 1}")
        ax.set_title(f"EEG Signal from {start_time:.2f}s to {start_time + duration:.2f}s")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper right")
        return fig
    except Exception as e:
        st.error(f"An error occurred while plotting EEG data: {e}")
        return None

st.title("EEG Seizure Detection App")
st.write("Upload an EEG file in `.edf` format to analyze it for seizures.")

uploaded_file = st.file_uploader("Upload an EDF file", type="edf")

if uploaded_file is None:
    st.session_state.clear()

if uploaded_file:
    if "file_path" not in st.session_state or st.session_state.file_path != uploaded_file.name:
        st.session_state.analysis_triggered = False

    if not st.session_state.get("analysis_triggered", False):
        with st.spinner("Analyzing EEG data..."):
            try:
                with NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name

                smoothed_predictions, seizure_timeframes, seizure_indices, raw = analyze_eeg(temp_file_path)
                st.session_state.results = {
                    "predictions": smoothed_predictions,
                    "seizure_timeframes": merge_timeframes(seizure_timeframes),
                    "seizure_indices": seizure_indices,
                    "raw": raw,
                }
                st.session_state.file_path = uploaded_file.name
                st.session_state.analysis_triggered = True
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

if "results" in st.session_state:
    results = st.session_state.results
    predictions = results["predictions"]
    seizure_timeframes = results["seizure_timeframes"]
    seizure_indices = results["seizure_indices"]
    raw = results["raw"]

    st.success(f"Analysis complete! Detected {len(seizure_timeframes)} seizures.")

    st.subheader("EEG Visualization")

    if seizure_indices.size > 0:
        selected_epoch = st.selectbox(
            "Select an epoch with detected seizures:",
            options=seizure_indices,
            format_func=lambda x: f"Epoch {x} ({x*3:.2f}s to {(x+1)*3:.2f}s)"
        )

        st.subheader("Selected Seizure Epoch")
        fig = plot_eeg(raw, selected_epoch * 3, 3.0)
        if fig:
            st.pyplot(fig)
    else:
        st.warning("No seizures were detected in the uploaded EEG file.")

    st.write("Or use sliders to view any timeframe:")
    start_time = st.slider("Start Time (seconds)", 0, int(raw.times[-1]), 0, 1)
    duration = st.slider("Duration (seconds)", 1, 30, 10, 1)

    st.subheader("Custom Timeframe")
    fig = plot_eeg(raw, start_time, duration)
    if fig:
        st.pyplot(fig)

    # st.subheader("Seizure Probabilities by Epoch")
    # st.bar_chart(predictions)

    st.subheader("Merged Seizure Timeframes")
    if seizure_timeframes:
        for i, (start, end) in enumerate(seizure_timeframes):
            st.write(f"Seizure {i + 1}: Start = {start:.2f}s, End = {end:.2f}s")
    else:
        st.write("No seizures detected.")
