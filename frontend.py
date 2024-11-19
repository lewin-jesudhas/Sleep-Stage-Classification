import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
from scipy import stats
import pandas as pd

# Load your .npy files
def load_data():
    wavelet_features = np.load('numpy_files\wavelet_features.npy')
    wavelet_labels = np.load('numpy_files\wavelet_labels.npy')
    return wavelet_features, wavelet_labels

# Model result loading functions
def load_model_results(model_name):
    accuracy = np.load(f'numpy_files\{model_name}_accuracy.npy')[0]
    classification_report_data = np.load(f'numpy_files\{model_name}_classification_report.npy', allow_pickle=True).item()
    cv_scores = np.load(f'numpy_files\{model_name}_cv_scores.npy')
    confusion_matrix = np.load(f'numpy_files\{model_name}_confusion_matrix.npy')
    return accuracy, classification_report_data, cv_scores, confusion_matrix

# Z-Test Function
def z_test(mean1, std1, mean2, std2, n):
    z_score = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / n)
    p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test
    return z_score, p_value

# Streamlit Layout
st.title("Sleep Stage Classification Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Model Results", "Visualization", "Statistics"])

# Model Results Page
if page == "Model Results":
    st.header("Model Results")
    model_name = st.selectbox("Select Model", ["random_forest", "xgboost", "knn"])

    # Load results for the selected model
    accuracy, classification_report_data, cv_scores, confusion_matrix = load_model_results(model_name)
    
    # Display Model Accuracy
    st.subheader("Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")

    # Classification Report
    st.subheader("Classification Report")
    st.write(classification_report_data)

    # Cross-Validation Scores
    st.subheader("Cross-Validation Scores")
    st.write(f"Cross-Validation Scores: {cv_scores}")
    st.write(f"Average CV Score: {np.mean(cv_scores):.2f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix).plot()
    st.pyplot(plt)

# Visualization Page
elif page == "Visualization":
    st.header("Sleep Stage Visualizations")

    # Sleep Stage Distribution Bar Chart
    st.subheader("Sleep Stage Distribution (Bar Chart)")
    sleep_stage_durations = np.load("numpy_files\sleep_stage_durations_full.npy", allow_pickle=True).item()
    plt.figure(figsize=(10, 6))
    plt.bar(sleep_stage_durations.keys(), sleep_stage_durations.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel('Sleep Stages')
    plt.ylabel('Total Duration (minutes)')
    plt.title('Sleep Stage Distribution Over Time')
    st.pyplot(plt)

    # Sleep Stage Distribution Pie Chart
    st.subheader("Sleep Stage Distribution (Pie Chart)")
    plt.figure(figsize=(8, 8))
    plt.pie(sleep_stage_durations.values(), labels=sleep_stage_durations.keys(), autopct='%1.1f%%', startangle=90, colors=['blue', 'orange', 'green', 'red', 'purple'])
    plt.title('Sleep Stage Distribution (Percentage)')
    st.pyplot(plt)

    # Transition Heatmap
    st.subheader("Sleep Stage Transition Heatmap")
    heatmap_image = Image.open("numpy_files\transition_matrix_heatmap.png")
    st.image(heatmap_image, caption='Sleep Stage Transition Probabilities', use_column_width=True)

# Statistics Page
elif page == "Statistics":
    st.header("Model Comparison Statistics")

    # Z-Test & T-Test for Model Comparison
    st.subheader("Z-Test between Models")
    cv_scores_rf = np.load('cv_scores.npy')
    cv_scores_xgb = np.load('xgb_cv_scores.npy')
    cv_scores_knn = np.load('knn_cv_scores.npy')

    # Z-test between Random Forest and XGBoost
    z_rf_xgb, p_rf_xgb = z_test(np.mean(cv_scores_rf), np.std(cv_scores_rf), np.mean(cv_scores_xgb), np.std(cv_scores_xgb), 5)
    st.write(f"Random Forest vs XGBoost: Z = {z_rf_xgb:.3f}, p = {p_rf_xgb:.3f}")

    # Z-test between Random Forest and KNN
    z_rf_knn, p_rf_knn = z_test(np.mean(cv_scores_rf), np.std(cv_scores_rf), np.mean(cv_scores_knn), np.std(cv_scores_knn), 5)
    st.write(f"Random Forest vs KNN: Z = {z_rf_knn:.3f}, p = {p_rf_knn:.3f}")

    # Z-test between XGBoost and KNN
    z_xgb_knn, p_xgb_knn = z_test(np.mean(cv_scores_xgb), np.std(cv_scores_xgb), np.mean(cv_scores_knn), np.std(cv_scores_knn), 5)
    st.write(f"XGBoost vs KNN: Z = {z_xgb_knn:.3f}, p = {p_xgb_knn:.3f}")

    # T-Test Results
    st.subheader("T-Test between Models")
    t_stat_rf_xgb, p_value_rf_xgb = stats.ttest_rel(cv_scores_rf, cv_scores_xgb)
    t_stat_rf_knn, p_value_rf_knn = stats.ttest_rel(cv_scores_rf, cv_scores_knn)
    t_stat_xgb_knn, p_value_xgb_knn = stats.ttest_rel(cv_scores_xgb, cv_scores_knn)

    st.write(f"Random Forest vs XGBoost: t = {t_stat_rf_xgb:.3f}, p = {p_value_rf_xgb:.3f}")
    st.write(f"Random Forest vs KNN: t = {t_stat_rf_knn:.3f}, p = {p_value_rf_knn:.3f}")
    st.write(f"XGBoost vs KNN: t = {t_stat_xgb_knn:.3f}, p = {p_value_xgb_knn:.3f}")

# Running the Streamlit app:
# To run this, save this code as `app.py` and run the following in your terminal:
# streamlit run app.py
