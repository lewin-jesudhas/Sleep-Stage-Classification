#All labels
'''
import numpy as np
all_wavelet_features = np.load('wavelet_features.npy')
all_labels = np.load('wavelet_labels.npy')
print(all_labels)
'''


#Random Forrest
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Load results
accuracy = np.load('accuracy.npy')[0]  # Single value saved as an array
classification_report_data = np.load('classification_report.npy', allow_pickle=True).item()
cv_scores = np.load('cv_scores.npy')
confusion_matrix = np.load('confusion_matrix.npy')

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
for label, metrics in classification_report_data.items():
    print(f"{label}: {metrics}")

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.2f}")
all_labels = np.load('wavelet_labels.npy')
# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(all_labels)).plot()
plt.title("Confusion Matrix")
plt.show()
'''

#Shape
'''
print("Features shape:", all_wavelet_features.shape)
print("Labels shape:", all_labels.shape)
'''


#XGBoost
'''
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load results
accuracy = np.load('xgb_accuracy.npy')[0]  # Single value saved as an array
classification_report_data = np.load('xgb_classification_report.npy', allow_pickle=True).item()
cv_scores = np.load('xgb_cv_scores.npy')
confusion_matrix = np.load('xgb_confusion_matrix.npy')

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
for label, metrics in classification_report_data.items():
    if isinstance(metrics, dict):  # Print only metric dictionaries
        print(f"{label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.2f}")
all_labels=np.load("wavelet_labels.npy")
# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(all_labels)).plot()
plt.title("Confusion Matrix")
plt.show()
'''

#KNN
'''
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load results
accuracy = np.load('knn_accuracy.npy')[0]  # Single value saved as an array
classification_report_data = np.load('knn_classification_report.npy', allow_pickle=True).item()
cv_scores = np.load('knn_cv_scores.npy')
confusion_matrix = np.load('knn_confusion_matrix.npy')

# Display results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
for label, metrics in classification_report_data.items():
    if isinstance(metrics, dict):  # Print only metric dictionaries
        print(f"{label}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.2f}")
all_labels=np.load("wavelet_labels.npy")
# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=np.unique(all_labels)).plot()
plt.title("Confusion Matrix")
plt.show()
'''

#Z-test
'''
import numpy as np
from scipy import stats

# Load CV scores for Random Forest, XGBoost, and k-NN
cv_scores_rf = np.load('cv_scores.npy')
cv_scores_xgb = np.load('xgb_cv_scores.npy')
cv_scores_knn = np.load('knn_cv_scores.npy')
# Calculate mean and standard deviation for each model
mean_rf, std_rf = np.mean(cv_scores_rf), np.std(cv_scores_rf, ddof=1)
mean_xgb, std_xgb = np.mean(cv_scores_xgb), np.std(cv_scores_xgb, ddof=1)
mean_knn, std_knn = np.mean(cv_scores_knn), np.std(cv_scores_knn, ddof=1)
# Define Z-test function
def z_test(mean1, std1, mean2, std2, n):
    z_score = (mean1 - mean2) / np.sqrt((std1**2 + std2**2) / n)
    p_value = stats.norm.sf(abs(z_score)) * 2  # Two-tailed test
    return z_score, p_value

n = 5  # Number of cross-validation folds

# Z-test between Random Forest and XGBoost
z_rf_xgb, p_rf_xgb = z_test(mean_rf, std_rf, mean_xgb, std_xgb, n)
print(f"Random Forest vs XGBoost: Z = {z_rf_xgb:.3f}, p = {p_rf_xgb:.3f}")

# Z-test between Random Forest and KNN
z_rf_knn, p_rf_knn = z_test(mean_rf, std_rf, mean_knn, std_knn, n)
print(f"Random Forest vs KNN: Z = {z_rf_knn:.3f}, p = {p_rf_knn:.3f}")

# Z-test between XGBoost and KNN
z_xgb_knn, p_xgb_knn = z_test(mean_xgb, std_xgb, mean_knn, std_knn, n)
print(f"XGBoost vs KNN: Z = {z_xgb_knn:.3f}, p = {p_xgb_knn:.3f}")
'''


#t-test
'''
import numpy as np
from scipy import stats

# Load CV scores for Random Forest, XGBoost, and k-NN
cv_scores_rf = np.load('cv_scores.npy')
cv_scores_xgb = np.load('xgb_cv_scores.npy')
cv_scores_knn = np.load('knn_cv_scores.npy')
# Paired t-test between Random Forest and XGBoost
t_stat_rf_xgb, p_value_rf_xgb = stats.ttest_rel(cv_scores_rf, cv_scores_xgb)
print(f"Random Forest vs XGBoost: t = {t_stat_rf_xgb:.3f}, p = {p_value_rf_xgb:.3f}")

# Paired t-test between Random Forest and KNN
t_stat_rf_knn, p_value_rf_knn = stats.ttest_rel(cv_scores_rf, cv_scores_knn)
print(f"Random Forest vs KNN: t = {t_stat_rf_knn:.3f}, p = {p_value_rf_knn:.3f}")

# Paired t-test between XGBoost and KNN
t_stat_xgb_knn, p_value_xgb_knn = stats.ttest_rel(cv_scores_xgb, cv_scores_knn)
print(f"XGBoost vs KNN: t = {t_stat_xgb_knn:.3f}, p = {p_value_xgb_knn:.3f}")
'''


#Singular Chart
'''
import numpy as np
import pandas as pd
from PIL import Image
chart_image = Image.open('sleep_stage_durations_chart.npy')
chart_image.show()

# Reload DataFrame stats
df_stats = np.load('annotations_df_stats.npy', allow_pickle=True).item()
print("DataFrame Info:", df_stats['info'])
print("DataFrame Describe:", pd.DataFrame(df_stats['describe']))
print("DataFrame Null Counts:", df_stats['isnull'])
print("DataFrame Shape:", df_stats['shape'])
'''

#Sleep stage barchart
'''
import numpy as np
import matplotlib.pyplot as plt

# Load sleep stage durations from the saved .npy file
sleep_stage_durations = np.load("sleep_stage_durations_full.npy", allow_pickle=True).item()

# Plot sleep stage distribution
plt.figure(figsize=(10, 6))
plt.bar(sleep_stage_durations.keys(), sleep_stage_durations.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
plt.xlabel('Sleep Stages')
plt.ylabel('Total Duration (minutes)')
plt.title('Sleep Stage Distribution Over Time')
plt.show()

#Piechart
import numpy as np
import matplotlib.pyplot as plt

# Load sleep stage durations from the saved .npy file
sleep_stage_durations = np.load("sleep_stage_durations_full.npy", allow_pickle=True).item()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    sleep_stage_durations.values(),
    labels=sleep_stage_durations.keys(),
    autopct='%1.1f%%',
    startangle=90,
    colors=['blue', 'orange', 'green', 'red', 'purple']
)
plt.title('Sleep Stage Distribution (Percentage)')
plt.show()
'''

#Sleep Transition
'''
from PIL import Image
import matplotlib.pyplot as plt

# Path to the saved heatmap image
heatmap_path = "transition_matrix_heatmap.png"

# Load the image
heatmap_image = Image.open(heatmap_path)

# Display the image using Matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(heatmap_image)
plt.axis('off')  # Turn off the axis
plt.title('Sleep Stage Transition Probabilities Heatmap')
plt.show()
'''

# To load and display the saved image
'''
from PIL import Image
graph_path = "sleep_stage_transition_graph.png"
graph_image = Image.open(graph_path)
graph_image.show()
'''
