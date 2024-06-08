import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the data
file_path = '/Users/muthusupriya/Documents/Hareni/sem4/BIO/DATASET/combined_features_with_dna_shapes_imputed.csv'
data = pd.read_csv(file_path)

# Clean up column names
data.columns = data.columns.str.strip()

# Check if 'Class' is correctly formatted and exists
print('Class' in data.columns)  # This should print True if 'Class' is correctly formatted

# Convert 'Class' column to numeric for correlation and PCA, but keep a categorical version for plotting
label_encoder = LabelEncoder()
data['NumericClass'] = label_encoder.fit_transform(data['Class'])
data['Class'] = data['NumericClass'].astype('category')  # Explicitly convert to categorical

# Correlation heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Attempt a pair plot with explicitly categorical 'Class'
try:
    # Using only numeric columns for pairplot to avoid any non-numeric issues
    numeric_cols = data.select_dtypes(include=[float, int]).columns
    sns.pairplot(data[numeric_cols.tolist()[:4] + ['Class']], hue='Class')  # Using first 4 numeric features and 'Class'
    plt.suptitle('Pair Plot of Selected Numeric Features', y=1.02)
    plt.show()
except Exception as e:
    print(f"Error in creating pairplot: {e}")

# Box plots for a few features
features_to_plot = ['aaa', 'aac', 'aag', 'aat', 'aca']
for feature in features_to_plot:
    sns.boxplot(x='Class', y=feature, data=data)
    plt.title(f'Box Plot of {feature}')
    plt.show()

# PCA plot using numeric class
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, :-2])  # Exclude the last two columns (categorical and numeric class)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_scaled)
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
pca_df['Class'] = data['Class']
sns.scatterplot(x='PCA1', y='PCA2', hue='Class', data=pca_df)
plt.title('PCA Plot')
plt.show()
