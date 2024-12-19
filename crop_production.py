import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('Crop_Production_data.csv')

# Exploratory Data Analysis (EDA)
# Plotting the distribution of Production
plt.figure(figsize=(10, 6))
sns.histplot(data['Production'], bins=50, kde=True, color='blue')
plt.title('Distribution of Production')
plt.xlabel('Production (Metric Tons)')
plt.ylabel('Frequency')
plt.show()

# Crop production trends over the years
plt.figure(figsize=(12, 6))
production_by_year = data.groupby('Crop_Year')['Production'].sum()
sns.lineplot(x=production_by_year.index, y=production_by_year.values, marker='o', color='green')
plt.title('Total Crop Production Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Production (Metric Tons)')
plt.grid(True)
plt.show()

# Crop production by season
plt.figure(figsize=(10, 6))
seasonal_production = data.groupby('Season')['Production'].sum().sort_values()
seasonal_production.plot(kind='bar', color='orange')
plt.title('Total Production by Season')
plt.xlabel('Season')
plt.ylabel('Total Production (Metric Tons)')
plt.show()

# Data Cleaning
# Fill missing Production values with median for each Crop and Season combination
data['Production'] = data.groupby(['Crop', 'Season'])['Production'].transform(
    lambda x: x.fillna(x.median())
)

# Feature Engineering
# Calculate production per unit area as a new feature
data['Production_per_Unit_Area'] = data['Production'] / data['Area']

# Clustering Analysis
# Prepare data for clustering
clustering_data = data.groupby(['District_Name', 'Crop'])[['Production']].sum().unstack(fill_value=0)
clustering_data.columns = clustering_data.columns.droplevel()
clustering_data = clustering_data.reset_index().set_index('District_Name')

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(clustering_data)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Add cluster labels to the clustering data
clustering_data['Cluster'] = clusters

# Visualize the cluster size distribution
plt.figure(figsize=(8, 5))
sns.countplot(x=clusters, palette='viridis')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Districts')
plt.show()

# Save cleaned data and clustering results for future use
clustering_data.to_csv('Clustering_Results.csv')
data.to_csv('Cleaned_Crop_Production_Data.csv')

# End of script
