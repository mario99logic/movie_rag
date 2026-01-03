"""Script to download and preprocess movies dataset from Kaggle.
Dataset contains 10000, this script downloads the dataset and filters to top 100 rated movies.

Dataset source:
https://www.kaggle.com/datasets/mohsin31202/top-rated-movies-dataset
"""

import kagglehub
import pandas as pd
import os

print("Downloading movies dataset from Kaggle...")

# Download the full movies dataset
path = kagglehub.dataset_download("mohsin31202/top-rated-movies-dataset")
print(f"Dataset downloaded to: {path}")

# Load the full dataset
csv_file = os.path.join(path, "Movies_dataset.csv")
df = pd.read_csv(csv_file)

print(f"\nOriginal dataset: {len(df)} movies")

# Filter: Take top 100 movies by rating with minimum votes for quality
df_filtered = (
    df[df["vote_count"] >= 1000].sort_values("vote_average", ascending=False).head(100)
)

print("Filtered to top 100 rated movies (vote_count >= 1000)")
print(f"Highest rating: {df_filtered['vote_average'].max()}")
print(f"Lowest rating: {df_filtered['vote_average'].min()}")

# Save to project data folder
project_root = os.path.dirname(os.path.dirname(__file__))
data_folder = os.path.join(project_root, "data")

# Ensure data folder exists
os.makedirs(data_folder, exist_ok=True)

output_path = os.path.join(data_folder, "top_movies.csv")
df_filtered.to_csv(output_path, index=False)

print(f"\n✓ Saved top 100 movies to: {output_path}")
