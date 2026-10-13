import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1. Loading directly using kagglehub
# The filename inside this dataset is 'Sleep_health_and_lifestyle_dataset.csv'
file_path = "Sleep_health_and_lifestyle_dataset.csv"

# Load the dataset (or use kagglehub.dataset_load in newer versions)
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uom190346a/sleep-health-and-lifestyle-dataset",
    file_path,
)

print("--- First 5 Records ---")
print(df.head())

print("\n--- Summary Info ---")
df.info()

print("\n--- Stress Level Distribution ---")
print(df["Stress Level"].value_counts().sort_index())
