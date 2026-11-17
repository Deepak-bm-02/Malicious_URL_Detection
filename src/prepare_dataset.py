
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find CSV file
csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "CSV file not found inside dataset/raw folder!"
    )

csv_path = csv_files[0]

print("Loading:", csv_path.name)

# Read CSV
df = pd.read_csv(csv_path, low_memory=False)

print("\nOriginal dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# Identify required columns
url_col = next(
    (c for c in df.columns if c.strip().lower() == "url"),
    None
)

label_col = next(
    (c for c in df.columns if c.strip().lower() == "classlabel"),
    None
)

if url_col is None or label_col is None:
    raise ValueError(
        "URL or ClassLabel column missing. Check the column names."
    )

# Select URL and label
data = df[[url_col, label_col]].copy()
data.columns = ["url", "original_label"]

# Clean records
data.dropna(inplace=True)
data["url"] = data["url"].astype(str).str.strip()
data["original_label"] = pd.to_numeric(
    data["original_label"], errors="coerce"
)

data.dropna(inplace=True)
data = data[data["url"] != ""]

# Convert labels:
# Original: 0 = Phishing, 1 = Legitimate
# Project:  0 = Legitimate, 1 = Malicious

data["label"] = data["original_label"].map({
    1: 0,
    0: 1
})

data.dropna(subset=["label"], inplace=True)

# Remove duplicate URLs
data.drop_duplicates(subset=["url"], inplace=True)

data["label"] = data["label"].astype(int)

# Final dataset
final_data = data[["url", "label"]]

# Shuffle
final_data = final_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
output_file = OUTPUT_DIR / "cleaned_urls.csv"
final_data.to_csv(output_file, index=False)

# Display summary
print("\n========== DATASET SUMMARY ==========")

print("Total URLs:", len(final_data))

print("\nClassification distribution:")

print(
    final_data["label"].value_counts().rename(
        index={
            0: "Legitimate",
            1: "Malicious"
        }
    )
)

print("\nSample URLs:")
print(final_data.head(10).to_string(index=False))

print("\nDataset preparation completed!")
print("Saved at:", output_file)
