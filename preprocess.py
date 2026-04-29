import pandas as pd

# Step 1: Load the datasets
print("Loading datasets...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Step 2: Add labels
# 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

# Step 3: Combine both into one dataset
df = pd.concat([fake_df, true_df], ignore_index=True)

# Step 4: Keep only the columns we need
df = df[["title", "text", "label"]]

# Step 5: Combine title and text into one column
df["content"] = df["title"] + " " + df["text"]

# Step 6: Drop rows with missing values
df = df.dropna()

# Step 7: Shuffle the dataset randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 8: Preview the data
print("\nDataset shape:", df.shape)
print("\nFirst 3 rows:")
print(df[["content", "label"]].head(3))
print("\nLabel counts:")
print(df["label"].value_counts())
print("\nPreprocessing done!")

# Step 9: Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
print("Saved cleaned_data.csv")