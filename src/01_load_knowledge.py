import pandas as pd

# Load the knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")

print("First 5 articles:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nArticles by category:")
print(df["category"].value_counts())

# Check article text length

df["text_length"] = (
    df["title"].str.len()
    + df["scenario"].str.len()
    + df["procedure"].str.len()
    + df["escalation"].str.len()
)

print("\nArticle text length:")
print(df["text_length"].describe())