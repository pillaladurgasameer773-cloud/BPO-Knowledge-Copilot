import pandas as pd

# Load knowledge base
df = pd.read_csv("../data/BPO_Knowledge_Copilot_Gold.csv")


def keyword_search(query):
    query = query.lower()

    results = df[
        df["title"].str.lower().str.contains(query, na=False)
        | df["scenario"].str.lower().str.contains(query, na=False)
        | df["procedure"].str.lower().str.contains(query, na=False)
        | df["aliases"].str.lower().str.contains(query, na=False)
    ]

    return results


# Test question
question = "customer says claim was rejected but they had coverage"

results = keyword_search(question)

print("Search question:")
print(question)

print("\nResults:")
print(results[["article_id", "title", "category"]])