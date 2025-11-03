import pandas as pd
import joblib
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --------------------------
# 1. Text Cleaning Function
# --------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text)  # keep punctuation
    return text.strip()

# --------------------------
# 2. Load and Prepare ALL Four Datasets
# --------------------------
print("Loading datasets...")
all_dfs = []

try:
    df_essays = pd.read_csv("data/train_drcat_01.csv")[['text', 'label']]
    all_dfs.append(df_essays)
    print(f"Loaded {len(df_essays)} student essays.")

    df_abstracts = pd.read_csv("data/GPT-vs-Human-Abstracts.csv")
    df_abstracts.rename(columns={'abstract': 'text', 'is_ai_generated': 'label'}, inplace=True)
    df_abstracts = df_abstracts[['text', 'label']]
    all_dfs.append(df_abstracts)
    print(f"Loaded {len(df_abstracts)} AI/Human research abstracts.")

    df_diverse = pd.read_csv("data/ai_vs_human_text.csv")
    df_diverse['label'] = df_diverse['label'].apply(lambda x: 1 if str(x).lower() == 'ai' else 0)
    df_diverse = df_diverse[['text', 'label']]
    all_dfs.append(df_diverse)
    print(f"Loaded {len(df_diverse)} diverse texts (blogs, news, etc.).")

    df_arxiv = pd.read_csv("data/arxiv_data.csv")
    df_arxiv.rename(columns={'abstract': 'text'}, inplace=True)
    df_arxiv['label'] = 0
    df_arxiv = df_arxiv[['text', 'label']]
    all_dfs.append(df_arxiv)
    print(f"Loaded {len(df_arxiv)} new HUMAN research abstracts.")

except FileNotFoundError as e:
    print(f"\n{'='*50}\nERROR: Could not find dataset file: {e.filename}")
    print("Ensure all four CSVs exist in the 'data/' folder.")
    print(f"{'='*50}\n")
    exit()

# --------------------------
# 3. Combine and Clean
# --------------------------
print("Combining and cleaning data...")
df_master = pd.concat(all_dfs, ignore_index=True)
df_master.dropna(inplace=True)
df_master.drop_duplicates(subset=['text'], inplace=True)
df_master['text'] = df_master['text'].apply(clean_text)

if len(df_master) > 50000:
    df_master = df_master.sample(50000, random_state=42)

print(f"Total unique samples: {len(df_master)}")

label_counts = df_master['label'].value_counts()
print("Label distribution:\n", label_counts)

# Balance dataset
min_label_count = label_counts.min()
df_balanced = pd.concat([
    df_master[df_master['label'] == 0].sample(min_label_count, random_state=42, replace=True),
    df_master[df_master['label'] == 1].sample(min_label_count, random_state=42, replace=True)
])

# --- THIS IS THE FIX ---
# Reset the index to prevent mismatching dimensions
df_balanced.reset_index(drop=True, inplace=True)
# --- END OF FIX ---

print(f"Balanced dataset size: {len(df_balanced)}")

# --------------------------
# 4. Feature Engineering
# --------------------------
print("Extracting stylometric features...")

def avg_word_len(text):
    words = text.split()
    return np.mean([len(w) for w in words]) if words else 0

def punct_ratio(text):
    return sum(c in ".,!?;:" for c in text) / (len(text) + 1)

df_balanced['avg_word_len'] = df_balanced['text'].apply(avg_word_len)
df_balanced['sent_len'] = df_balanced['text'].apply(lambda t: len(t.split()))
df_balanced['punct_ratio'] = df_balanced['text'].apply(punct_ratio)

X = df_balanced['text']
y = df_balanced['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 5. TF-IDF Vectorization (Character-level)
# --------------------------
print("Vectorizing text with character-level TF-IDF...")

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 6),
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Add extra stylometric features
# Use .loc to ensure we get the correct rows based on the index from X_train/X_test
extra_train = csr_matrix(
    df_balanced.loc[X_train.index, ['avg_word_len', 'sent_len', 'punct_ratio']].values
)
extra_test = csr_matrix(
    df_balanced.loc[X_test.index, ['avg_word_len', 'sent_len', 'punct_ratio']].values
)

X_train_combined = hstack([X_train_vec, extra_train])
X_test_combined = hstack([X_test_vec, extra_test])

# --------------------------
# 6. Train Improved Logistic Regression
# --------------------------
print("Training Logistic Regression (C=10)...")
model = LogisticRegression(max_iter=2000, C=10)
model.fit(X_train_combined, y_train)

# --------------------------
# 7. Evaluate
# --------------------------
print("Evaluating model...")
y_pred = model.predict(X_test_combined)
acc = accuracy_score(y_test, y_pred) * 100
print(f"\n{'='*40}")
print(f"✅ Improved Model Accuracy: {acc:.2f}%")
print(f"{'='*40}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 8. Save Model and Vectorizer
# --------------------------
joblib.dump(model, "model/trained_model_v2.pkl")
joblib.dump(vectorizer, "model/vectorizer_v2.pkl")
print("🎉 Model and vectorizer saved to 'model/' folder as v2.")

# --------------------------
# 9. Example Usage
# --------------------------
print("\nRunning example test...")
sample_text = """Artificial intelligence continuously improves automation through structured data optimization and predictive computation."""
sample_clean = clean_text(sample_text)

vec = vectorizer.transform([sample_clean])
extra = csr_matrix([[avg_word_len(sample_clean), len(sample_clean.split()), punct_ratio(sample_clean)]])
X_sample = hstack([vec, extra])

prob = model.predict_proba(X_sample)[0][1]
print(f"🔎 Example text AI probability: {prob*100:.2f}%")