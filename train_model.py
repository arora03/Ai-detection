import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re
import warnings

# Suppress warnings from scikit-learn
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --------------------------
# 1. Text Cleaning Function
# --------------------------
def clean_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep only letters, numbers, space
    return text.strip()

# --------------------------
# 2. Load and Prepare ALL Four Datasets
# --------------------------
print("Loading datasets...")
all_dfs = []

try:
    # --- Dataset 1: Student Essays ---
    df_essays = pd.read_csv("data/train_drcat_01.csv")
    df_essays = df_essays[['text', 'label']]
    all_dfs.append(df_essays)
    print(f"Loaded {len(df_essays)} student essays.")

    # --- Dataset 2: Research Abstracts (AI vs Human) ---
    df_abstracts = pd.read_csv("data/GPT-vs-Human-Abstracts.csv")
    df_abstracts.rename(columns={'abstract': 'text', 'is_ai_generated': 'label'}, inplace=True)
    df_abstracts = df_abstracts[['text', 'label']]
    all_dfs.append(df_abstracts)
    print(f"Loaded {len(df_abstracts)} AI/Human research abstracts.")

    # --- Dataset 3: Blogs, News, Tech ---
    df_diverse = pd.read_csv("data/ai_vs_human_text.csv")
    df_diverse['label'] = df_diverse['label'].apply(lambda x: 1 if str(x).lower() == 'ai' else 0)
    df_diverse = df_diverse[['text', 'label']]
    all_dfs.append(df_diverse)
    print(f"Loaded {len(df_diverse)} diverse texts (blogs, news, etc.).")

    # --- Dataset 4: MORE Human Research Abstracts ---
    df_arxiv = pd.read_csv("data/arxiv_data.csv")
    df_arxiv.rename(columns={'abstract': 'text'}, inplace=True)
    df_arxiv['label'] = 0  # We know these are all human-written
    df_arxiv = df_arxiv[['text', 'label']]
    all_dfs.append(df_arxiv)
    print(f"Loaded {len(df_arxiv)} new HUMAN research abstracts.")

except FileNotFoundError as e:
    print(f"\n{'='*50}\nERROR: Could not find a dataset file!")
    print(f"Missing file: {e.filename}")
    print("Please make sure all four CSV files are in your 'data/' folder.")
    print(f"{'='*50}\n")
    exit()

# --------------------------
# 3. Combine and Clean
# --------------------------
print("Combining all datasets and cleaning text...")
df_master = pd.concat(all_dfs, ignore_index=True)

df_master.dropna(inplace=True) # Remove any empty rows
df_master.drop_duplicates(subset=['text'], inplace=True) # Drop duplicate texts
df_master['text'] = df_master['text'].apply(clean_text)

# We have lots of data, let's take a good sample
# Using 50,000 samples to keep training fast, but get a good mix
if len(df_master) > 50000:
    df_master = df_master.sample(50000, random_state=42)

print(f"Total unique samples for training: {len(df_master)}")
print("Balancing data (if needed)...")
# Check the balance
label_counts = df_master['label'].value_counts()
print(label_counts)

# Simple undersampling if data is very unbalanced (optional but good practice)
min_label_count = label_counts.min()
if min_label_count < len(df_master) * 0.1: # If one class is less than 10%
    min_label_count = int(min_label_count * 1.5) # Let's keep it slightly unbalanced

df_balanced = pd.concat([
    df_master[df_master['label'] == 0].sample(min_label_count, random_state=42, replace=True),
    df_master[df_master['label'] == 1].sample(min_label_count, random_state=42, replace=True)
])

print(f"Training on a balanced set of {len(df_balanced)} samples.")
X = df_balanced['text']
y = df_balanced['label']

# --------------------------
# 4. Split and Vectorize
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

print(f"Vectorizing {len(X_train)} text samples...")

vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=10000,  # Increased features for a more complex dataset
    ngram_range=(1, 2)   # Look at single words AND two-word phrases
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------
# 5. Train Model
# --------------------------
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --------------------------
# 6. Evaluate
# --------------------------
print("Evaluating new 'generalist' model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n" + "="*30)
print(f"✅ New Model Accuracy: {accuracy:.2f} %")
print("="*30)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------------
# 7. Save Model
# --------------------------
joblib.dump(model, "model/trained_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl") # <-- IMPORTANT: Save the vectorizer
print("🎉 New 'generalist' model training complete and saved!")