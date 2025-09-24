import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------
# 1. Load Dataset
# ------------------------
df = pd.read_csv("IMDB Dataset.csv")

# ------------------------
# 2. Preprocessing
# ------------------------
def clean_text(text):
    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # lowercase
    text = text.lower()
    return text

df["review"] = df["review"].apply(clean_text)

X = df["review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})  # convert to 1/0

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# 3. Vectorization
# ------------------------
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------
# 4. Train Models
# ------------------------

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_vec, y_train)
y_pred_lr = log_reg.predict(X_test_vec)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
y_pred_nb = nb.predict(X_test_vec)
acc_nb = accuracy_score(y_test, y_pred_nb)

# ------------------------
# 5. Results
# ------------------------
print("=== Model Accuracies on Test Set ===")
print(f"Logistic Regression: {acc_lr:.4f}")
print(f"Naive Bayes:        {acc_nb:.4f}")
