# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your dataset (replace 'imdb.csv' with your file)
# Dataset should have 'review' and 'sentiment' columns
df = pd.read_csv('IMDB Dataset.csv')

# Optional: quick look at the data
print(df.head())

# 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 3. Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------
# 4A. Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# ---------------------------
# 4B. Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ---------------------------
# 5. Visualize top positive/negative words from LR
feature_names = vectorizer.get_feature_names_out()
coefs = lr_model.coef_[0]

# Top positive words
top_positive = sorted(zip(coefs, feature_names), reverse=True)[:20]
# Top negative words
top_negative = sorted(zip(coefs, feature_names))[:20]

plt.figure(figsize=(12,6))
sns.barplot(x=[x[0] for x in top_positive], y=[x[1] for x in top_positive], color='green')
plt.title("Top Positive Words")
plt.show()

plt.figure(figsize=(12,6))
sns.barplot(x=[x[0] for x in top_negative], y=[x[1] for x in top_negative], color='red')
plt.title("Top Negative Words")
plt.show()
