import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample data for suicidal vs non-suicidal
texts = [
    "I want to end my life", "I feel hopeless and want to die",
    "Kill myself", "I am so depressed want to suicide",
    "I am happy today", "I love my life", "Feeling great",
    "Life is beautiful", "I am excited for tomorrow"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=suicidal, 0=non-suicidal

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

# Save both
data = {'vectorizer': vectorizer, 'classifier': model}
with open('model.pkl', 'wb') as f:
    pickle.dump(data, f)

print("model.pkl created successfully!")