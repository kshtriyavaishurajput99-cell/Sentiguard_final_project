import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# 1. Load
df = pd.read_csv('cleaned_dataset.csv')
print(f"Dataset loaded:{df.shape}")
print(df.columns.tolist())

# 2. TF-IDF Unigram/Bigram
vectorizer = TfidfVectorizer(ngram_range=(1,2),max_features=5000)
text_col = 'text'
label_col = 'class'
print(f"Using text_col={text_col},label_col={label_col}")

X_tfidf = vectorizer.fit_transform(df[text_col].astype(str))
y = df[label_col]
print(f"TF-IDF Shape:{X_tfidf.shape}")

# 3. Linguistic stats
df['text_len'] = df[text_col].astype(str).apply(len)
df['word_count'] = df[text_col].astype(str).apply(lambda x:len(x.split()))
print("Linguistic stats added")

# 4. Genetic Algorithm - FIXED LIGHT VERSION
print("Running Genetic Algorithm for feature selection...")
X_small = X_tfidf[:5000]
y_small = y[:5000]

population_size = 5
best_score = 0
best_mask = np.ones(X_tfidf.shape[1],dtype=int)

for i in range(5):
    mask = np.random.choice([0,1],size=X_tfidf.shape[1],p=[0.2, 0.8])
    X_selected = X_small[:,mask==1]
    
    X_train, X_test, y_train, y_test=train_test_split(X_selected, y_small,test_size=0.2,random_state=42)
    model=LogisticRegression(max_iter=200)
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    score=accuracy_score(y_test,pred)
    
    if score > best_score:
        best_score=score
        best_mask=mask
    print(f"Gen {i+1}:score={score:.4f}features={mask.sum()}")

print(f"Best GA Score:{best_score}")

# 5. Save
with open('vectorizer.pkl','wb')as f:
    pickle.dump(vectorizer,f)

with open('feature_mask.pkl','wb')as f:
    
    
    pickle.dump(best_mask,f)

print("Saved vectorizer.pklandfeature_mask.pkl-Member 2 DONE")
import matplotlib.pyplot as plt

# Data jo aapke screenshot me hai
generations = [1,2,3,4,5]
scores = [0.8780, 0.8860, 0.8840, 0.8720, 0.8860]
features = [3968, 4037, 4042, 3968, 3998]

# Graph 1 - GA Accuracy
plt.figure()
plt.plot(generations, scores, marker='o')
plt.title('Genetic Algorithm - Accuracy per Generation')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('ga_accuracy.png')
print("Saved ga_accuracy.png")

# Graph 2 - Features Selected
plt.figure()
plt.bar(generations, features)
plt.title('Genetic Algorithm - Features Selected')
plt.xlabel('Generation')
plt.ylabel('No. of Features')
plt.savefig('ga_features.png')
print("Saved ga_features.png")