# best.py

import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Carregao dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vetorização 
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#  Define grade de hiperparametros
param_grid = {
    'n_estimators': [5, 10,50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5,10,20],
}

# Combinacoes
combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split']
))

# Testa cada uma
results = []

for n_estimators, max_depth, min_samples_split in combinations:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    recall_spam = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity_ham = TN / (TN + FP) if (TN + FP) > 0 else 0
    gmean = np.sqrt(recall_spam * specificity_ham)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'accuracy': accuracy,
        'precision': precision,
        'gmean': gmean
    })

# Ordenar e mostrar 
df_results = pd.DataFrame(results)
best = df_results.sort_values(by='gmean', ascending=False).head(5)

print("Top 5 melhores por gmean:")
print(best.to_string(index=False))
