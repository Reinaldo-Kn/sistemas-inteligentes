# Bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import numpy as np

# Carregar o dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  
df.columns = ['label', 'text']  
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  

# Visualizar distribuição das classes
sns.countplot(data=df, x='label')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title('Distribuição das Mensagens')
plt.xlabel('Tipo de Mensagem')
plt.ylabel('Quantidade')
plt.show()

# Separar dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Vetorização das mensagens 
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Treinar modelo Random Forest
model = RandomForestClassifier(random_state=42,n_estimators=100,max_depth=None,min_samples_split=20,n_jobs=-1)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

# Avaliar o modelo
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

recall_spam = TP / (TP + FN) 
specificity_ham = TN / (TN + FP) 
gmean = np.sqrt(recall_spam * specificity_ham)  
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Precisão (Precision): {precision:.4f}")
print(f"Acurácia (Accuracy): {accuracy:.4f}")
print(f"G-Mean: {gmean:.4f}")

# Matriz de confusao
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Palavras mais importantes em Spam e Ham

feature_names = vectorizer.get_feature_names_out()
X_all_tfidf = vectorizer.transform(df['text'])
spam_indices = (df['label'] == 1).to_numpy().nonzero()[0]
ham_indices = (df['label'] == 0).to_numpy().nonzero()[0]

spam_tfidf = X_all_tfidf[spam_indices]
ham_tfidf = X_all_tfidf[ham_indices]

spam_means = np.asarray(spam_tfidf.mean(axis=0)).flatten()
ham_means = np.asarray(ham_tfidf.mean(axis=0)).flatten()

df_spam_words = pd.DataFrame({'word': feature_names, 'importance': spam_means})
df_ham_words = pd.DataFrame({'word': feature_names, 'importance': ham_means})

# Selecionar top 20 palavras
top_spam = df_spam_words.sort_values(by='importance', ascending=False).head(20)
top_ham = df_ham_words.sort_values(by='importance', ascending=False).head(20)

# Gráfico das palavras mais importantes no Spam
plt.figure(figsize=(10, 6))
sns.barplot(data=top_spam, y='word', x='importance', color='red')
plt.title('Top 20 Palavras em Mensagens Spam')
plt.xlabel('Importância ')
plt.ylabel('Palavra')
plt.tight_layout()
plt.show()

# Gráfico das palavras mais importantes no Ham
plt.figure(figsize=(10, 6))
sns.barplot(data=top_ham, y='word', x='importance', color='green')
plt.title('Top 20 Palavras em Mensagens Ham')
plt.xlabel('Importância ')
plt.ylabel('Palavra')
plt.tight_layout()
plt.show()

# Testar mensagens novas

custom_messages = [
    # Mensagens de spam
    "free entry into our contest! text win to 80085 now!",
    "you’ve been selected for a $500 walmart gift card. click to claim.",
    "congratulations! you've won a free ticket to the bahamas. call now!",
    "urgent! you have won a 1,000,000 prize. reply yes to claim.",
    "win a brand new car! click the link below now!!!",
    "you are chosen for an exclusive deal. limited time only!",
    "earn money from home with this simple trick. limited slots!",
    "claim your free trial now. no credit card required!",
    "act now! this offer expires in 12 hours!",
    "you've been selected for a financial grant of $2000. click to apply!",

    # Mensagens ham
    "hey, can you pick me up after work today?",
    "don't forget to bring your notebook to class.",
    "are we still meeting for lunch today?",
    "i'll be late, caught in traffic.",
    "hey, i’ll call you back after the meeting.",
    "sure, see you at 6!",
    "thanks for the update. i'll check it out.",
    "do you need help with your homework?",
    "let's grab coffee tomorrow at 10.",
    "that sounds good. talk later!"
]

# Vetorizar as mensagens novas
custom_tfidf = vectorizer.transform(custom_messages)

# Fazer a previsao
custom_preds = model.predict(custom_tfidf)

# Mostrar os resultados
print("\n== Previsões para novas mensagens ==")
for msg, pred in zip(custom_messages, custom_preds):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}] {msg}")
