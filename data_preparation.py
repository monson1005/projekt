import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Svenska stoppord lista
swedish_stopwords = [
    "och", "i", "att", "det", "som", "en", "på", "är", "av", "för", "med", "till", "den",
    "har", "de", "inte", "om", "ett", "men", "var", "sig", "så", "vi", "kan", "man", "hade",
    "där", "eller", "vad", "alla", "kommer", "vilka", "fram", "sådana", "också", "in", "kan",
    "om", "bara", "hur", "därför", "än", "någon", "finns", "mer", "mycket", "några", "här", "blir",
    "gå", "något", "de", "ett", "där", "deras", "dessa", "utan", "varit", "vilket", "sina",
    "hos", "själv", "denna", "då", "sådan", "under", "även", "ja", "nej", "ja", "tack"
]

# Ladda data
data = pd.read_csv("2023.csv", sep=";", names=[
    "Id", "Headline", "Application_deadline", "Amount", "Description", 
    "Type", "Salary", "Duration", "Working_hours", "Region", "Municipality", 
    "Employer_name", "Employer_workplace", "Publication_date"
])

# Skapa etiketter baserat på nyckelord
keywords = ['sjuksköterska', 'patient']
data['label'] = data['Description'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in keywords) else 0)

# Kontrollera fördelningen av etiketter
print(data['label'].value_counts())

# Om det inte finns några positiva exempel, avbryt och meddela användaren
if data['label'].value_counts().get(1, 0) == 0:
    raise ValueError(f"No positive examples found for the keywords: {keywords}")

# Dela upp data i tränings- och testmängd
X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['label'], test_size=0.2, random_state=42)

# Omvandla textdata till tf-idf funktioner med justerade parametrar
vectorizer = TfidfVectorizer(stop_words=swedish_stopwords, max_df=0.95, min_df=0.01)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Träna en klassificeringsmodell
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Utvärdera modellen
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Spara modellen och vectorizern
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

print("Modellen och vectorizern är tränade och sparade.")

# Visa de mest viktiga orden
import numpy as np
feature_names = np.array(vectorizer.get_feature_names_out())
sorted_coef_index = model.coef_[0].argsort()
print("De mest viktiga orden för positiv klass:")
print(feature_names[sorted_coef_index[:-11:-1]])
print("De mest viktiga orden för negativ klass:")
print(feature_names[sorted_coef_index[:10]])
