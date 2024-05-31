import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Svenska stoppord lista
swedish_stopwords = [
    "och", "i", "att", "det", "som", "en", "på", "är", "av", "för", "med", "till", "den",
    "har", "de", "inte", "om", "ett", "men", "var", "sig", "så", "vi", "kan", "man", "hade",
    "där", "eller", "vad", "alla", "kommer", "vilka", "fram", "sådana", "också", "in", "kan",
    "om", "bara", "hur", "därför", "än", "någon", "finns", "mer", "mycket", "några", "här", "blir",
    "gå", "något", "de", "ett", "där", "deras", "dessa", "utan", "varit", "vilket", "sina",
    "hos", "själv", "denna", "då", "sådan", "under", "även", "ja", "nej", "ja", "tack"
]

# Hitta CSV-filer i mappen
csv_files = [file for file in os.listdir("/home/monasaffari/Desktop/Hello") if file.endswith('.csv')]

# Kontrollera om filer finns och ladda data
if csv_files:
    data_frames = []
    for file in csv_files:
        file_path = os.path.join("/home/appuser/Desktop/Hello", file)
        data_frames.append(pd.read_csv(file_path, sep=";", names=[
            "Id", "Headline", "Application_deadline", "Amount", "Description", 
            "Type", "Salary", "Duration", "Working_hours", "Region", "Municipality", 
            "Employer_name", "Employer_workplace", "Publication_date"
        ]))
    
    # Slå samman dataframes om det finns några
    if data_frames:
        data = pd.concat(data_frames, ignore_index=True)
        
        # Skapa etiketter baserat på nyckelord
        keywords = ['sjuksköterska', 'patient']
        data['label'] = data['Description'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in keywords) else 0)

        # Dela upp data i tränings- och testmängd
        X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['label'], test_size=0.2, random_state=42)

        # Omvandla textdata till tf-idf funktioner med justerade parametrar
        vectorizer = TfidfVectorizer(stop_words=swedish_stopwords, max_df=0.95, min_df=1)
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
    else:
        print("Inga CSV-filer hittades i mappen.")
else:
    print("Inga CSV-filer hittades i mappen.")

