import pandas as pd
import re
from collections import Counter
import streamlit as st
from openai import OpenAI
import config
import os
import joblib

st.title("Nurse Bot 👩‍⚕️")

# Ange den fullständiga sökvägen till CSV-filen
csv_file_path = os.path.join(os.path.dirname(__file__), "2023.csv")

# Läs in data från CSV-filen med rätt separator och specifiera kolumnnamn
data = pd.read_csv(csv_file_path, sep=";", names=[
    "Id", "Headline", "Application_deadline", "Amount", "Description", 
    "Type", "Salary", "Duration", "Working_hours", "Region", "Municipality", 
    "Employer_name", "Employer_workplace", "Publication_date"
])

# Ladda klassificeringsmodellen och vectorizern
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Funktion för att extrahera sjukskötersketyp från Headline
def extract_nurse_type(headline):
    match = re.search(r'\b(sjuksköterska|barnmorska|anestesisjuksköterska|operationssjuksköterska|röntgensjuksköterska|intensivvårdssjuksköterska|distriktssköterska|psykiatrisjuksköterska)\b', headline.lower())
    return match.group(0) if match else None

data['Nurse_type'] = data['Headline'].apply(extract_nurse_type)
data = data.dropna(subset=['Nurse_type'])

nurse_type_counts = Counter(data['Nurse_type'])
common_nurse_types = [nurse for nurse, count in nurse_type_counts.items() if count > 5]

data['Municipality'] = data['Municipality'].str.lower()
data['Working_hours'] = data['Working_hours'].astype(str).str.lower()
data['Type'] = data['Type'].str.lower()

def get_municipality_choice(user_input):
    municipalities = data['Municipality'].unique()
    if user_input.lower() in municipalities:
        return user_input.lower()
    else:
        return None

def get_nurse_type_choice(user_input):
    if user_input.lower() in common_nurse_types:
        return user_input.lower()
    else:
        return None

def get_working_hours_choice(user_input):
    working_hours_types = data['Working_hours'].dropna().unique()
    if user_input.lower() in working_hours_types:
        return user_input.lower()
    else:
        return None

def filter_jobs_by_keyword(keyword):
    keyword_tfidf = vectorizer.transform(data['Description'].astype(str))
    predictions = model.predict(keyword_tfidf)
    return data[predictions == 1]

client = OpenAI(api_key=config.OPENAI_API_KEY)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if not st.session_state.messages:
    initial_welcome = "Välkommen till Nurse Bot! Vi hjälper dig att hitta ett jobb som passar dig som sjuksköterska. Svara på frågorna nedan för att börja!"
    st.session_state.messages.append({"role": "assistant", "content": initial_welcome})

    first_question = "Vänligen ange vilken stad du vill jobba i:"
    st.session_state.messages.append({"role": "assistant", "content": first_question})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Skriv ditt svar här..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = ""
    show_results = False
    if "selected_city" not in st.session_state:
        selected_city = get_municipality_choice(prompt)
        if selected_city:
            st.session_state.selected_city = selected_city
            response = f"Du har valt {selected_city.capitalize()}.\nVilken typ av sjuksköterska är du intresserad av? Här är alternativen:\n"
            for nurse_type in common_nurse_types:
                response += f"- {nurse_type.capitalize()}\n"
        else:
            response = "Staden finns inte i våran lista! Försök igen."
    elif "selected_nurse_type" not in st.session_state:
        selected_nurse_type = get_nurse_type_choice(prompt)
        if selected_nurse_type:
            st.session_state.selected_nurse_type = selected_nurse_type
            response = f"Du har valt {selected_nurse_type.capitalize()}.\nVill du jobba heltid eller deltid?\n"
        else:
            response = "Denna typ av sjuksköterska finns inte i våran lista! Försök igen."
    elif "selected_working_hours" not in st.session_state:
        selected_working_hours = get_working_hours_choice(prompt)
        if selected_working_hours:
            st.session_state.selected_working_hours = selected_working_hours
            response = "Skriv in ett eller flera nyckelord för att filtrera jobb ytterligare:\n"
        else:
            response = "Denna arbetstid finns inte i våran lista! Försök igen."
    elif "selected_keywords" not in st.session_state:
        st.session_state.selected_keywords = prompt
        filtered_data = data[
            (data['Municipality'] == st.session_state.selected_city) &
            (data['Nurse_type'] == st.session_state.selected_nurse_type) &
            (data['Working_hours'] == st.session_state.selected_working_hours)
        ]
        keyword_filtered_data = filter_jobs_by_keyword(st.session_state.selected_keywords)
        final_filtered_data = pd.merge(
            filtered_data, keyword_filtered_data, how='inner',
            on=['Id', 'Headline', 'Application_deadline', 'Amount', 'Description', 'Type', 'Salary', 'Duration', 'Working_hours', 'Region', 'Municipality', 'Employer_name', 'Employer_workplace', 'Publication_date']
        )

        response = f"Resultat för jobb i {st.session_state.selected_city.capitalize()} som {st.session_state.selected_nurse_type.capitalize()} med {st.session_state.selected_working_hours} arbetstid och nyckelord '{st.session_state.selected_keywords}':\n\n"
        show_results = True
        if final_filtered_data.empty:
            response += "Inga jobb hittades."

    with st.chat_message("assistant"):
        st.markdown(response)

    if show_results and not final_filtered_data.empty:
        for index, row in final_filtered_data.iterrows():
            with st.expander(f"{row['Headline']}"):
                st.write(f"**Id:** {row['Id']}")
                st.write(f"**Titel:** {row['Headline']}")
                st.write(f"**Beskrivning:** {row['Description']}")
                st.write(f"**Typ:** {row['Type']}")
                st.write(f"**Lön:** {row['Salary']}")
                st.write(f"**Varaktighet:** {row['Duration']}")
                st.write(f"**Arbetstid:** {row['Working_hours']}")
                st.write(f"**Region:** {row['Region']}")
                st.write(f"**Kommun:** {row['Municipality']}")
                st.write(f"**Arbetsgivare:** {row['Employer_name']}")
                st.write(f"**Arbetsplats:** {row['Employer_workplace']}")
                st.write(f"**Publiceringsdatum:** {row['Publication_date']}\n")

    st.session_state.messages.append({"role": "assistant", "content": response})






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

