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
            response = f"Du har valt {selected_city.capitalize()}.\nVilken typ av sjuksköterska är du specialicerad som? Här är alternativen:\n"
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
            response = "Ange ett eller flera nyckelord som beskriver de egenskaper du vill att ditt jobb ska innehålla, till exempel utvecklande, ledarskap, ansvar, roligt eller liknande :\n"
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
