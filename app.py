
import json

import pandas as pd
import streamlit as st
from pycaret.clustering import load_model, predict_model
import plotly.express as px

MODEL_NAME1 = 'welcome_survey_clustering_pipeline_v1'
DATA1 = 'welcome_survey_simple_v1.csv'
CLUSTERS_NAMES_AND_DESCRIPTIONS1 = 'welcome_survey_cluster_names_and_descriptions_v1.json'

MODEL_NAME2 = 'welcome_survey_clustering_pipeline_v2'
DATA2 = 'welcome_survey_simple_v2.csv'
CLUSTERS_NAMES_AND_DESCRIPTIONS2 = 'welcome_survey_cluster_names_and_descriptions_v2.json'

st.session_state.setdefault("data_version", 'v1')
st.session_state.setdefault("MODEL_NAME", MODEL_NAME1)
st.session_state.setdefault("DATA", DATA1)
st.session_state.setdefault("CLUSTERS_NAMES_AND_DESCRIPTIONS", CLUSTERS_NAMES_AND_DESCRIPTIONS1)

def get_model():
    return load_model(st.session_state['MODEL_NAME'])


def get_clusters_names_and_descriptions():
    with open(st.session_state['CLUSTERS_NAMES_AND_DESCRIPTIONS'], 'rb') as f:
        return json.loads(f.read())

def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(st.session_state['DATA'], sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


def set_data_version():
    if st.session_state['data_version'] == 'v2':
        st.session_state.update(
            MODEL_NAME=MODEL_NAME2, 
            DATA=DATA2, 
            CLUSTERS_NAMES_AND_DESCRIPTIONS=CLUSTERS_NAMES_AND_DESCRIPTIONS2
        )
    else:
        st.session_state.update(
            MODEL_NAME=MODEL_NAME1, 
            DATA=DATA1, 
            CLUSTERS_NAMES_AND_DESCRIPTIONS=CLUSTERS_NAMES_AND_DESCRIPTIONS1
        )

st.title("Znajdź znajomych")

with st.sidebar:
    st.radio('Wersja danych', ['v1', 'v2'], key='data_version', on_change=set_data_version )
   
    
    st.header('Powiedź nam coś o sobie')
    st.markdown(
        "Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox(
        "Wiek", 
        ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    edu_level = st.selectbox(
        "Wykształcenie", 
        ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox(
        "Ulubione zwierzę", 
        ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i psy'])
    fav_place = st.selectbox(
        "Ulubione miejsce", 
        ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio(
        "Płeć", 
        ['Kobieta', 'Mężczyzna'])

    person_df = pd.DataFrame([
        {
            'age':age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender

        }        
    ])

st.write("Wybrane dane:")
st.dataframe(person_df, hide_index=True)

model = get_model()
all_df = get_all_participants()
clusters_names_and_descriptions = get_clusters_names_and_descriptions()

predicted_cluster_id = predict_model(
    model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = clusters_names_and_descriptions[predicted_cluster_id]
st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(f"{predicted_cluster_data['description']}")
same_cluster_df = all_df[all_df['Cluster'] == predicted_cluster_id]
st.metric("Liczba Twoich znajomych: ", len(same_cluster_df))

st.header("Osoby z grupy")
fig = px.histogram(same_cluster_df.sort_values('age'), x="age")
fig.update_layout(
    title="Rozkład wieku w grupie",
    xaxis_title="wiek",
    yaxis_title='liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df.sort_values('edu_level'), x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia w grupie",
    xaxis_title="wykształcenie",
    yaxis_title='liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df.sort_values('fav_animals'), x="fav_animals")
fig.update_layout(
    title="Rozkład ulub. zwierząt w grupie",
    xaxis_title="fav_animals",
    yaxis_title='liczba osób'
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df.sort_values('gender'), x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="gender",
    yaxis_title='liczba osób'
)
st.plotly_chart(fig)
