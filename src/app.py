import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="Anime High Score Predictor",
    page_icon="🎌",
    layout="centered"
)


@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "anime_model.pkl")
    model_package = joblib.load(model_path)
    return model_package

model_package = load_model()

model = model_package["model"]
final_features = model_package["final_features"]
top_genres = model_package["top_genres"]


st.title("🎌 Anime High Score Predictor")

st.write(
    """
    Esta aplicación predice si un anime podría tener una puntuación alta 
    según señales de recepción comunitaria y características generales.
    """
)

st.markdown("---")


st.subheader("Características del anime")

num_list_users = st.number_input(
    "Número de usuarios que agregaron el anime a su lista",
    min_value=0,
    value=100000,
    step=1000
)

num_scoring_users = st.number_input(
    "Número de usuarios que calificaron el anime",
    min_value=0,
    value=50000,
    step=1000
)

num_episodes = st.number_input(
    "Número de episodios",
    min_value=0,
    value=12,
    step=1
)

average_episode_duration = st.number_input(
    "Duración promedio del episodio en segundos",
    min_value=0,
    value=1440,
    step=60
)

media_type = st.selectbox(
    "Tipo de anime",
    ["tv", "movie", "ova", "ona", "special", "music", "unknown"]
)

rating = st.selectbox(
    "Rating",
    ["g", "pg", "pg_13", "r", "r+", "rx", "unknown"]
)

selected_genres = st.multiselect(
    "Géneros",
    top_genres
)


input_data = {
    "num_list_users": num_list_users,
    "num_scoring_users": num_scoring_users,
    "num_episodes": num_episodes,
    "average_episode_duration": average_episode_duration,
    "media_type": media_type,
    "rating": rating
}


for genre in top_genres:
    input_data[f"genre_{genre}"] = 1 if genre in selected_genres else 0

input_df = pd.DataFrame([input_data])


input_df = input_df[final_features]


st.markdown("---")

if st.button("Predecir puntuación alta"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Resultado")

    if prediction == 1:
        st.success("El modelo predice que este anime podría tener puntuación alta.")
    else:
        st.warning("El modelo predice que este anime probablemente no tendrá puntuación alta.")

    st.write(f"Probabilidad estimada de high score: **{probability:.2%}**")

    st.markdown("---")

    st.caption(
        "Nota: esta predicción depende del dataset usado y no representa una verdad absoluta sobre la calidad artística del anime."
    )


with st.expander("Ver datos enviados al modelo"):
    st.dataframe(input_df)