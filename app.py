import streamlit as st
import pandas as pd
import pickle

# Load dataset
data = pd.read_csv('anime.csv')

# Load model
with open('anime_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title('Anime Recommendation System')

menu = ['Home', 'Anime Data', 'Recommendation']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Home':
    st.write("""
        # Welcome to Anime Recommendation System
        Silakan pilih menu di sebelah kiri untuk menjelajahi data atau mendapatkan rekomendasi.
    """)

elif choice == 'Anime Data':
    st.subheader('Dataset Anime')
    st.dataframe(data)

elif choice == 'Recommendation':
    st.subheader('Anime Recommendation')

    # Pilih anime untuk referensi
    anime_list = data['name'].tolist()
    selected_anime = st.selectbox('Select an Anime', anime_list)

    # Ambil rekomendasi
    def recommend(selected_anime):
        index = data[data['name'] == selected_anime].index[0]
        distances = model[index]
        anime_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended = []
        for i in anime_indices:
            recommended.append(data.iloc[i[0]]['name'])
        return recommended

    if st.button('Show Recommendation'):
        recommendations = recommend(selected_anime)
        st.write('Recommended Animes:')
        for anime in recommendations:
            st.write(f'- {anime}')
