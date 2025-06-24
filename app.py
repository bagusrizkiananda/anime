import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('anime.csv')

# Preprocessing: Pastikan tidak ada NaN di kolom 'genre'
data['genre'] = data['genre'].fillna('')

# Hitung TF-IDF dari genre
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['genre'])

# Hitung cosine similarity antar anime
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Buat indeks nama anime untuk pencarian cepat
indices = pd.Series(data.index, index=data['name']).drop_duplicates()

# Fungsi rekomendasi berdasarkan anime
def recommend(anime_name):
    if anime_name not in indices:
        return []
    
    idx = indices[anime_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Ambil 5 rekomendasi teratas
    anime_indices = [i[0] for i in sim_scores]
    return data.iloc[anime_indices][['name', 'genre', 'rating']]  # Mengambil nama, genre, rating

# Fungsi pencarian anime berdasarkan genre
def search_by_genre(genre_keyword):
    filtered = data[data['genre'].str.contains(genre_keyword, case=False, na=False)]
    return filtered[['name', 'genre', 'rating']].sort_values(by='rating', ascending=False).head(10)

# Streamlit UI
st.set_page_config(page_title="Anime Recommender", page_icon="🎌", layout="wide")
st.title('🎌 Sistem Rekomendasi Anime 🎌')

menu = ['🏠 Home', '📚 Anime Data', '✨ Recommendation', '🔍 Search by Genre']
choice = st.sidebar.selectbox('Menu', menu)

if choice == '🏠 Home':
    st.markdown("""
        ## Irasshaimase! 👋
        Selamat datang di **Sistem Rekomendasi Anime** berbasis kemiripan genre.
        
        > Pilih menu *Recommendation* di sidebar untuk mulai mencari rekomendasi anime favoritmu.
    """)

elif choice == '📚 Anime Data':
    st.subheader('📄 Data List Anime')
    st.dataframe(data)

elif choice == '✨ Recommendation':
    st.subheader('✨ Rekomendasi Anime Serupa')

    anime_list = data['name'].tolist()
    selected_anime = st.selectbox('Select an Anime', anime_list)

    if st.button('Show Recommendation'):
        recommendations = recommend(selected_anime)
        if len(recommendations) == 0:
            st.warning('Anime tidak ditemukan dalam data.')
        else:
            st.success('Berikut adalah anime rekomendasi untuk kamu:')
            for idx, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style='background-color:#2b2b2b; padding:15px; border-radius:10px; margin-bottom:10px; color:#ffffff;'>
                        <h4 style='margin-bottom:5px;'>{row['name']}</h4>
                        <p style='margin-bottom:5px;'><b>Genre:</b> {row['genre']}</p>
                        <p><b>Rating:</b> {row['rating']}</p>
                    </div>
                    """, unsafe_allow_html=True)

elif choice == '🔍 Search by Genre':
    st.subheader('🔎 Cari Anime Berdasarkan Genre')
    genre_input = st.text_input('Masukkan Genre (Contoh: Action, Comedy, Romance)')

    if st.button('Search'):
        results = search_by_genre(genre_input)
        if len(results) == 0:
            st.warning('Tidak ditemukan anime dengan genre tersebut.')
        else:
            st.success('Berikut adalah anime dengan genre yang kamu cari:')
            for idx, row in results.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style='background-color:#2b2b2b; padding:15px; border-radius:10px; margin-bottom:10px; color:#ffffff;'>
                        <h4 style='margin-bottom:5px;'>{row['name']}</h4>
                        <p style='margin-bottom:5px;'><b>Genre:</b> {row['genre']}</p>
                        <p><b>Rating:</b> {row['rating']}</p>
                    </div>
                    """, unsafe_allow_html=True)
