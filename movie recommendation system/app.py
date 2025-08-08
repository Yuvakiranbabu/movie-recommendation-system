import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
df = pd.read_csv('tmdb_5000_movies.csv')

# Drop rows with missing overviews
df = df.dropna(subset=['overview']).reset_index(drop=True)

# Create the TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Create a reverse mapping of indices and movie titles
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Define the recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    title = title.lower()
    matches = indices.index[indices.index.str.lower() == title]
    if matches.empty:
        return []
    idx = indices[matches[0]]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 recommendations

    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter a movie title:")

if st.button("Recommend"):
    if movie_name:
        results = get_recommendations(movie_name)
        if results:
            st.subheader("Top 5 Recommended Movies:")
            for movie in results:
                st.write("• " + movie)
        else:
            st.warning("Movie not found. Please check the spelling or try a different title.")
    else:
        st.warning("Please enter a movie title.")
