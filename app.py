from flask import Flask, jsonify, request, render_template
import gensim.downloader
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask
app = Flask(__name__)

# Load the GloVe model
glove = gensim.downloader.load('glove-wiki-gigaword-100')

# Load the metadata dataset
df_metadata = pd.read_csv('data/metadata.csv', index_col=0)

# Clean the metadata
df_metadata = df_metadata.dropna(subset=["original_title", "overview"], how="any")

# Text Cleaning
df_metadata["combined"] = df_metadata.original_title + " " + df_metadata.overview
df_metadata["combined"] = df_metadata["combined"].str.lower()
df_metadata["tokenized"] = df_metadata.combined.apply(lambda x: word_tokenize(x))
df_metadata["clean_tokenized"] = df_metadata["tokenized"].apply(
    lambda tokens: [word for word in tokens if word.isalpha() and word not in stopwords.words("english")]
)
df_metadata.drop(columns=["combined", "tokenized"], inplace=True)

def get_embedding(list_of_tokens):
    embeddings = np.zeros(100)
    for token in list_of_tokens:
        if token in glove:
            embeddings += glove[token]
    return embeddings

df_metadata["embedding"] = df_metadata["clean_tokenized"].apply(lambda x: get_embedding(x))

def recommend(df_metadata, top_n=50):
    # Ensure all embeddings are numpy arrays of the correct shape
    df_metadata["embedding"] = df_metadata["embedding"].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    # Convert all embeddings to a 2D array
    all_embeddings = np.array(df_metadata["embedding"].tolist())

    # Calculate cosine similarity for each item with all other items
    def calculate_cosine_similarity(x):
        if isinstance(x, np.ndarray) and x.ndim == 1:
            return cosine_similarity(x.reshape(1, -1), all_embeddings)[0]
        else:
            return np.zeros(all_embeddings.shape[0])

    df_metadata["cosine"] = df_metadata.embedding.apply(calculate_cosine_similarity)

    # Get top N recommendations based on content similarity
    recommendations = []
    for index, row in df_metadata.iterrows():
        similar_items = sorted(
            zip(df_metadata["item_id"], row["cosine"], df_metadata["original_title"]),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        recommendations.extend([{"item_id": item[0], "original_title": item[2], "cosine": item[1]} for item in similar_items])

    return pd.DataFrame(recommendations).drop_duplicates(subset=["item_id"]).head(top_n)

# Test the recommendation function
recommendations = recommend(df_metadata)
print("Recommendations:", recommendations)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    if request.method == "POST":
        recommendations_df = recommend(df_metadata)
        
        if not recommendations_df.empty:
            recommendations = recommendations_df.to_dict(orient='records')
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
