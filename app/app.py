from pathlib import Path
import re
import unicodedata

import joblib
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix, hstack


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "cleaned_spotify.csv"
MODEL_PATH = BASE_DIR / "models" / "hybrid_recommender.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
GENRE_ENCODER_PATH = BASE_DIR / "models" / "genre_encoder.joblib"


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    genre_encoder = joblib.load(GENRE_ENCODER_PATH)

    return model, scaler, genre_encoder


df = load_data()
model, scaler, genre_encoder = load_models()


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

AUDIO_WEIGHT = 1.0
GENRE_WEIGHT = 1.5


# --------------------------------------------------
# BUILD SONG VECTOR
# --------------------------------------------------

def build_song_vector(song):

    audio_values = (
        song[audio_features]
        .astype(float)
        .to_numpy()
        .reshape(1, -1)
    )

    audio_scaled = scaler.transform(audio_values)

    audio_matrix = csr_matrix(
        audio_scaled * AUDIO_WEIGHT
    )

    genres = str(song["track_genre"]).split(", ")

    genre_matrix = genre_encoder.transform(
        [genres]
    )

    genre_matrix = genre_matrix * GENRE_WEIGHT

    song_vector = hstack(
        [
            audio_matrix,
            genre_matrix
        ]
    ).tocsr()

    return song_vector


# --------------------------------------------------
# NORMALIZE SONG TITLES
# --------------------------------------------------

def normalize_title(title):

    title = str(title).lower()

    # Normalize unicode characters
    title = unicodedata.normalize(
        "NFKD",
        title
    )

    # Remove punctuation
    title = re.sub(
        r"[^\w\s]",
        " ",
        title
    )

    # Remove common alternate-version descriptions
    version_terms = [
        "acoustic version",
        "piano version",
        "live version",
        "radio edit",
        "remastered",
        "remaster",
        "bonus track"
    ]

    for term in version_terms:
        title = title.replace(term, "")

    # Remove extra spaces
    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    return title


# --------------------------------------------------
# RECOMMENDATION FUNCTION
# --------------------------------------------------

def recommend_songs(
    song_index,
    n_recommendations=10
):

    input_song = df.iloc[song_index]

    song_vector = build_song_vector(
        input_song
    )

    # Retrieve more songs than necessary
    # because some will be filtered out
    number_of_neighbors = min(
        n_recommendations + 50,
        len(df)
    )

    distances, indices = model.kneighbors(
        song_vector,
        n_neighbors=number_of_neighbors
    )

    recommendations = df.iloc[
        indices[0]
    ].copy()

    recommendations["similarity_score"] = (
        1 - distances[0]
    )

    # Remove exact input song
    recommendations = recommendations[
        recommendations["track_id"]
        != input_song["track_id"]
    ]

    # Remove alternate recordings
    # of the same song
    input_title_normalized = normalize_title(
        input_song["track_name"]
    )

    recommendations = recommendations[
        recommendations["track_name"].apply(normalize_title)
        != input_title_normalized
    ]

    # Remove duplicate song + artist combinations
    recommendations["_normalized_title"] = (
        recommendations["track_name"].apply(normalize_title)
    )

    recommendations["_normalized_artist"] = (
        recommendations["artists"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    recommendations = recommendations.drop_duplicates(
        subset=[
            "_normalized_title",
            "_normalized_artist"
        ],
        keep="first"
    )

    # Remove temporary helper columns
    recommendations = recommendations.drop(
        columns=[
            "_normalized_title",
            "_normalized_artist"
        ]
    )

    recommendations = recommendations.head(
        n_recommendations
    )

    return recommendations.reset_index(
        drop=True
    )


# --------------------------------------------------
# INTERFACE
# --------------------------------------------------

st.title(
    "🎵 Music Recommendation System"
)

st.write(
    """
    Discover songs similar to music you already enjoy.

    The recommendation engine combines Spotify audio
    characteristics with genre information to identify
    musically similar tracks.
    """
)

st.divider()


# --------------------------------------------------
# SONG SELECTION
# --------------------------------------------------

st.subheader("Choose a song")

song_index = st.selectbox(
    "Search for a song:",
    options=range(len(df)),
    format_func=lambda index:
        f"{df.iloc[index]['track_name']} — "
        f"{df.iloc[index]['artists']}"
)

selected_song = df.iloc[song_index]


# --------------------------------------------------
# SELECTED SONG INFORMATION
# --------------------------------------------------

st.subheader(
    selected_song["track_name"]
)

st.write(
    f"**{selected_song['artists']}**"
)

st.write(
    f"**Genre:** "
    f"{selected_song['track_genre']}"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Popularity",
        int(
            selected_song["popularity"]
        )
    )


with col2:

    duration_minutes = (
        selected_song["duration_ms"]
        / 60000
    )

    st.metric(
        "Duration",
        f"{duration_minutes:.1f} min"
    )


# --------------------------------------------------
# NUMBER OF RECOMMENDATIONS
# --------------------------------------------------

number_of_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)


# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button(
    "🎧 Recommend Songs",
    type="primary"
):

    with st.spinner(
        "Finding similar songs..."
    ):

        recommendations = recommend_songs(
            song_index,
            number_of_recommendations
        )


    # --------------------------------------------------
    # RECOMMENDATION TABLE
    # --------------------------------------------------

    st.subheader(
        "Recommended Songs"
    )


    display_df = recommendations[
        [
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "similarity_score"
        ]
    ].copy()


    display_df.columns = [
        "Song",
        "Artist",
        "Genre",
        "Popularity",
        "Similarity"
    ]


    display_df["Similarity"] = (
        display_df["Similarity"]
        * 100
    ).round(1)


    display_df.index = range(
        1,
        len(display_df) + 1
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={

            "Song":
                st.column_config.TextColumn(
                    "🎵 Song",
                    width="large"
                ),

            "Artist":
                st.column_config.TextColumn(
                    "🎤 Artist",
                    width="medium"
                ),

            "Genre":
                st.column_config.TextColumn(
                    "Genre",
                    width="medium"
                ),

            "Popularity":
                st.column_config.ProgressColumn(
                    "Popularity",
                    min_value=0,
                    max_value=100,
                    format="%d"
                ),

            "Similarity":
                st.column_config.ProgressColumn(
                    "Similarity",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                )
        }
    )


    # --------------------------------------------------
    # EXPLAINABLE RECOMMENDATIONS
    # --------------------------------------------------

    st.subheader(
        "Why these recommendations?"
    )

    st.write(
        """
        Recommendations are based on similarity across Spotify
        audio characteristics such as danceability, energy,
        acousticness, valence and tempo, together with genre
        information.
        """
    )


    comparison_features = [
        "danceability",
        "energy",
        "acousticness",
        "valence",
        "liveness",
        "speechiness"
    ]


    top_recommendation = (
        recommendations.iloc[0]
    )


    comparison_df = pd.DataFrame({

        "Feature": [
            "Danceability",
            "Energy",
            "Acousticness",
            "Valence",
            "Liveness",
            "Speechiness"
        ],

        "Selected Song": [
            selected_song[feature]
            for feature
            in comparison_features
        ],

        "Top Recommendation": [
            top_recommendation[feature]
            for feature
            in comparison_features
        ]
    })


    comparison_df = (
        comparison_df.set_index(
            "Feature"
        )
    )


    st.write(
        f"### "
        f"{selected_song['track_name']} "
        f"vs "
        f"{top_recommendation['track_name']}"
    )


    st.bar_chart(
        comparison_df,
        stack=False
    )


    # --------------------------------------------------
    # TECHNICAL EXPLANATION
    # --------------------------------------------------

    with st.expander(
        "How does the recommendation system work?"
    ):

        st.markdown(
            """
            **1. Audio features are standardized**

            Features such as tempo and danceability exist
            on different numerical scales, so
            `StandardScaler` is used to make them comparable.

            **2. Genre information is encoded**

            Each song's genres are converted into binary
            features using a multilabel encoder.

            **3. Audio and genre features are combined**

            Genre features are given additional weight to
            improve stylistic relevance.

            **4. Similar songs are identified**

            A Nearest Neighbors model using cosine distance
            finds songs whose combined feature vectors are
            closest to the selected song.

            **5. Duplicate and alternate recordings are filtered**

            The system removes the selected track and
            alternate versions before displaying the final
            recommendations.
            """
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Hybrid content-based recommendation system "
    "using audio features and genre similarity."
)