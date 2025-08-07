#!/usr/bin/env python3
"""
🎵 Music AI Inferencing Dashboard - Bulletproof Version
Comprehensive Streamlit app for analyzing music data from 2016-2024
with AI inferencing, year filtering, dynamic recommendations, Last.fm integration,
automatic CSV to JSON conversion, and bulletproof error handling

Usage: streamlit run music_ai_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import os
import tempfile
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import Last.fm integration and CSV converter
from lastfm_integration import LastFMAPI, generate_lastfm_recommendations, display_lastfm_recommendations, test_lastfm_api
from csv_to_json_converter import convert_csv_to_json, validate_converted_data

# Page configuration
st.set_page_config(
    page_title="🎵 Music AI Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1DB954;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .lastfm-section {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #1DB954;
        margin: 1rem 0;
        text-align: center;
    }
    .csv-conversion-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_music_data(file_path, is_csv=False):
    """Load and process music data with caching for performance"""
    try:
        if is_csv:
            # Convert CSV to JSON format
            data = convert_csv_to_json(file_path)
            
            # Validate the converted data
            is_valid, errors = validate_converted_data(data)
            if not is_valid:
                st.error(f"CSV conversion validation failed: {'; '.join(errors)}")
                return None
            
            st.success(f"✅ Successfully converted CSV to JSON format ({len(data)} records)")
        else:
            # Load JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert timestamps to datetime
        if 'Added At' in df.columns:
            df['Added At'] = pd.to_datetime(df['Added At'], unit='ms')
        if 'Release Date' in df.columns:
            df['Release Date'] = pd.to_datetime(df['Release Date'], unit='ms')
        
        # Handle genres (convert list to string for easier processing)
        if 'Genres' in df.columns:
            df['Genres_str'] = df['Genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            df['Primary_Genre'] = df['Genres'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'Unknown')
        
        # Ensure all audio features are numeric
        audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                         'Instrumentalness', 'Liveness', 'Speechiness', 'Tempo', 'Loudness']
        
        for feature in audio_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def classify_mood(row):
    """Classify song mood based on valence and energy"""
    valence = row['Valence']
    energy = row['Energy']
    
    if pd.isna(valence) or pd.isna(energy):
        return 'Unknown'
    
    if valence >= 0.6 and energy >= 0.6:
        return 'Happy/Energetic'
    elif valence >= 0.6 and energy < 0.6:
        return 'Happy/Calm'
    elif valence < 0.4 and energy >= 0.6:
        return 'Intense/Aggressive'
    else:
        return 'Sad/Mellow'

def perform_clustering(df, n_clusters=3):
    """
    Perform K-means clustering on audio features with improved error handling
    
    FIXED: Now handles cases where n_samples < n_clusters
    """
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    # Prepare data
    X = df[audio_features].fillna(df[audio_features].mean())
    
    # FIX: Check if we have enough samples for clustering
    n_samples = len(X)
    
    if n_samples < 2:
        # Not enough samples for clustering - return single cluster
        st.warning(f"⚠️ Only {n_samples} song(s) available. Clustering requires at least 2 songs.")
        clusters = np.zeros(n_samples, dtype=int)  # All songs in cluster 0
        return clusters, None, None
    
    # FIX: Adjust number of clusters if we don't have enough samples
    effective_clusters = min(n_clusters, n_samples)
    
    if effective_clusters < n_clusters:
        st.info(f"ℹ️ Adjusted clusters from {n_clusters} to {effective_clusters} due to limited data ({n_samples} songs)")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering with adjusted number of clusters
        kmeans = KMeans(n_clusters=effective_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        return clusters, kmeans, scaler
        
    except Exception as e:
        # FIX: Fallback error handling
        st.error(f"❌ Clustering failed: {str(e)}")
        st.info("🔄 Falling back to single cluster assignment")
        clusters = np.zeros(n_samples, dtype=int)  # All songs in cluster 0
        return clusters, None, None

def create_radar_chart(df, title="Audio Features Profile"):
    """Create a radar chart for audio features"""
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    # Calculate mean values
    means = df[audio_features].mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=means.values,
        theta=audio_features,
        fill='toself',
        name='Your Profile',
        line_color='#1DB954'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        height=400
    )
    
    return fig

def generate_dynamic_recommendations(df, user_profile, n_recommendations=10, randomness=0.3, tier_range=None):
    """Generate dynamic recommendations with controlled randomness and tier filtering"""
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    # Apply tier filtering if specified
    if tier_range:
        # Sort by popularity and take the specified tier
        df_sorted = df.sort_values('Popularity', ascending=False)
        start_idx = int(len(df_sorted) * tier_range[0])
        end_idx = int(len(df_sorted) * tier_range[1])
        df = df_sorted.iloc[start_idx:end_idx]
    
    # Calculate similarity scores
    similarities = []
    for idx, song in df.iterrows():
        song_features = song[audio_features].fillna(user_profile).values
        user_features = user_profile.values
        
        # Calculate cosine similarity
        dot_product = np.dot(song_features, user_features)
        norms = np.linalg.norm(song_features) * np.linalg.norm(user_features)
        
        if norms == 0:
            similarity = 0
        else:
            similarity = dot_product / norms
        
        # Add controlled randomness
        random_factor = random.uniform(1 - randomness, 1 + randomness)
        adjusted_similarity = similarity * random_factor
        
        similarities.append(adjusted_similarity)
    
    df_temp = df.copy()
    df_temp['Similarity'] = similarities
    
    # Get top recommendations
    recommendations = df_temp.nlargest(n_recommendations, 'Similarity')
    
    return recommendations[['Track Name', 'Artist Name(s)', 'Album Name', 'Similarity', 'Year', 'Popularity']]

def safe_slider(label, min_val, max_val, default_val, help_text=None, key=None):
    """
    BULLETPROOF SLIDER: Ensures min_value < max_value always
    
    This function prevents the StreamlitAPIException by validating slider parameters
    """
    # BULLETPROOF FIX: Ensure min_val < max_val
    if min_val >= max_val:
        # If min >= max, adjust values to make them valid
        if max_val <= 1:
            # Very small dataset - use fixed values
            min_val = 1
            max_val = 2
            default_val = 1
            st.warning(f"⚠️ {label}: Adjusted to minimum range due to limited data")
        else:
            # Adjust min_val to be less than max_val
            min_val = max(1, max_val - 1)
            default_val = min(default_val, max_val)
            st.info(f"ℹ️ {label}: Adjusted range to {min_val}-{max_val} due to data constraints")
    
    # BULLETPROOF FIX: Ensure default_val is within range
    default_val = max(min_val, min(default_val, max_val))
    
    # Create the slider with validated parameters
    return st.slider(label, min_value=min_val, max_value=max_val, value=default_val, help=help_text, key=key)

def setup_data_source():
    """Setup data source selection with CSV and JSON support"""
    st.sidebar.header("📁 Data Source")
    
    # Option to upload file or use existing path
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload file", "Use file path"]
    )
    
    data_file = None
    is_csv = False
    
    if data_source == "Upload file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your music data file",
            type=['json', 'csv'],
            help="Upload your Spotify history JSON file or CSV file (will be auto-converted to JSON)"
        )
        
        if uploaded_file is not None:
            # Determine file type
            file_extension = uploaded_file.name.lower().split('.')[-1]
            is_csv = file_extension == 'csv'
            
            # Save uploaded file temporarily
            temp_path = f"/tmp/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            data_file = temp_path
            
            # Show conversion info for CSV files
            if is_csv:
                st.sidebar.markdown("""
                <div class="csv-conversion-info">
                    <h4>🔄 CSV Auto-Conversion</h4>
                    <p>Your CSV file will be automatically converted to JSON format for analysis.</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        # File path input
        default_path = "./spotify_history_final_cleaned.json"
        data_file = st.sidebar.text_input(
            "File path to music data:",
            value=default_path,
            help="Enter the path to your JSON or CSV file"
        )
        
        if data_file and os.path.exists(data_file):
            # Determine file type from extension
            file_extension = data_file.lower().split('.')[-1]
            is_csv = file_extension == 'csv'
            
            if is_csv:
                st.sidebar.info("🔄 CSV file detected - will be auto-converted to JSON")
        elif data_file:
            st.sidebar.error(f"File not found: {data_file}")
            data_file = None
    
    return data_file, is_csv

def setup_lastfm_integration():
    """Setup Last.fm integration in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎵 Last.fm Integration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Last.fm API Key",
        type="password",
        help="Enter your Last.fm API key for enhanced recommendations"
    )
    
    if api_key:
        # Test API key
        if st.sidebar.button("Test API Key"):
            with st.spinner("Testing Last.fm API key..."):
                if test_lastfm_api(api_key):
                    st.sidebar.success("✅ API key is valid!")
                    st.session_state.lastfm_api_key = api_key
                else:
                    st.sidebar.error("❌ Invalid API key or connection failed")
        
        # Store API key in session state if valid
        if 'lastfm_api_key' not in st.session_state:
            st.session_state.lastfm_api_key = api_key
    
    return api_key

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 Music AI Inferencing Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze your music listening patterns from 2016-2024 with AI-powered insights and Last.fm integration**")
    st.markdown("**🛡️ Bulletproof Edition: Unbreakable error handling + CSV support!**")
    
    # Setup data source
    data_file, is_csv = setup_data_source()
    
    if not data_file:
        st.markdown("""
        <div class="upload-section">
            <h3>🎵 Welcome to Music AI Dashboard!</h3>
            <p>To get started, please upload your music data file or specify the file path in the sidebar.</p>
            <p><strong>Supported formats:</strong></p>
            <ul style="text-align: left; display: inline-block;">
                <li><strong>JSON</strong> - Direct compatibility with dashboard</li>
                <li><strong>CSV</strong> - Automatically converted to JSON format</li>
            </ul>
            <p><strong>Expected fields:</strong> Track Name, Artist Name(s), Danceability, Energy, Valence, etc.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**JSON Format:**")
            sample_json = {
                "Track URI": "spotify:track:example",
                "Track Name": "Example Song",
                "Artist Name(s)": "Example Artist",
                "Album Name": "Example Album",
                "Year": 2023,
                "Danceability": 0.75,
                "Energy": 0.85,
                "Valence": 0.65,
                "Acousticness": 0.15,
                "Instrumentalness": 0.05,
                "Liveness": 0.12,
                "Speechiness": 0.08,
                "Popularity": 75,
                "Genres": ["pop", "electronic"]
            }
            st.json(sample_json)
        
        with col2:
            st.write("**CSV Format (will be auto-converted):**")
            sample_csv = """Track Name,Artist Name(s),Danceability,Energy,Valence
Example Song,Example Artist,0.75,0.85,0.65
Another Song,Another Artist,0.60,0.70,0.80"""
            st.code(sample_csv, language="csv")
        
        return
    
    # Load data
    df = load_music_data(data_file, is_csv)
    
    if df is None:
        st.error("Failed to load music data. Please check the file format and try again.")
        return
    
    # Show data source info
    if is_csv:
        st.info(f"📊 Loaded CSV data with automatic JSON conversion: {len(df)} tracks")
    else:
        st.info(f"📊 Loaded JSON data: {len(df)} tracks")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Year filter
    available_years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Years to Analyze",
        options=available_years,
        default=available_years,
        help="Choose which years to include in the analysis"
    )
    
    if not selected_years:
        st.warning("Please select at least one year to analyze.")
        return
    
    # Filter data by selected years
    filtered_df = df[df['Year'].isin(selected_years)].copy()
    
    # Additional filters
    st.sidebar.subheader("Additional Filters")
    
    # Genre filter
    all_genres = []
    for genres in filtered_df['Genres'].dropna():
        if isinstance(genres, list):
            all_genres.extend(genres)
    unique_genres = sorted(list(set(all_genres)))
    
    selected_genres = st.sidebar.multiselect(
        "Filter by Genres",
        options=unique_genres,
        help="Leave empty to include all genres"
    )
    
    if selected_genres:
        filtered_df = filtered_df[filtered_df['Genres'].apply(
            lambda x: any(genre in selected_genres for genre in x) if isinstance(x, list) else False
        )]
    
    # BULLETPROOF FIX: Popularity filter with safe slider
    if len(filtered_df) > 0:
        min_pop = int(filtered_df['Popularity'].min())
        max_pop = int(filtered_df['Popularity'].max())
        
        # Ensure min < max for slider
        if min_pop >= max_pop:
            if max_pop == 0:
                min_pop, max_pop = 0, 1
            else:
                min_pop = max(0, max_pop - 1)
        
        min_popularity, max_popularity = st.sidebar.slider(
            "Popularity Range",
            min_value=min_pop,
            max_value=max_pop,
            value=(min_pop, max_pop),
            help="Filter songs by Spotify popularity score"
        )
        
        filtered_df = filtered_df[
            (filtered_df['Popularity'] >= min_popularity) & 
            (filtered_df['Popularity'] <= max_popularity)
        ]
    
    # Setup Last.fm integration
    lastfm_api_key = setup_lastfm_integration()
    
    # Display basic stats
    st.sidebar.markdown("---")
    st.sidebar.metric("Total Songs", len(filtered_df))
    if len(selected_years) > 0:
        st.sidebar.metric("Years Span", f"{min(selected_years)}-{max(selected_years)}")
    st.sidebar.metric("Unique Artists", filtered_df['Artist Name(s)'].nunique())
    
    if len(filtered_df) == 0:
        st.warning("No songs match the selected filters. Please adjust your selection.")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🎯 AI Clustering", "😊 Mood Analysis", 
        "🎵 AI Recommendations", "🌟 Last.fm Recommendations", "📈 Trends"
    ])
    
    with tab1:
        st.header("📊 Music Collection Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Songs", len(filtered_df))
        with col2:
            st.metric("Unique Artists", filtered_df['Artist Name(s)'].nunique())
        with col3:
            avg_popularity = filtered_df['Popularity'].mean()
            st.metric("Avg Popularity", f"{avg_popularity:.1f}")
        with col4:
            if 'Duration (ms)' in filtered_df.columns:
                total_hours = filtered_df['Duration (ms)'].sum() / (1000 * 60 * 60)
                st.metric("Total Hours", f"{total_hours:.1f}")
            else:
                st.metric("Total Hours", "N/A")
        
        # Audio features profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎵 Your Audio Features Profile")
            radar_fig = create_radar_chart(filtered_df, "Your Music Preferences")
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with col2:
            st.subheader("📅 Songs by Year")
            year_counts = filtered_df['Year'].value_counts().sort_index()
            fig = px.bar(
                x=year_counts.index, 
                y=year_counts.values,
                labels={'x': 'Year', 'y': 'Number of Songs'},
                color=year_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top artists and genres
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎤 Top Artists")
            top_artists = filtered_df['Artist Name(s)'].value_counts().head(10)
            fig = px.bar(
                x=top_artists.values,
                y=top_artists.index,
                orientation='h',
                labels={'x': 'Number of Songs', 'y': 'Artist'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎼 Top Genres")
            genre_counts = {}
            for genres in filtered_df['Genres'].dropna():
                if isinstance(genres, list):
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            if genre_counts:
                top_genres = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
                fig = px.bar(
                    x=list(top_genres.values()),
                    y=list(top_genres.keys()),
                    orientation='h',
                    labels={'x': 'Number of Songs', 'y': 'Genre'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🎯 AI-Powered Music Clustering")
        st.markdown("Discover hidden patterns in your music taste using machine learning clustering.")
        
        # BULLETPROOF: Better clustering controls with validation
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # BULLETPROOF FIX: Limit max clusters based on available data
            max_possible_clusters = min(6, len(filtered_df))
            if max_possible_clusters < 2:
                st.warning("⚠️ Need at least 2 songs for clustering")
                n_clusters = 1
            else:
                n_clusters = safe_slider(
                    "Number of Clusters", 
                    min_val=2, 
                    max_val=max_possible_clusters, 
                    default_val=min(3, max_possible_clusters),
                    help_text=f"Maximum {max_possible_clusters} clusters available for {len(filtered_df)} songs"
                )
        
        # BULLETPROOF: Perform clustering with improved error handling
        try:
            clusters, kmeans, scaler = perform_clustering(filtered_df, n_clusters)
            filtered_df['Cluster'] = clusters
            
            # Get actual number of clusters created
            actual_clusters = len(np.unique(clusters))
            
            # Cluster analysis
            st.subheader("🔍 Cluster Analysis")
            
            cluster_stats = []
            for i in range(actual_clusters):
                cluster_songs = filtered_df[filtered_df['Cluster'] == i]
                if len(cluster_songs) > 0:
                    cluster_stats.append({
                        'Cluster': f'Cluster {i+1}',
                        'Songs': len(cluster_songs),
                        'Avg Energy': cluster_songs['Energy'].mean(),
                        'Avg Valence': cluster_songs['Valence'].mean(),
                        'Avg Danceability': cluster_songs['Danceability'].mean(),
                        'Top Artist': cluster_songs['Artist Name(s)'].mode().iloc[0] if len(cluster_songs) > 0 else 'N/A'
                    })
            
            if cluster_stats:
                cluster_df = pd.DataFrame(cluster_stats)
                st.dataframe(cluster_df, use_container_width=True)
                
                # Cluster visualizations (only if we have valid clustering)
                if kmeans is not None and len(filtered_df) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA visualization
                        audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                                         'Instrumentalness', 'Liveness', 'Speechiness']
                        X = filtered_df[audio_features].fillna(filtered_df[audio_features].mean())
                        
                        if len(X) >= 2:  # Need at least 2 samples for PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
                            
                            fig = px.scatter(
                                x=X_pca[:, 0], 
                                y=X_pca[:, 1], 
                                color=filtered_df['Cluster'].astype(str),
                                title="Music Clusters (PCA Visualization)",
                                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', 
                                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
                                hover_data={'Track Name': filtered_df['Track Name'], 
                                           'Artist': filtered_df['Artist Name(s)']}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Energy vs Valence by cluster
                        fig = px.scatter(
                            filtered_df, 
                            x='Valence', 
                            y='Energy',
                            color='Cluster',
                            title="Energy vs Valence by Cluster",
                            hover_data=['Track Name', 'Artist Name(s)']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sample songs from each cluster
                st.subheader("🎵 Sample Songs from Each Cluster")
                for i in range(actual_clusters):
                    cluster_songs = filtered_df[filtered_df['Cluster'] == i]
                    if len(cluster_songs) > 0:
                        st.write(f"**Cluster {i+1} ({len(cluster_songs)} songs):**")
                        sample_songs = cluster_songs.sample(min(5, len(cluster_songs)))
                        for _, song in sample_songs.iterrows():
                            st.write(f"• {song['Track Name']} by {song['Artist Name(s)']}")
                        st.write("")
            else:
                st.info("No clusters could be created with the current data.")
                
        except Exception as e:
            st.error(f"❌ Clustering analysis failed: {str(e)}")
            st.info("💡 Try adjusting your filters to include more songs.")
    
    with tab3:
        st.header("😊 Mood Analysis & Classification")
        st.markdown("AI-powered mood classification based on audio features.")
        
        # Add mood classification
        filtered_df['Mood'] = filtered_df.apply(classify_mood, axis=1)
        
        # Mood distribution
        col1, col2 = st.columns(2)
        
        with col1:
            mood_counts = filtered_df['Mood'].value_counts()
            fig = px.pie(
                values=mood_counts.values,
                names=mood_counts.index,
                title="Mood Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mood by year
            mood_year = filtered_df.groupby(['Year', 'Mood']).size().unstack(fill_value=0)
            fig = px.bar(
                mood_year,
                title="Mood Distribution by Year",
                labels={'value': 'Number of Songs', 'index': 'Year'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Mood characteristics
        st.subheader("🎭 Mood Characteristics")
        
        audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness']
        mood_features = filtered_df.groupby('Mood')[audio_features].mean()
        
        fig = px.bar(
            mood_features.T,
            title="Average Audio Features by Mood",
            labels={'index': 'Audio Feature', 'value': 'Average Value'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Train mood classifier and show accuracy
        if len(filtered_df) > 10:  # Ensure enough data for training
            audio_features_full = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                                  'Instrumentalness', 'Liveness', 'Speechiness']
            
            X = filtered_df[audio_features_full].fillna(filtered_df[audio_features_full].mean())
            y = filtered_df['Mood']
            
            if len(y.unique()) > 1:  # Ensure multiple classes
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
                    rf_classifier.fit(X_train, y_train)
                    accuracy = rf_classifier.score(X_test, y_test)
                    
                    st.success(f"🎯 AI Mood Classification Accuracy: {accuracy:.1%}")
                except Exception as e:
                    st.warning(f"⚠️ Could not train mood classifier: {str(e)}")
    
    with tab4:
        st.header("🎵 AI-Powered Dynamic Recommendations")
        st.markdown("Get personalized music recommendations based on your listening patterns using advanced AI algorithms.")
        
        # BULLETPROOF FIX: Recommendation controls with safe sliders
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # BULLETPROOF FIX: Safe slider for recommendations count
            max_recs = len(filtered_df)
            n_recommendations = safe_slider(
                "Number of Recommendations", 
                min_val=1, 
                max_val=max(1, min(20, max_recs)), 
                default_val=min(10, max_recs),
                help_text=f"Maximum {max_recs} recommendations available"
            )
            
        with col2:
            randomness = st.slider("Randomness Level", 0.0, 0.5, 0.2, 
                                 help="Higher values = more diverse recommendations")
        with col3:
            tier_options = {
                "All Songs": None,
                "Top Tier (Top 25%)": (0.0, 0.25),
                "High Tier (25-50%)": (0.25, 0.50),
                "Mid Tier (50-75%)": (0.50, 0.75),
                "Lower Tier (75-100%)": (0.75, 1.0)
            }
            selected_tier = st.selectbox("Artist Tier", list(tier_options.keys()))
            tier_range = tier_options[selected_tier]
        with col4:
            if st.button("🔄 Generate New Recommendations"):
                st.rerun()
        
        # Calculate user profile
        audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                         'Instrumentalness', 'Liveness', 'Speechiness']
        user_profile = filtered_df[audio_features].mean()
        
        # Generate recommendations
        recommendations = generate_dynamic_recommendations(
            filtered_df, user_profile, n_recommendations, randomness, tier_range
        )
        
        # Display recommendations
        st.subheader("🎯 Your Personalized AI Recommendations")
        
        for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**{i}. {rec['Track Name']}**")
                album_info = f" • {rec['Album Name']}" if pd.notna(rec['Album Name']) else ""
                st.write(f"by {rec['Artist Name(s)']}{album_info} ({rec['Year']})")
            
            with col2:
                st.metric("AI Similarity", f"{rec['Similarity']:.3f}")
                st.write(f"Popularity: {rec['Popularity']}")
        
        # Show user profile
        st.subheader("👤 Your Music Profile")
        profile_df = pd.DataFrame({
            'Feature': user_profile.index,
            'Your Average': user_profile.values
        })
        
        fig = px.bar(
            profile_df,
            x='Feature',
            y='Your Average',
            title="Your Audio Feature Preferences"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("🌟 Last.fm Enhanced Recommendations")
        
        if not lastfm_api_key:
            st.markdown("""
            <div class="lastfm-section">
                <h3>🔑 Last.fm API Key Required</h3>
                <p>To get enhanced recommendations from Last.fm, please:</p>
                <ol>
                    <li>Go to <a href="https://www.last.fm/api/account/create" target="_blank">Last.fm API</a> and create an account</li>
                    <li>Get your API key</li>
                    <li>Enter it in the sidebar under "Last.fm Integration"</li>
                </ol>
                <p>Last.fm provides access to a vast database of music similarity data and can give you recommendations based on what other users with similar taste listen to.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("🎵 **Enhanced recommendations powered by Last.fm's vast music database**")
            
            # Last.fm recommendation controls
            col1, col2 = st.columns(2)
            
            with col1:
                # BULLETPROOF FIX: Safe slider for Last.fm recommendations
                max_recommendations = safe_slider(
                    "Max Recommendations per Category", 
                    min_val=1, 
                    max_val=20, 
                    default_val=10
                )
            with col2:
                if st.button("🔄 Get New Last.fm Recommendations"):
                    st.rerun()
            
            # Get user's top artists and tracks for recommendations
            top_artists = filtered_df['Artist Name(s)'].value_counts().head(10).index.tolist()
            top_tracks = filtered_df.groupby(['Artist Name(s)', 'Track Name']).size().reset_index(name='count')
            top_tracks = top_tracks.nlargest(10, 'count')
            user_tracks = [(row['Artist Name(s)'], row['Track Name']) for _, row in top_tracks.iterrows()]
            
            # Generate Last.fm recommendations
            if st.button("🚀 Generate Last.fm Recommendations") or 'lastfm_recommendations' not in st.session_state:
                with st.spinner("Fetching recommendations from Last.fm..."):
                    try:
                        lastfm_api = LastFMAPI(lastfm_api_key)
                        recommendations = generate_lastfm_recommendations(
                            lastfm_api, top_artists, user_tracks, max_recommendations
                        )
                        st.session_state.lastfm_recommendations = recommendations
                        st.success("✅ Last.fm recommendations generated!")
                    except Exception as e:
                        st.error(f"Error generating Last.fm recommendations: {str(e)}")
                        st.session_state.lastfm_recommendations = None
            
            # Display Last.fm recommendations
            if 'lastfm_recommendations' in st.session_state and st.session_state.lastfm_recommendations:
                display_lastfm_recommendations(st.session_state.lastfm_recommendations)
            else:
                st.info("Click 'Generate Last.fm Recommendations' to get personalized suggestions!")
    
    with tab6:
        st.header("📈 Listening Trends Over Time")
        st.markdown("Analyze how your music preferences have evolved from 2016-2024.")
        
        # Trends over time
        yearly_features = filtered_df.groupby('Year')[
            ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Popularity']
        ].mean()
        
        # Feature trends
        st.subheader("🎵 Audio Feature Trends")
        
        feature_to_plot = st.selectbox(
            "Select Feature to Analyze",
            ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Popularity']
        )
        
        fig = px.line(
            x=yearly_features.index,
            y=yearly_features[feature_to_plot],
            title=f"{feature_to_plot} Trend Over Time",
            labels={'x': 'Year', 'y': feature_to_plot}
        )
        fig.add_scatter(
            x=yearly_features.index,
            y=yearly_features[feature_to_plot],
            mode='markers',
            marker=dict(size=8)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Multiple features comparison
        st.subheader("📊 Multiple Features Comparison")
        
        fig = go.Figure()
        
        features_to_compare = ['Danceability', 'Energy', 'Valence', 'Acousticness']
        colors = ['#1DB954', '#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for feature, color in zip(features_to_compare, colors):
            fig.add_trace(go.Scatter(
                x=yearly_features.index,
                y=yearly_features[feature],
                mode='lines+markers',
                name=feature,
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="Audio Features Evolution Over Time",
            xaxis_title="Year",
            yaxis_title="Feature Value",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Genre evolution
        st.subheader("🎼 Genre Evolution")
        
        # Calculate genre popularity by year
        genre_year_data = []
        for year in selected_years:
            year_data = filtered_df[filtered_df['Year'] == year]
            genre_counts = {}
            for genres in year_data['Genres'].dropna():
                if isinstance(genres, list):
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            for genre, count in genre_counts.items():
                genre_year_data.append({'Year': year, 'Genre': genre, 'Count': count})
        
        if genre_year_data:
            genre_df = pd.DataFrame(genre_year_data)
            top_genres = genre_df.groupby('Genre')['Count'].sum().nlargest(8).index
            
            genre_df_filtered = genre_df[genre_df['Genre'].isin(top_genres)]
            
            fig = px.line(
                genre_df_filtered,
                x='Year',
                y='Count',
                color='Genre',
                title="Top Genres Evolution Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

