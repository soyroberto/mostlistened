#!/usr/bin/env python3
"""
AI Inferencing Demonstrations with 2016 Top Songs Data
This script demonstrates various AI/ML techniques on personal music data
Not GUI based, runs in terminal
python3 music_ai_demo.py
Best to run in a virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data(file_path):
    """Load the CSV data and perform basic exploration"""
    print("=== LOADING AND EXPLORING YOUR MUSIC DATA ===")
    
    # Load data
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape[0]} songs, {df.shape[1]} features")
    
    # Basic info
    print(f"\nDate range: {df['Added At'].min()} to {df['Added At'].max()}")
    print(f"Unique artists: {df['Artist Name(s)'].nunique()}")
    print(f"Unique genres: {df['Genres'].nunique()}")
    
    # Audio features summary
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness', 'Tempo']
    
    print(f"\n=== YOUR MUSIC PREFERENCE PROFILE ===")
    for feature in audio_features:
        mean_val = df[feature].mean()
        print(f"{feature}: {mean_val:.3f} (0.0 = low, 1.0 = high)")
    
    return df

def demonstrate_clustering(df):
    """Demonstrate unsupervised learning - clustering your music taste"""
    print(f"\n=== AI INFERENCING DEMO 1: MUSIC TASTE CLUSTERING ===")
    print("Using K-means clustering to discover hidden patterns in your music preferences...")
    
    # Select audio features for clustering
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    # Prepare data
    X = df[audio_features].fillna(df[audio_features].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    # Analyze clusters
    print(f"\nYour music falls into 3 distinct preference clusters:")
    for i in range(3):
        cluster_songs = df_clustered[df_clustered['Cluster'] == i]
        print(f"\n--- CLUSTER {i+1} ({len(cluster_songs)} songs) ---")
        
        # Show representative songs
        print("Representative songs:")
        for _, song in cluster_songs.head(3).iterrows():
            print(f"  • {song['Track Name']} by {song['Artist Name(s)']}")
        
        # Show cluster characteristics
        print("Cluster characteristics:")
        for feature in audio_features:
            avg = cluster_songs[feature].mean()
            print(f"  {feature}: {avg:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title('Your Music Clusters (PCA Visualization)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.colorbar(scatter)
    
    # Feature comparison across clusters
    plt.subplot(2, 2, 2)
    cluster_means = df_clustered.groupby('Cluster')[audio_features].mean()
    cluster_means.T.plot(kind='bar', ax=plt.gca())
    plt.title('Audio Features by Cluster')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster')
    
    # Energy vs Danceability
    plt.subplot(2, 2, 3)
    for i in range(3):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        plt.scatter(cluster_data['Danceability'], cluster_data['Energy'], 
                   label=f'Cluster {i+1}', alpha=0.7)
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.title('Energy vs Danceability by Cluster')
    plt.legend()
    
    # Valence vs Acousticness
    plt.subplot(2, 2, 4)
    for i in range(3):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        plt.scatter(cluster_data['Acousticness'], cluster_data['Valence'], 
                   label=f'Cluster {i+1}', alpha=0.7)
    plt.xlabel('Acousticness')
    plt.ylabel('Valence (Musical Positivity)')
    plt.title('Valence vs Acousticness by Cluster')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/music_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_clustered

def demonstrate_popularity_prediction(df):
    """Demonstrate supervised learning - predicting song popularity"""
    print(f"\n=== AI INFERENCING DEMO 2: POPULARITY PREDICTION ===")
    print("Training AI model to predict song popularity based on audio features...")
    
    # Prepare features and target
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness', 'Tempo', 'Loudness']
    
    X = df[audio_features].fillna(df[audio_features].mean())
    y = df['Popularity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_regressor.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  R² Score: {r2:.3f} (1.0 = perfect prediction)")
    print(f"  Mean Squared Error: {mse:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': audio_features,
        'importance': rf_regressor.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nMost important features for predicting popularity:")
    for _, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Popularity')
    plt.ylabel('Predicted Popularity')
    plt.title(f'Popularity Prediction (R² = {r2:.3f})')
    
    plt.subplot(1, 2, 2)
    feature_importance.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
    plt.title('Feature Importance for Popularity Prediction')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/popularity_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rf_regressor, feature_importance

def demonstrate_mood_classification(df):
    """Demonstrate classification - mood prediction based on audio features"""
    print(f"\n=== AI INFERENCING DEMO 3: MOOD CLASSIFICATION ===")
    print("Creating AI model to classify song moods based on audio characteristics...")
    
    # Create mood categories based on valence and energy
    def classify_mood(row):
        valence = row['Valence']
        energy = row['Energy']
        
        if valence >= 0.6 and energy >= 0.6:
            return 'Happy/Energetic'
        elif valence >= 0.6 and energy < 0.6:
            return 'Happy/Calm'
        elif valence < 0.4 and energy >= 0.6:
            return 'Intense/Aggressive'
        else:
            return 'Sad/Mellow'
    
    df['Mood'] = df.apply(classify_mood, axis=1)
    
    # Show mood distribution
    mood_counts = df['Mood'].value_counts()
    print(f"\nYour 2016 music mood distribution:")
    for mood, count in mood_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {mood}: {count} songs ({percentage:.1f}%)")
    
    # Prepare features for classification
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    X = df[audio_features].fillna(df[audio_features].mean())
    y = df['Mood']
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluate
    accuracy = rf_classifier.score(X_test, y_test)
    print(f"\nMood Classification Accuracy: {accuracy:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Mood distribution
    plt.subplot(2, 2, 1)
    mood_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Your Music Mood Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Songs')
    
    # Valence vs Energy colored by mood
    plt.subplot(2, 2, 2)
    for mood in df['Mood'].unique():
        mood_data = df[df['Mood'] == mood]
        plt.scatter(mood_data['Valence'], mood_data['Energy'], 
                   label=mood, alpha=0.7)
    plt.xlabel('Valence (Musical Positivity)')
    plt.ylabel('Energy')
    plt.title('Songs by Mood (Valence vs Energy)')
    plt.legend()
    
    # Feature importance for mood classification
    plt.subplot(2, 2, 3)
    mood_importance = pd.DataFrame({
        'feature': audio_features,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mood_importance.plot(x='feature', y='importance', kind='bar', ax=plt.gca())
    plt.title('Feature Importance for Mood Classification')
    plt.xticks(rotation=45)
    plt.ylabel('Importance')
    
    # Average features by mood
    plt.subplot(2, 2, 4)
    mood_features = df.groupby('Mood')[['Valence', 'Energy', 'Danceability', 'Acousticness']].mean()
    mood_features.plot(kind='bar', ax=plt.gca())
    plt.title('Average Audio Features by Mood')
    plt.xticks(rotation=45)
    plt.ylabel('Feature Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/mood_classification.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rf_classifier

def create_recommendation_system(df, df_clustered):
    """Demonstrate recommendation system based on user preferences"""
    print(f"\n=== AI INFERENCING DEMO 4: MUSIC RECOMMENDATION SYSTEM ===")
    print("Building AI recommendation system based on your 2016 preferences...")
    
    # Calculate your preference profile
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    user_profile = df[audio_features].mean()
    print(f"\nYour Musical Preference Profile:")
    for feature, value in user_profile.items():
        print(f"  {feature}: {value:.3f}")
    
    # Simulate some "new" songs (using existing songs as examples)
    # In real application, these would be songs not in your library
    np.random.seed(42)
    sample_indices = np.random.choice(df.index, size=10, replace=False)
    new_songs = df.loc[sample_indices].copy()
    
    # Calculate similarity scores
    def calculate_similarity(song_features, user_profile):
        """Calculate cosine similarity between song and user profile"""
        song_vector = song_features[audio_features].values
        user_vector = user_profile.values
        
        # Handle any NaN values
        if np.any(np.isnan(song_vector)):
            song_vector = np.nan_to_num(song_vector, nan=user_vector)
        
        # Calculate cosine similarity
        dot_product = np.dot(song_vector, user_vector)
        norms = np.linalg.norm(song_vector) * np.linalg.norm(user_vector)
        
        if norms == 0:
            return 0
        
        return dot_product / norms
    
    # Calculate recommendation scores
    recommendation_scores = []
    for idx, song in new_songs.iterrows():
        similarity = calculate_similarity(song, user_profile)
        recommendation_scores.append({
            'Track Name': song['Track Name'],
            'Artist': song['Artist Name(s)'],
            'Similarity Score': similarity,
            'Predicted Rating': similarity * 5  # Scale to 1-5 rating
        })
    
    # Sort by similarity
    recommendations = pd.DataFrame(recommendation_scores).sort_values('Similarity Score', ascending=False)
    
    print(f"\nTop 5 Recommended Songs (based on your 2016 preferences):")
    for i, (_, rec) in enumerate(recommendations.head(5).iterrows(), 1):
        print(f"  {i}. {rec['Track Name']} by {rec['Artist']}")
        print(f"     Similarity Score: {rec['Similarity Score']:.3f}")
        print(f"     Predicted Rating: {rec['Predicted Rating']:.1f}/5.0")
        print()
    
    return recommendations

def main():
    """Main function to run all AI inferencing demonstrations"""
    print("🎵 AI INFERENCING WITH YOUR 2016 MUSIC DATA 🎵")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data('/home/ubuntu/upload/your_top_songs_2016.csv')
    
    # Demonstration 1: Clustering
    df_clustered = demonstrate_clustering(df)
    
    # Demonstration 2: Popularity Prediction
    popularity_model, feature_importance = demonstrate_popularity_prediction(df)
    
    # Demonstration 3: Mood Classification
    mood_model = demonstrate_mood_classification(df)
    
    # Demonstration 4: Recommendation System
    recommendations = create_recommendation_system(df, df_clustered)
    
    print(f"\n" + "=" * 60)
    print("🎯 AI INFERENCING COMPLETE!")
    print("Generated visualizations:")
    print("  • music_clustering_analysis.png - Your music taste clusters")
    print("  • popularity_prediction.png - Popularity prediction model")
    print("  • mood_classification.png - Mood classification analysis")
    print("=" * 60)

if __name__ == "__main__":
    main()

