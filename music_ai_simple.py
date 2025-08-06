#!/usr/bin/env python3
"""
Simplified AI Inferencing Demonstrations with 2016 Top Songs Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def analyze_music_data():
    """Complete AI inferencing analysis of the music data"""
    print("🎵 AI INFERENCING WITH YOUR 2016 MUSIC DATA 🎵")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('/Users/gitspo/your_top_songs_2016.csv')
    print(f"Loaded {len(df)} songs from your 2016 collection")
    
    # Clean data
    audio_features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 
                     'Instrumentalness', 'Liveness', 'Speechiness']
    
    # Fill any missing values with column means
    for feature in audio_features:
        df[feature] = df[feature].fillna(df[feature].mean())
    
    print(f"\n=== YOUR MUSIC PREFERENCE PROFILE ===")
    for feature in audio_features:
        mean_val = df[feature].mean()
        print(f"{feature}: {mean_val:.3f}")
    
    # 1. CLUSTERING ANALYSIS
    print(f"\n=== AI DEMO 1: DISCOVERING YOUR MUSIC CLUSTERS ===")
    
    X = df[audio_features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    print(f"Your music falls into 3 distinct taste clusters:")
    for i in range(3):
        cluster_songs = df[df['Cluster'] == i]
        print(f"\nCluster {i+1}: {len(cluster_songs)} songs")
        print("Sample songs:")
        for _, song in cluster_songs.head(2).iterrows():
            print(f"  • {song['Track Name']} by {song['Artist Name(s)']}")
    
    # 2. MOOD CLASSIFICATION
    print(f"\n=== AI DEMO 2: MOOD CLASSIFICATION ===")
    
    # Create mood categories
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
    
    mood_counts = df['Mood'].value_counts()
    print(f"Your 2016 music mood distribution:")
    for mood, count in mood_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {mood}: {count} songs ({percentage:.1f}%)")
    
    # Train mood classifier
    X = df[audio_features]
    y = df['Mood']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_classifier.fit(X_train, y_train)
    accuracy = rf_classifier.score(X_test, y_test)
    print(f"AI Mood Classification Accuracy: {accuracy:.1%}")
    
    # 3. MUSIC PREFERENCE INSIGHTS
    print(f"\n=== AI DEMO 3: PREFERENCE INSIGHTS ===")
    
    # Analyze your preferences vs typical ranges
    insights = []
    
    if df['Danceability'].mean() > 0.6:
        insights.append("You prefer highly danceable music")
    elif df['Danceability'].mean() < 0.4:
        insights.append("You prefer less danceable, more contemplative music")
    
    if df['Energy'].mean() > 0.7:
        insights.append("You love high-energy music")
    elif df['Energy'].mean() < 0.5:
        insights.append("You prefer calmer, lower-energy music")
    
    if df['Valence'].mean() > 0.6:
        insights.append("Your music tends to be positive and uplifting")
    elif df['Valence'].mean() < 0.4:
        insights.append("You gravitate toward more melancholic music")
    
    if df['Acousticness'].mean() > 0.5:
        insights.append("You enjoy acoustic and organic sounds")
    else:
        insights.append("You prefer electronic/produced music")
    
    print("AI-discovered insights about your music taste:")
    for insight in insights:
        print(f"  • {insight}")
    
    # 4. CREATE VISUALIZATIONS
    print(f"\n=== CREATING VISUALIZATIONS ===")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('AI Analysis of Your 2016 Music Preferences', fontsize=16)
    
    # 1. Cluster visualization
    axes[0,0].scatter(df['Valence'], df['Energy'], c=df['Cluster'], cmap='viridis', alpha=0.7)
    axes[0,0].set_xlabel('Valence (Musical Positivity)')
    axes[0,0].set_ylabel('Energy')
    axes[0,0].set_title('Music Clusters')
    
    # 2. Mood distribution
    mood_counts.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Mood Distribution')
    axes[0,1].set_ylabel('Number of Songs')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Audio features radar
    features_mean = df[audio_features].mean()
    angles = np.linspace(0, 2*np.pi, len(audio_features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    values = np.concatenate((features_mean.values, [features_mean.values[0]]))
    
    axes[0,2].plot(angles, values, 'o-', linewidth=2)
    axes[0,2].fill(angles, values, alpha=0.25)
    axes[0,2].set_xticks(angles[:-1])
    axes[0,2].set_xticklabels([f.replace('ness', '') for f in audio_features])
    axes[0,2].set_title('Your Audio Feature Profile')
    axes[0,2].set_ylim(0, 1)
    
    # 4. Energy vs Danceability by mood
    for mood in df['Mood'].unique():
        mood_data = df[df['Mood'] == mood]
        axes[1,0].scatter(mood_data['Danceability'], mood_data['Energy'], 
                         label=mood, alpha=0.7)
    axes[1,0].set_xlabel('Danceability')
    axes[1,0].set_ylabel('Energy')
    axes[1,0].set_title('Energy vs Danceability by Mood')
    axes[1,0].legend()
    
    # 5. Popularity distribution
    axes[1,1].hist(df['Popularity'], bins=15, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Popularity Score')
    axes[1,1].set_ylabel('Number of Songs')
    axes[1,1].set_title('Song Popularity Distribution')
    
    # 6. Feature importance for mood prediction
    feature_importance = pd.DataFrame({
        'feature': audio_features,
        'importance': rf_classifier.feature_importances_
    }).sort_values('importance', ascending=True)
    
    axes[1,2].barh(feature_importance['feature'], feature_importance['importance'])
    axes[1,2].set_xlabel('Importance')
    axes[1,2].set_title('Features Important for Mood Prediction')
    
    plt.tight_layout()
    plt.savefig('/Users/gitspo/complete_music_ai_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. RECOMMENDATION SIMULATION
    print(f"\n=== AI DEMO 4: RECOMMENDATION SYSTEM ===")
    
    # Calculate your preference profile
    user_profile = df[audio_features].mean()
    
    # Simulate recommendations by finding songs most similar to your profile
    similarities = []
    for idx, song in df.iterrows():
        song_features = song[audio_features].values
        user_features = user_profile.values
        
        # Calculate simple distance-based similarity
        distance = np.sqrt(np.sum((song_features - user_features) ** 2))
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        similarities.append(similarity)
    
    df['Similarity'] = similarities
    top_matches = df.nlargest(5, 'Similarity')
    
    print("Songs most similar to your overall 2016 preferences:")
    for i, (_, song) in enumerate(top_matches.iterrows(), 1):
        print(f"  {i}. {song['Track Name']} by {song['Artist Name(s)']}")
        print(f"     Similarity: {song['Similarity']:.3f}")
    
    print(f"\n" + "=" * 60)
    print("✅ AI INFERENCING COMPLETE!")
    print("Generated: complete_music_ai_analysis.png")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    df = analyze_music_data()

