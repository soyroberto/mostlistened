#!/usr/bin/env python3
"""
Last.fm API Integration Module - Fixed for Classical Music
Provides functions to get similar artists and tracks from Last.fm
FIXED: Handles comma-separated artist names for classical music
"""

import requests
import json
import time
from typing import List, Dict, Optional
import streamlit as st
import re

class LastFMAPI:
    """Last.fm API client for music recommendations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://ws.audioscrobbler.com/2.0/"
        self.session = requests.Session()
        
    def _make_request(self, method: str, params: Dict) -> Optional[Dict]:
        """Make a request to the Last.fm API with error handling and rate limiting"""
        params.update({
            'method': method,
            'api_key': self.api_key,
            'format': 'json'
        })
        
        try:
            # Rate limiting - Last.fm allows 5 requests per second
            time.sleep(0.2)
            
            response = self.session.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if 'error' in data:
                st.warning(f"Last.fm API Error: {data.get('message', 'Unknown error')}")
                return None
                
            return data
            
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse response: {str(e)}")
            return None
    
    def get_similar_artists(self, artist_name: str, limit: int = 10) -> List[Dict]:
        """Get similar artists from Last.fm"""
        params = {
            'artist': artist_name,
            'limit': limit,
            'autocorrect': 1
        }
        
        data = self._make_request('artist.getsimilar', params)
        
        if not data or 'similarartists' not in data:
            return []
        
        similar_artists = []
        artists_data = data['similarartists'].get('artist', [])
        
        # Handle case where only one artist is returned (not in a list)
        if isinstance(artists_data, dict):
            artists_data = [artists_data]
        
        for artist in artists_data:
            similar_artists.append({
                'name': artist.get('name', ''),
                'match': float(artist.get('match', 0)),
                'url': artist.get('url', ''),
                'mbid': artist.get('mbid', '')
            })
        
        return similar_artists
    
    def get_similar_tracks(self, artist_name: str, track_name: str, limit: int = 10) -> List[Dict]:
        """Get similar tracks from Last.fm"""
        params = {
            'artist': artist_name,
            'track': track_name,
            'limit': limit,
            'autocorrect': 1
        }
        
        data = self._make_request('track.getsimilar', params)
        
        if not data or 'similartracks' not in data:
            return []
        
        similar_tracks = []
        tracks_data = data['similartracks'].get('track', [])
        
        # Handle case where only one track is returned (not in a list)
        if isinstance(tracks_data, dict):
            tracks_data = [tracks_data]
        
        for track in tracks_data:
            artist_info = track.get('artist', {})
            artist_name = artist_info.get('name', '') if isinstance(artist_info, dict) else str(artist_info)
            
            similar_tracks.append({
                'name': track.get('name', ''),
                'artist': artist_name,
                'match': float(track.get('match', 0)),
                'url': track.get('url', ''),
                'mbid': track.get('mbid', '')
            })
        
        return similar_tracks
    
    def get_artist_info(self, artist_name: str) -> Optional[Dict]:
        """Get detailed artist information from Last.fm"""
        params = {
            'artist': artist_name,
            'autocorrect': 1
        }
        
        data = self._make_request('artist.getinfo', params)
        
        if not data or 'artist' not in data:
            return None
        
        artist = data['artist']
        
        return {
            'name': artist.get('name', ''),
            'mbid': artist.get('mbid', ''),
            'url': artist.get('url', ''),
            'bio': artist.get('bio', {}).get('summary', ''),
            'tags': [tag.get('name', '') for tag in artist.get('tags', {}).get('tag', [])],
            'similar': [sim.get('name', '') for sim in artist.get('similar', {}).get('artist', [])]
        }
    
    def get_track_info(self, artist_name: str, track_name: str) -> Optional[Dict]:
        """Get detailed track information from Last.fm"""
        params = {
            'artist': artist_name,
            'track': track_name,
            'autocorrect': 1
        }
        
        data = self._make_request('track.getinfo', params)
        
        if not data or 'track' not in data:
            return None
        
        track = data['track']
        
        return {
            'name': track.get('name', ''),
            'artist': track.get('artist', {}).get('name', ''),
            'mbid': track.get('mbid', ''),
            'url': track.get('url', ''),
            'duration': track.get('duration', ''),
            'tags': [tag.get('name', '') for tag in track.get('toptags', {}).get('tag', [])],
            'wiki': track.get('wiki', {}).get('summary', '')
        }
    
    def get_top_tracks_by_artist(self, artist_name: str, limit: int = 10) -> List[Dict]:
        """Get top tracks by an artist from Last.fm"""
        params = {
            'artist': artist_name,
            'limit': limit,
            'autocorrect': 1
        }
        
        data = self._make_request('artist.gettoptracks', params)
        
        if not data or 'toptracks' not in data:
            return []
        
        top_tracks = []
        tracks_data = data['toptracks'].get('track', [])
        
        # Handle case where only one track is returned (not in a list)
        if isinstance(tracks_data, dict):
            tracks_data = [tracks_data]
        
        for track in tracks_data:
            top_tracks.append({
                'name': track.get('name', ''),
                'artist': track.get('artist', {}).get('name', ''),
                'playcount': int(track.get('playcount', 0)),
                'url': track.get('url', ''),
                'mbid': track.get('mbid', '')
            })
        
        return top_tracks

def split_artist_names(artist_string: str) -> List[str]:
    """
    FIXED: Split comma-separated artist names into individual artists
    
    Handles classical music cases like:
    - "Ludwig van Beethoven,Berliner Philharmoniker,Herbert von Karajan"
    - "Johann Sebastian Bach,Zino Francescatti,Lucerne Festival Strings,Rudolf Baumgartner"
    
    Args:
        artist_string: Comma-separated artist names
        
    Returns:
        List of individual artist names, cleaned and filtered
    """
    if not artist_string:
        return []
    
    # Split by comma and clean each artist name
    artists = [artist.strip() for artist in artist_string.split(',')]
    
    # Filter out empty strings and very short names (likely not real artist names)
    artists = [artist for artist in artists if len(artist) > 2]
    
    # For classical music, prioritize composer names (usually first) and well-known performers
    # Common classical music patterns to prioritize
    classical_composers = [
        'bach', 'beethoven', 'mozart', 'chopin', 'brahms', 'tchaikovsky', 
        'vivaldi', 'handel', 'haydn', 'schubert', 'schumann', 'liszt',
        'debussy', 'ravel', 'stravinsky', 'prokofiev', 'rachmaninoff',
        'mendelssohn', 'grieg', 'sibelius', 'dvorak', 'wagner', 'verdi',
        'puccini', 'rossini', 'donizetti', 'bellini', 'monteverdi',
        'purcell', 'palestrina', 'victoria', 'hildegard', 'caldara'
    ]
    
    # Prioritize composers and solo performers over orchestras/ensembles
    prioritized_artists = []
    other_artists = []
    
    for artist in artists:
        artist_lower = artist.lower()
        
        # Check if it's a composer
        is_composer = any(composer in artist_lower for composer in classical_composers)
        
        # Check if it's likely an orchestra/ensemble (contains certain keywords)
        orchestra_keywords = ['orchestra', 'philharmonic', 'symphony', 'ensemble', 'choir', 'chor', 'quartet', 'quintet']
        is_ensemble = any(keyword in artist_lower for keyword in orchestra_keywords)
        
        if is_composer:
            prioritized_artists.insert(0, artist)  # Composers first
        elif not is_ensemble:
            prioritized_artists.append(artist)  # Solo performers next
        else:
            other_artists.append(artist)  # Orchestras/ensembles last
    
    # Combine prioritized artists with others, limit to avoid too many API calls
    final_artists = prioritized_artists + other_artists
    
    # Return top 3 artists to balance API efficiency with coverage
    return final_artists[:3]

def generate_lastfm_recommendations(lastfm_api: LastFMAPI, user_artists: List[str], 
                                  user_tracks: List[tuple], max_recommendations: int = 20) -> Dict:
    """
    FIXED: Generate recommendations using Last.fm API with comma-separated artist handling
    
    Args:
        lastfm_api: LastFMAPI instance
        user_artists: List of artist names from user's library (may contain comma-separated names)
        user_tracks: List of (artist, track) tuples from user's library
        max_recommendations: Maximum number of recommendations to return
    
    Returns:
        Dictionary with artist and track recommendations
    """
    
    recommendations = {
        'similar_artists': [],
        'similar_tracks': [],
        'top_tracks_from_similar_artists': []
    }
    
    # FIXED: Split comma-separated artist names before processing
    expanded_artists = []
    for artist_string in user_artists[:10]:  # Limit initial artists to avoid too many API calls
        individual_artists = split_artist_names(artist_string)
        expanded_artists.extend(individual_artists)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_artists = []
    for artist in expanded_artists:
        if artist not in seen:
            seen.add(artist)
            unique_artists.append(artist)
    
    # Limit to top artists to manage API rate limits
    final_artists = unique_artists[:8]
    
    # Get similar artists based on user's top artists
    st.write("🔍 Finding similar artists...")
    progress_bar = st.progress(0)
    
    for i, artist in enumerate(final_artists):
        try:
            similar_artists = lastfm_api.get_similar_artists(artist, limit=5)
            if similar_artists:  # Only add if we got results
                recommendations['similar_artists'].extend(similar_artists)
            else:
                st.write(f"   No similar artists found for: {artist}")
        except Exception as e:
            st.write(f"   Error getting similar artists for {artist}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(final_artists))
    
    # Remove duplicates and sort by match score
    seen_artists = set()
    unique_similar_artists = []
    for artist in recommendations['similar_artists']:
        if artist['name'] not in seen_artists and artist['name'] not in final_artists:
            seen_artists.add(artist['name'])
            unique_similar_artists.append(artist)
    
    recommendations['similar_artists'] = sorted(
        unique_similar_artists, 
        key=lambda x: x['match'], 
        reverse=True
    )[:max_recommendations]
    
    # FIXED: Handle comma-separated artists in track recommendations
    st.write("🎵 Finding similar tracks...")
    progress_bar = st.progress(0)
    
    processed_tracks = []
    for artist_string, track in user_tracks[:8]:  # Limit to avoid rate limits
        # Split artist names and try with the primary artist (usually first/composer)
        individual_artists = split_artist_names(artist_string)
        if individual_artists:
            # Use the first (most important) artist for track similarity
            primary_artist = individual_artists[0]
            processed_tracks.append((primary_artist, track))
    
    for i, (artist, track) in enumerate(processed_tracks):
        try:
            similar_tracks = lastfm_api.get_similar_tracks(artist, track, limit=3)
            if similar_tracks:
                recommendations['similar_tracks'].extend(similar_tracks)
            else:
                st.write(f"   No similar tracks found for: {track} by {artist}")
        except Exception as e:
            st.write(f"   Error getting similar tracks for {track} by {artist}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(processed_tracks))
    
    # Remove duplicates and sort by match score
    seen_tracks = set()
    unique_similar_tracks = []
    user_track_set = set(f"{artist}|{track}" for artist, track in processed_tracks)
    
    for track in recommendations['similar_tracks']:
        track_key = f"{track['artist']}|{track['name']}"
        if track_key not in seen_tracks and track_key not in user_track_set:
            seen_tracks.add(track_key)
            unique_similar_tracks.append(track)
    
    recommendations['similar_tracks'] = sorted(
        unique_similar_tracks, 
        key=lambda x: x['match'], 
        reverse=True
    )[:max_recommendations]
    
    # Get top tracks from similar artists
    st.write("🎤 Getting top tracks from similar artists...")
    progress_bar = st.progress(0)
    
    for i, artist_info in enumerate(recommendations['similar_artists'][:5]):  # Top 5 similar artists
        try:
            top_tracks = lastfm_api.get_top_tracks_by_artist(artist_info['name'], limit=3)
            for track in top_tracks:
                track['similarity_score'] = artist_info['match']  # Inherit similarity from artist
            recommendations['top_tracks_from_similar_artists'].extend(top_tracks)
        except Exception as e:
            st.write(f"   Error getting top tracks for {artist_info['name']}: {str(e)}")
        
        progress_bar.progress((i + 1) / min(5, len(recommendations['similar_artists'])))
    
    # Sort by artist similarity and track popularity
    recommendations['top_tracks_from_similar_artists'] = sorted(
        recommendations['top_tracks_from_similar_artists'],
        key=lambda x: (x.get('similarity_score', 0), x.get('playcount', 0)),
        reverse=True
    )[:max_recommendations]
    
    progress_bar.empty()
    
    # Show summary of what was processed
    st.success(f"✅ Processed {len(final_artists)} individual artists from {len(user_artists)} original entries")
    
    return recommendations

def display_lastfm_recommendations(recommendations: Dict):
    """Display Last.fm recommendations in Streamlit"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎤 Similar Artists")
        if recommendations['similar_artists']:
            for i, artist in enumerate(recommendations['similar_artists'][:10], 1):
                st.write(f"**{i}. {artist['name']}**")
                st.write(f"   Similarity: {artist['match']:.3f}")
                if artist['url']:
                    st.write(f"   [Last.fm Profile]({artist['url']})")
                st.write("")
        else:
            st.write("No similar artists found.")
    
    with col2:
        st.subheader("🎵 Similar Tracks")
        if recommendations['similar_tracks']:
            for i, track in enumerate(recommendations['similar_tracks'][:10], 1):
                st.write(f"**{i}. {track['name']}**")
                st.write(f"   by {track['artist']}")
                st.write(f"   Similarity: {track['match']:.3f}")
                if track['url']:
                    st.write(f"   [Last.fm Page]({track['url']})")
                st.write("")
        else:
            st.write("No similar tracks found.")
    
    with col3:
        st.subheader("🔥 Popular Tracks from Similar Artists")
        if recommendations['top_tracks_from_similar_artists']:
            for i, track in enumerate(recommendations['top_tracks_from_similar_artists'][:10], 1):
                st.write(f"**{i}. {track['name']}**")
                st.write(f"   by {track['artist']}")
                st.write(f"   Plays: {track['playcount']:,}")
                if track['url']:
                    st.write(f"   [Last.fm Page]({track['url']})")
                st.write("")
        else:
            st.write("No tracks found.")

# Test function for API key validation
def test_lastfm_api(api_key: str) -> bool:
    """Test if the Last.fm API key is valid"""
    try:
        api = LastFMAPI(api_key)
        # Test with a simple request
        result = api.get_similar_artists("The Beatles", limit=1)
        return len(result) > 0
    except Exception:
        return False

