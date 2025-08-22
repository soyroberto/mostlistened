#!/usr/bin/env python3
"""
Streamlined iTunes Music Preview System
Optimized for integration with Last.fm recommendations
Enhanced with search history, favorites, and batch operations
"""
import streamlit as st
import requests
import urllib.parse
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import re

# Page configuration
st.set_page_config(
    page_title="🎵 Music Preview Hub",
    page_icon="🎵",
    layout="wide"
)

class MusicPreviewEngine:
    """Streamlined music preview engine optimized for Last.fm integration"""
    
    def __init__(self):
        self.base_url = "https://itunes.apple.com/search"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MusicPreviewApp/1.0'
        })
    
    def quick_search(self, term: str, limit: int = 10) -> List[Dict]:
        """Quick search optimized for Last.fm integration"""
        params = {
            "term": term,
            "media": "music",
            "entity": "song",
            "limit": limit,
            "country": "US"
        }
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            return response.json().get("results", [])
        except:
            return []
    
    def precise_artist_search(self, artist_name: str, exclude_bands: List[str] = None) -> List[Dict]:
        """
        Precise artist search with band exclusion
        Perfect for distinguishing solo work from band work
        """
        exclude_bands = exclude_bands or []
        
        # Search for the artist
        results = self.quick_search(f'"{artist_name}"', limit=30)
        
        # Filter out excluded bands
        filtered_results = []
        for result in results:
            result_artist = result.get("artistName", "").lower()
            
            # Check if this result should be excluded
            should_exclude = False
            for band in exclude_bands:
                if band.lower() in result_artist:
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_results.append(result)
        
        return filtered_results[:10]  # Limit to top 10
    
    def batch_search(self, terms: List[str]) -> Dict[str, List[Dict]]:
        """Batch search for multiple terms (useful for Last.fm recommendations)"""
        results = {}
        for term in terms:
            results[term] = self.quick_search(term, limit=5)
        return results
    
    def search_by_song_and_artist(self, song: str, artist: str) -> List[Dict]:
        """Search for specific song by specific artist"""
        search_term = f"{song} {artist}"
        results = self.quick_search(search_term, limit=20)
        
        # Filter to exact matches
        exact_matches = []
        for result in results:
            if (artist.lower() in result.get("artistName", "").lower() and
                song.lower() in result.get("trackName", "").lower()):
                exact_matches.append(result)
        
        return exact_matches[:5]

class SearchHistory:
    """Manage search history and favorites"""
    
    @staticmethod
    def add_to_history(term: str, mode: str):
        """Add search to history"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        # Add new search to beginning of list
        search_entry = {
            'term': term,
            'mode': mode,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        # Remove duplicates and add to front
        st.session_state.search_history = [
            search_entry
        ] + [h for h in st.session_state.search_history if h['term'] != term]
        
        # Keep only last 20 searches
        st.session_state.search_history = st.session_state.search_history[:20]
    
    @staticmethod
    def get_history() -> List[Dict]:
        """Get search history"""
        return st.session_state.get('search_history', [])
    
    @staticmethod
    def add_favorite(track_info: Dict):
        """Add track to favorites"""
        if 'favorites' not in st.session_state:
            st.session_state.favorites = []
        
        # Check if already in favorites
        track_id = track_info.get('trackId')
        if not any(fav.get('trackId') == track_id for fav in st.session_state.favorites):
            st.session_state.favorites.append(track_info)
    
    @staticmethod
    def remove_favorite(track_id: int):
        """Remove track from favorites"""
        if 'favorites' in st.session_state:
            st.session_state.favorites = [
                fav for fav in st.session_state.favorites 
                if fav.get('trackId') != track_id
            ]
    
    @staticmethod
    def get_favorites() -> List[Dict]:
        """Get favorites list"""
        return st.session_state.get('favorites', [])

def display_compact_interface():
    """Compact interface optimized for integration"""
    st.markdown("### 🎵 Music Preview Hub")
    
    # Quick search bar
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_term = st.text_input(
            "Search",
            placeholder="Artist, song, or album...",
            label_visibility="collapsed",
            key="main_search_input"
        )
    
    with col2:
        search_mode = st.selectbox(
            "Mode",
            ["Quick", "Precise", "Song+Artist"],
            label_visibility="collapsed",
            key="main_search_mode"
        )
    
    with col3:
        search_button = st.button("🔍 Search", type="primary", key="main_search_button")
    
    return search_term, search_mode, search_button

def display_advanced_search():
    """Advanced search options"""
    with st.expander("🎛️ Advanced Search"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Precise Artist Search**")
            artist_name = st.text_input(
                "Artist Name", 
                placeholder="e.g., Gustavo Cerati",
                key="precise_artist_name"
            )
            exclude_bands = st.text_input(
                "Exclude Bands", 
                placeholder="e.g., Soda Stereo, Virus",
                help="Comma-separated list of bands to exclude",
                key="exclude_bands_input"
            )
            
            if st.button("🎤 Search Artist Only", key="search_artist_only_button"):
                if artist_name:
                    exclude_list = [band.strip() for band in exclude_bands.split(",") if band.strip()]
                    return "precise_artist", artist_name, exclude_list
        
        with col2:
            st.markdown("**Song + Artist Search**")
            song_name = st.text_input(
                "Song Name", 
                placeholder="e.g., Puente",
                key="song_name_input"
            )
            song_artist = st.text_input(
                "Artist Name", 
                placeholder="e.g., Gustavo Cerati",
                key="song_artist_name"
            )
            
            if st.button("🎵 Search Song", key="search_song_button"):
                if song_name and song_artist:
                    return "song_artist", f"{song_name}|{song_artist}", []
    
    return None, None, None

def display_search_history():
    """Display search history sidebar"""
    history = SearchHistory.get_history()
    
    if history:
        st.sidebar.markdown("### 📜 Recent Searches")
        for i, search in enumerate(history[:5]):
            if st.sidebar.button(
                f"🔍 {search['term'][:20]}{'...' if len(search['term']) > 20 else ''}",
                key=f"history_{i}",
                help=f"{search['mode']} - {search['timestamp']}"
            ):
                st.session_state.quick_search = search['term']
                st.session_state.quick_mode = search['mode']
                st.rerun()

def display_favorites():
    """Display favorites sidebar"""
    favorites = SearchHistory.get_favorites()
    
    if favorites:
        st.sidebar.markdown("### ⭐ Favorites")
        for i, fav in enumerate(favorites[:3]):
            track_name = fav.get('trackName', 'Unknown')[:15]
            artist_name = fav.get('artistName', 'Unknown')[:15]
            
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(f"**{track_name}**<br>*{artist_name}*", unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_fav_{i}", help="Remove from favorites"):
                    SearchHistory.remove_favorite(fav.get('trackId'))
                    st.rerun()
            
            if fav.get('previewUrl'):
                st.sidebar.audio(fav['previewUrl'])

def display_compact_results(results: List[Dict], max_results: int = 5):
    """Compact results display"""
    if not results:
        st.info("No results found")
        return
    
    st.markdown(f"**Found {len(results)} results**")
    
    for i, track in enumerate(results[:max_results]):
        track_name = track.get("trackName", "Unknown")
        artist_name = track.get("artistName", "Unknown")
        preview_url = track.get("previewUrl")
        artwork_url = track.get("artworkUrl60", "")
        
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
            
            with col1:
                if artwork_url:
                    st.image(artwork_url, width=50)
            
            with col2:
                st.markdown(f"**{track_name}**")
                st.markdown(f"*{artist_name}*")
            
            with col3:
                if preview_url:
                    st.audio(preview_url)
                else:
                    st.markdown("No preview")
            
            with col4:
                if st.button("⭐", key=f"fav_{i}", help="Add to favorites"):
                    SearchHistory.add_favorite(track)
                    st.success("Added!")
            
            st.divider()

def lastfm_integration_helper(recommendations: List[str]) -> Dict[str, List[Dict]]:
    """Helper function for Last.fm integration"""
    engine = MusicPreviewEngine()
    return engine.batch_search(recommendations)

def main():
    """Main streamlined application"""
    
    # Initialize
    engine = MusicPreviewEngine()
    
    # Sidebar
    display_search_history()
    display_favorites()
    
    # Main interface
    search_term, search_mode, search_button = display_compact_interface()
    
    # Handle session state quick search
    if hasattr(st.session_state, 'quick_search'):
        search_term = st.session_state.quick_search
        search_mode = st.session_state.get('quick_mode', 'Quick')
        search_button = True
        # Clear session state
        del st.session_state.quick_search
        if hasattr(st.session_state, 'quick_mode'):
            del st.session_state.quick_mode
    
    # Advanced search
    advanced_mode, advanced_term, advanced_params = display_advanced_search()
    
    # Process search
    if search_button and search_term:
        SearchHistory.add_to_history(search_term, search_mode)
        
        with st.spinner("Searching..."):
            if search_mode == "Quick":
                results = engine.quick_search(search_term)
            elif search_mode == "Precise":
                results = engine.precise_artist_search(search_term)
            elif search_mode == "Song+Artist":
                parts = search_term.split()
                if len(parts) >= 2:
                    song = parts[0]
                    artist = " ".join(parts[1:])
                    results = engine.search_by_song_and_artist(song, artist)
                else:
                    results = engine.quick_search(search_term)
            else:
                results = engine.quick_search(search_term)
        
        display_compact_results(results)
    
    elif advanced_mode:
        if advanced_mode == "precise_artist":
            with st.spinner("Searching artist..."):
                results = engine.precise_artist_search(advanced_term, advanced_params)
            display_compact_results(results)
        
        elif advanced_mode == "song_artist":
            song, artist = advanced_term.split("|")
            with st.spinner("Searching song..."):
                results = engine.search_by_song_and_artist(song, artist)
            display_compact_results(results)
    
    # Integration examples
    with st.expander("🔗 Last.fm Integration Examples"):
        st.markdown("""
        **For Last.fm Integration:**
        ```python
        from itunes_streamlined import lastfm_integration_helper
        
        # Get previews for Last.fm recommendations
        recommendations = ["Artist 1", "Artist 2", "Artist 3"]
        previews = lastfm_integration_helper(recommendations)
        ```
        
        **Precise Artist Search:**
        ```python
        # Search only Gustavo Cerati (exclude Soda Stereo)
        engine = MusicPreviewEngine()
        results = engine.precise_artist_search("Gustavo Cerati", ["Soda Stereo"])
        ```
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**🎵 Music Preview Hub** | Ready for Last.fm Integration | By @soyroberto")

if __name__ == "__main__":
    main()

