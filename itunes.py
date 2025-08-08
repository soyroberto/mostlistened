import streamlit as st
import requests
# Song preview from Apple Music
# This script allows users to search for songs on Apple Music and listen to previews
st.title("👩🏻‍🎤 Music Previewer powered by Apple 👩🏻‍🎤")
st.markdown("Search for songs and listen to 30-second previews!")

search_term = st.text_input("Search for a song or artist", "Everything But the Girl")

if st.button("Search"):
    url = f"https://itunes.apple.com/search?term={search_term}&media=music&limit=10"
    response = requests.get(url)
    
    if response.status_code == 200:
        results = response.json().get("results", [])
        
        if not results:
            st.warning("No results found.")
        
        for i, track in enumerate(results):
            name = track.get("trackName")
            artist = track.get("artistName")
            preview = track.get("previewUrl")

            st.markdown(f"### {i+1}. {name} by *{artist}*")
            if preview:
                st.audio(preview)
            else:
                st.write("❌ No preview available")
    else:
        st.error("Failed to fetch data from Apple Music")
