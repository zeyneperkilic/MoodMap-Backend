from fastapi import FastAPI, Request, HTTPException, Depends, Query
import pandas as pd
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import uvicorn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware
import logging
import random
import os
from spotify_auth import get_auth_url, get_token_info, get_spotify_client, refresh_token_if_expired
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from session import session_manager
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from session import SessionManager
import requests

app = FastAPI()

# Frontend dizinini belirt
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
templates_dir = os.path.join(frontend_dir, "templates")

# Statik dosyaları ve template'leri ayarla
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Güvenlik için production'da spesifik domainler belirtilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "processed_data.csv")
data = pd.read_csv(file_path)

features = ['danceability', 'energy', 'valence', 'tempo']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_scaled)
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]
data['PCA3'] = pca_result[:, 2]

feedback_file = "feedback.json"

try:
    with open(feedback_file, "r") as f:
        feedback_data = json.load(f)
except:
    feedback_data = {}

logging.basicConfig(filename='recommendation_check.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_weighted_check(cluster_id, intensity, songs):
    try:
        ascending = intensity < 5
        scores = []
        for song in songs:
            score = 0.4 * song.get('energy', 0) + 0.3 * song.get('valence', 0) + 0.2 * song.get('danceability', 0) + 0.1 * song.get('tempo', 0)
            scores.append(score)
        sorted_scores = sorted(scores, reverse=not ascending)

        if scores == sorted_scores:
            logging.info(f"Cluster {cluster_id}, intensity={intensity} weighted score correct.")
        else:
            logging.error(f"Cluster {cluster_id}, intensity={intensity} weighted score not correct.")
    except Exception as e:
        logging.error(f"Weighted control error: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("cover.html", {"request": request})

@app.get("/mood_selection", response_class=HTMLResponse)
async def mood_selection(request: Request):
    return templates.TemplateResponse("mood_selection.html", {"request": request})

@app.get("/mood_map", response_class=HTMLResponse)
async def mood_map(request: Request):
    return templates.TemplateResponse("mood_map.html", {"request": request})

@app.get("/feedback")
def get_feedback():
    return feedback_data

@app.get("/clusters/{cluster_id}")
def get_cluster_songs(cluster_id: int):
    cluster_songs = data[data["Cluster"] == cluster_id]
    if cluster_songs.empty:
        return {"songs": []}
    return {"songs": cluster_songs.to_dict(orient="records")}

@app.get("/recommend/{cluster_id}")
def recommend_songs(cluster_id: int, intensity: int = 5):
    try:
        cluster_songs = data[data["Cluster"] == cluster_id].copy()
        if cluster_songs.empty:
            return {"cluster": cluster_id, "songs": []}

        
        if cluster_id == 0:
            cluster_songs = cluster_songs[
                (cluster_songs['energy'] < 0.7) &
                (cluster_songs['valence'] < 0.7) &
                (cluster_songs['tempo'] < 115) &
                (cluster_songs['danceability'] < 0.7)
            ]
        
        else:
            min_tempo = 60 + (intensity * 8)   
            max_tempo = 180 - (intensity * 3)
            min_energy = intensity * 0.1
            max_energy = min(1.0, 0.6 + (intensity * 0.05))

            cluster_songs = cluster_songs[
                (cluster_songs['tempo'] >= min_tempo) &
                (cluster_songs['tempo'] <= max_tempo) &
                (cluster_songs['energy'] >= min_energy) &
                (cluster_songs['energy'] <= max_energy)
            ]

        ascending = intensity < 5
        cluster_songs['weighted_score'] = (
            0.4 * cluster_songs['energy'] +
            0.3 * cluster_songs['valence'] +
            0.2 * cluster_songs['danceability'] +
            0.1 * cluster_songs['tempo']
        )

        cluster_songs = cluster_songs.sort_values(by='weighted_score', ascending=ascending)
        top_songs = cluster_songs.head(50)
        recommendations = top_songs.sample(n=min(10, len(top_songs)))[
            ["uri", "PCA1", "PCA2", "PCA3", "energy", "valence", "tempo", "danceability"]
        ].to_dict(orient="records")

        log_weighted_check(cluster_id, intensity, recommendations)
        return {"cluster": cluster_id, "songs": recommendations}

    except Exception as e:
        print(f"Error in recommend_songs: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/feedback")
async def save_feedback(request: Request):
    data = await request.json()
    song = data.get("song", "Unknown")
    liked = data.get("liked")
    comment = data.get("comment", "")
    cluster_id = data.get("cluster_id")
    intensity = data.get("intensity")
    weighted_score = data.get("weighted_score")

    if song not in feedback_data:
        feedback_data[song] = {
            "likes": 0,
            "dislikes": 0,
            "comments": [],
            "sentiment": 0,
            "cluster_id": cluster_id,
            "intensity": intensity,
            "weighted_score": weighted_score
        }
    else:
        feedback_data[song]["cluster_id"] = cluster_id
        feedback_data[song]["intensity"] = intensity
        feedback_data[song]["weighted_score"] = weighted_score

    if liked is True:
        feedback_data[song]["likes"] += 1
    elif liked is False:
        feedback_data[song]["dislikes"] += 1

    if comment:
        feedback_data[song]["comments"].append(comment)
        feedback_data[song]["sentiment"] = analyze_sentiment_vader(comment)

    with open(feedback_file, "w") as f:
        json.dump(feedback_data, f, indent=4)

    return {"message": f"Feedback received for {song}"}

def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

@app.get("/spotify/login")
async def spotify_login():
    try:
        auth_url = get_auth_url()
        print(f"Generated Spotify auth URL: {auth_url}")
        return RedirectResponse(url=auth_url, status_code=303)
    except Exception as e:
        print(f"Error in spotify_login: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/callback")
async def spotify_callback(code: str = Query(None), request: Request = None):
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")
    
    try:
        token_info = get_token_info(code)
        print(f"Received token info: {token_info}")
        
        # Session'ı ayarla
        response = RedirectResponse(url="http://127.0.0.1:5001/mood-selection", status_code=303)
        session_manager.set_session(response, {"token_info": token_info})
        
        return response
    except Exception as e:
        print(f"Callback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/spotify/me")
async def get_spotify_user(request: Request):
    session_data = session_manager.get_session(request)
    token_info = session_data.get("token_info")
    
    if not token_info:
        raise HTTPException(status_code=401, detail="No token info in session")
    
    try:
        token_info = refresh_token_if_expired(token_info)
        response = JSONResponse(content={"message": "Success"})
        session_manager.set_session(response, {"token_info": token_info})
        
        sp = get_spotify_client(token_info)
        user_info = sp.current_user()
        return response
    except Exception as e:
        print(f"Error getting user info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Renk ve duygu eşleştirmeleri
COLOR_MOOD_MAPPING = {
    "red": {
        "moods": ["energetic", "passionate", "intense"],
        "description": "High energy and intense emotions",
        "genres": ["rock", "metal", "punk", "electronic"],
        "name": "Energetic"
    },
    "black": {
        "moods": ["mysterious", "deep", "emotional"],
        "description": "Deep and introspective feelings",
        "genres": ["alternative", "indie", "dark jazz", "ambient"],
        "name": "Sad"
    },
    "yellow": {
        "moods": ["happy", "upbeat", "cheerful"],
        "description": "Bright and positive vibes",
        "genres": ["pop", "dance", "funk", "disco"],
        "name": "Happy"
    },
    "green": {
        "moods": ["peaceful", "calm", "relaxed"],
        "description": "Peaceful and harmonious state",
        "genres": ["classical", "jazz", "acoustic", "ambient"],
        "name": "Calm"
    }
}

@app.get("/mood-map/{color}")
def get_mood_map(color: str):
    color_to_cluster = {
        "black": 0,  # Sad
        "yellow": 1, # Happy
        "red": 2,    # Energetic
        "green": 3   # Calm
    }
    
    cluster_id = color_to_cluster.get(color.lower())
    if cluster_id is None:
        raise HTTPException(status_code=404, detail="Invalid mood color")
    
    mood_names = {
        "black": "Sad",
        "yellow": "Happy",
        "red": "Energetic",
        "green": "Calm"
    }
    
    cluster_songs = data[data["Cluster"] == cluster_id].copy()
    songs = cluster_songs[["uri", "PCA1", "PCA2", "PCA3", "energy", "valence", "tempo", "danceability"]].to_dict(orient="records")
    
    return {
        "mood_name": mood_names.get(color.lower(), "Unknown"),
        "mood_color": color.lower(),
        "cluster_id": cluster_id,
        "all_songs": songs
    }

@app.get("/recommendations/{color}")
async def get_recommendations(color: str, request: Request):
    if color not in COLOR_MOOD_MAPPING:
        raise HTTPException(status_code=404, detail="Color not found")
    
    session_data = session_manager.get_session(request)
    token_info = session_data.get("token_info")
    
    if token_info:
        try:
            token_info = refresh_token_if_expired(token_info)
            sp = get_spotify_client(token_info)
            
            # Renk için uygun türleri al
            genres = COLOR_MOOD_MAPPING[color]["genres"]
            
            # Spotify'dan önerileri al
            recommendations = sp.recommendations(
                seed_genres=genres[:5],  # En fazla 5 tür kullanabiliriz
                limit=10,
                target_energy=0.8 if color == "red" else 0.6 if color == "yellow" else 0.4,
                target_valence=0.8 if color in ["yellow", "green"] else 0.4
            )
            
            # Önerileri formatlayıp döndür
            tracks = []
            for track in recommendations["tracks"]:
                tracks.append({
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "album": track["album"]["name"],
                    "preview_url": track["preview_url"],
                    "external_url": track["external_urls"]["spotify"]
                })
            
            return {"tracks": tracks}
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Spotify girişi yapılmamışsa, varsayılan öneriler
        return {
            "tracks": [
                {
                    "name": "Please login with Spotify",
                    "artist": "to get personalized recommendations",
                    "album": "",
                    "preview_url": None,
                    "external_url": None
                }
            ]
        }

@app.post("/create-playlist")
async def create_playlist(request: Request):
    try:
        # Get session data
        session_data = session_manager.get_session(request)
        if not session_data:
            raise HTTPException(status_code=401, detail="Not authenticated")

        # Get token info
        token_info = session_data.get('token_info')
        if not token_info:
            raise HTTPException(status_code=401, detail="No token information found")

        # Check if token needs refresh
        token_info = refresh_token_if_expired(token_info)
        
        # Get request data
        data = await request.json()
        playlist_name = data.get('name')
        track_uris = data.get('tracks')

        if not playlist_name or not track_uris:
            raise HTTPException(status_code=400, detail="Missing playlist name or tracks")

        # Create Spotify client
        sp = get_spotify_client(token_info['access_token'])
        
        # Get user ID
        user_info = sp.current_user()
        user_id = user_info['id']

        # Create playlist
        playlist = sp.user_playlist_create(user_id, playlist_name, public=False)
        playlist_id = playlist['id']

        # Add tracks to playlist
        sp.playlist_add_items(playlist_id, track_uris)

        return {"success": True, "playlist_id": playlist_id}

    except Exception as e:
        print(f"Error creating playlist: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/logout")
async def logout(request: Request):
    try:
        session_id = request.cookies.get('session_id')
        if session_id:
            session_manager.clear_session(session_id)
        return {"success": True}
    except Exception as e:
        print(f"Error during logout: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Environment variables
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "b10eaf728ba24184ae191fe5dd193197")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "b10eaf728ba24184ae191fe5dd193197")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8000/callback")

# Session manager
session_manager = SessionManager()

# Spotify OAuth
sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope="user-read-private user-read-email user-library-read playlist-read-private playlist-modify-public playlist-modify-private"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
