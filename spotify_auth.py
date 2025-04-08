import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from fastapi import HTTPException

load_dotenv()

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI')

if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
    raise Exception("Spotify credentials not found in .env file")

# Scopes for Spotify authorization
SCOPES = [
    'user-read-private',
    'user-read-email',
    'user-library-read',
    'playlist-read-private',
    'playlist-modify-public',
    'playlist-modify-private'
]

def get_spotify_oauth():
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID.strip('"'),
        client_secret=SPOTIFY_CLIENT_SECRET.strip('"'),
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=' '.join(SCOPES),
        open_browser=False
    )

def get_spotify_client(token_info):
    try:
        return spotipy.Spotify(auth=token_info['access_token'])
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid Spotify token")

def get_auth_url():
    try:
        sp_oauth = get_spotify_oauth()
        auth_url = sp_oauth.get_authorize_url()
        print(f"Generated auth URL: {auth_url}")
        return auth_url
    except Exception as e:
        print(f"Error generating auth URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate auth URL: {str(e)}")

def get_token_info(code):
    sp_oauth = get_spotify_oauth()
    try:
        token_info = sp_oauth.get_access_token(code, as_dict=True)
        return token_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to get access token: {str(e)}")

def refresh_token_if_expired(token_info):
    sp_oauth = get_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    return token_info 