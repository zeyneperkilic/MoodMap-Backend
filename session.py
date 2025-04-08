import json
import os
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.session_file = "sessions.json"
        self.load_sessions()

    def load_sessions(self):
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, "r") as f:
                    self.sessions = json.load(f)
        except Exception as e:
            print(f"Error loading sessions: {e}")
            self.sessions = {}

    def save_sessions(self):
        try:
            with open(self.session_file, "w") as f:
                json.dump(self.sessions, f)
        except Exception as e:
            print(f"Error saving sessions: {e}")

    def get_session(self, request):
        session_id = request.cookies.get("session_id")
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return None

    def set_session(self, request, response, data):
        session_id = request.cookies.get("session_id") or os.urandom(16).hex()
        self.sessions[session_id] = data
        self.save_sessions()
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=True,
            samesite="lax"
        )
        return session_id

    def clear_session(self, request, response):
        session_id = request.cookies.get("session_id")
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.save_sessions()
        response.delete_cookie("session_id")

# Daha güvenli bir secret key kullanıyoruz
session_manager = SessionManager() 