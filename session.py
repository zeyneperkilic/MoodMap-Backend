import json
import os
from datetime import datetime, timedelta

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_session(self, request):
        session_id = request.cookies.get("session_id")
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return None

    def set_session(self, request, data):
        session_id = request.cookies.get("session_id") or os.urandom(16).hex()
        self.sessions[session_id] = data
        return session_id

    def clear_session(self, request):
        session_id = request.cookies.get("session_id")
        if session_id in self.sessions:
            del self.sessions[session_id]

# Daha güvenli bir secret key kullanıyoruz
session_manager = SessionManager() 