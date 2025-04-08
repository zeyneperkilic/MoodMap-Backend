from itsdangerous import URLSafeSerializer
from fastapi import Request, Response

class SessionManager:
    def __init__(self, secret_key: str):
        self.serializer = URLSafeSerializer(secret_key)
        self.cookie_name = "session"
        self.sessions = {}

    def create_session(self, session_id, data):
        self.sessions[session_id] = data

    def get_session(self, request: Request) -> dict:
        session_id = request.cookies.get('session_id')
        if not session_id:
            print("No session cookie found")
            return {}
        try:
            data = self.sessions.get(session_id)
            if data:
                print("Session data:", data)
                return data
            else:
                print("Session data not found")
                return {}
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            return {}

    def set_session(self, response: Response, data: dict):
        try:
            session_data = self.serializer.dumps(data)
            response.set_cookie(
                key=self.cookie_name,
                value=session_data,
                httponly=True,
                samesite='lax',
                secure=False,  # Development için secure=False
                path="/"  # Tüm path'lerde geçerli
            )
            print("Session set successfully")
        except Exception as e:
            print(f"Error setting session: {str(e)}")

    def clear_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]

# Daha güvenli bir secret key kullanıyoruz
session_manager = SessionManager("moodmap-secret-key-2024") 