import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask session secret; override with environment variable in production
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret')

    # Use SQLite for development if MySQL is not available
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        None  # Will be set in app.py if MySQL fails
    )
    
    if DATABASE_URL:
        SQLALCHEMY_DATABASE_URI = DATABASE_URL
    else:
        # Default to SQLite for development
        SQLALCHEMY_DATABASE_URI = 'sqlite:///student_management.db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = True