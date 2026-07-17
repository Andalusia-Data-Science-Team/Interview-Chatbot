import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "****************************")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat")

SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "*****************")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "***********************")

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
NUM_QUESTIONS = 5
OUTPUT_DIR = "results"
DB_PATH = "interviews.db"

os.makedirs(OUTPUT_DIR, exist_ok=True)
