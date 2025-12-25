import os
from dotenv import load_dotenv

# Load variables from .env into Python
load_dotenv()


class Settings:
    ENVIRONMENT = os.getenv("ENVIRONMENT")
    STOCK_DATA_PROVIDER = os.getenv("STOCK_DATA_PROVIDER")

    # Gemini
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL")


settings = Settings()
