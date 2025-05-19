import os
from dotenv import load_dotenv
from euriai import EuriaiClient
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")


EURI_CLIENT = EuriaiClient(
    api_key =os.getenv("EURI_API_KEY"),
    model = "gpt-4.1-nano"
)
