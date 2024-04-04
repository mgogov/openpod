from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

print(f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY')}")

