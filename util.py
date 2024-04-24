import os
import openai

from dotenv import load_dotenv, find_dotenv

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

def get_openai_llm():
    openai.api_key = get_openai_api_key()
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    return llm