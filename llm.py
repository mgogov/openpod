from enum import Enum
import os
from dotenv import load_dotenv, find_dotenv

import openai
from llama_index.llms.openai import OpenAI

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

class Providers(Enum):
    OPENAI_GPT_35_TURBO = "openai-gpt-3.5-turbo"
    OPENAI_GPT_4 = "openai-gpt-4"

    DEFAULT = OPENAI_GPT_35_TURBO

def create(llm):
    if llm == Providers.OPENAI_GPT_35_TURBO:
        openai.api_key = get_openai_api_key()
        return OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    elif llm == Providers.OPENAI_GPT_4:
        openai.api_key = get_openai_api_key()
        return OpenAI(model="gpt-4", temperature=0.1)
    else:
        raise ValueError(f"Unknown LLM provider: {llm}")