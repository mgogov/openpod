import os
import openai
import glob
from dotenv import load_dotenv, find_dotenv
from llama_index import Document

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

from llama_index import SimpleDirectoryReader

def summaries_doc():
    reader = SimpleDirectoryReader(
        input_files=glob.glob("./data/practicalai/summaries/gpt4-podcast-summary-pro/*.md")
    )

    docs = reader.load_data()
    full_doc = Document(text="\n\n".join([doc.text for doc in docs]))

    return full_doc