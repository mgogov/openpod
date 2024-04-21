import os
import openai
import glob
import logging

from dotenv import load_dotenv, find_dotenv
from llama_index import Document
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index import ServiceContext
from llama_index.llms import OpenAI

logging.basicConfig(level=logging.INFO)

def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")

def get_openai_llm():
    openai.api_key = get_openai_api_key()
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

    return llm

from llama_index import SimpleDirectoryReader

def summaries_doc():
    reader = SimpleDirectoryReader(
        input_files=glob.glob("./data/practicalai/summaries/gpt4-podcast-summary-pro/*.md")
    )

    docs = reader.load_data()
    full_doc = Document(text="\n\n".join([doc.text for doc in docs]))

    return full_doc

def get_node_parser(text):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    return node_parser

def get_service_context(llm, node_parser):
    context = ServiceContext.from_defaults(
        llm=llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        node_parser=node_parser,
    )

    return context

def get_index(doc, service_context, reindex=False):
    if reindex or not os.path.exists("./index_storage"):
        logging.info("Reindexing...")

        index = VectorStoreIndex.from_documents(
            [doc], service_context=service_context
        )

        index.storage_context.persist(persist_dir="./index_storage")

        logging.info("Index created and saved.")
    else:
        logging.info("Loading index from storage...")

        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./index_storage"),
            service_context=service_context
        )

        logging.info("Index loaded.")

    return index