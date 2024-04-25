import glob
import logging
import os

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import Document, ServiceContext, VectorStoreIndex, StorageContext, SimpleDirectoryReader, load_index_from_storage
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank

def summaries_doc():
    reader = SimpleDirectoryReader(
        input_files=glob.glob("./data/practicalai/summaries/gpt4-podcast-summary-pro/*.md")
    )

    docs = reader.load_data()
    full_doc = Document(text="\n\n".join([doc.text for doc in docs]))

    return full_doc

def get_node_parser(text):
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

def get_query_engine(index, postprocessors=[]):
    postprocessor = MetadataReplacementPostProcessor(
        target_metadata_key="window"
    )

    reranker = SentenceTransformerRerank(
        top_n=2, model="BAAI/bge-reranker-base"
    )
    engine = index.as_query_engine(
        similarity_top_k=6, node_postprocessors=[postprocessor, reranker]
    )

    return engine

def setup(document, llm, reindex=False):
    node_parser = get_node_parser(document.text)
    nodes = node_parser.get_nodes_from_documents([Document(text=document.text)])
    context = get_service_context(llm, node_parser)
    index = get_index(document, context, reindex=reindex)
    query_engine = get_query_engine(index)

    return query_engine