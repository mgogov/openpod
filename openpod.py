import util
import json
import os
import argparse
import logging

from llama_index.core import Document

# set debug level to debug
logging.basicConfig(level=logging.DEBUG)

def setup_rag(reindex=False):
    doc = util.summaries_doc()
    node_parser = util.get_node_parser(doc.text)
    llm = util.get_openai_llm()
    nodes = node_parser.get_nodes_from_documents([Document(text=doc.text)])
    context = util.get_service_context(llm, node_parser)
    index = util.get_index(doc, context, reindex=reindex)
    query_engine = util.get_query_engine(index)

    return query_engine

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def chat(reindex=False):
    query_engine = setup_rag(reindex=reindex)

    # loop over questions from stdin and answer them
    while True:
        question = input("Ask a question: ")
        result = query_engine.query(question)
        print(result)

        # break if the user types "exit"
        if question == "exit":
            break

        # break if the user presses ctrl+c
        try:
            pass
        except KeyboardInterrupt:
            break

def setup_argparser():
    parser = argparse.ArgumentParser(prog='openpod', description='OpenPod command line interface.')
    # parser.add_argument('--help', action='help', help='Show this help message')
    parser.add_argument('--eval', action='store_true', help='Evaluate how well the LLM is answering the benchmark questions')
    parser.add_argument('--chat', action='store_true', help='Chat with the podcast')
    parser.add_argument('--reindex', action='store_true', help='Force reindexing of the podcast data')

    return parser

def main():
    parser = setup_argparser()
    args = parser.parse_args()

    # if no arguments are passed, show the help message
    if not any(vars(args).values()):
        parser.print_help()
        parser.exit()

    if args.chat:
        chat(reindex=args.reindex)

    if args.eval:
        # TODO implement
        pass

if __name__ == "__main__":
    main()
