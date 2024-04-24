import util
import rag
import os
import argparse
import logging
import sys

from llama_index.core import Document

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def chat(reindex=False):
    llm = rag.get_openai_llm(util.get_openai_api_key())
    query_engine = rag.setup(rag.summaries_doc(), llm, reindex=reindex)

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
