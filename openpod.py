import os
import argparse
import logging
import sys
import rag
import eval
import chat
import llm as llm_util

from llama_index.core import Document

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def setup_argparser():
    parser = argparse.ArgumentParser(prog='openpod', description='OpenPod command line interface.')

    parser.add_argument('--chat', action='store_true', help='Chat with the podcast')
    parser.add_argument('--reindex', action='store_true', help='Force reindexing of the podcast data')
    parser.add_argument('--llm', choices=[llm_util.Providers.OPENAI_GPT_35_TURBO.value, llm_util.Providers.OPENAI_GPT_4.value], default=llm_util.Providers.DEFAULT.value, help='Specify the LLM to use')
    parser.add_argument('--eval', action='store_true', help='Evaluate how well the LLM is answering the benchmark questions')
    parser.add_argument('--eval-id', action='store_true', help='The ID to be used for the evaluation', default="<default_id>")
    parser.add_argument('--eval-reset-db', action='store_true', help='Whether to reset the evaluation DB', default="true")

    return parser

def main():
    parser = setup_argparser()

    if len(sys.argv) == 1:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if not any([args.eval, args.chat, args.reindex]):
        print("Please specify at least one action to perform: one of --eval, --chat, or --reindex.")
        parser.exit(-1)

    if args.eval_id and not args.eval:
        print("--eval-id can only be specified if --eval is used")
        parser.exit(-1)

    if args.eval_reset_db and not args.eval:
        print("--eval-reset-db can only be specified if --eval is used")
        parser.exit(-1)

    if args.chat or args.reindex or args.eval:
        llm = llm_util.create(llm_util.Providers(args.llm))
        query_engine = rag.setup(rag.summaries_doc(), llm, reindex=args.reindex)

    if args.chat:
        chat.run(query_engine)

    if args.eval:
        eval.run(query_engine, id=args.eval_id)

if __name__ == "__main__":
    main()
