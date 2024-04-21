import util
import json
import os

from llama_index import Document

doc = util.summaries_doc()
node_parser = util.get_node_parser(doc.text)
llm = util.get_openai_llm()
nodes = node_parser.get_nodes_from_documents([Document(text=doc.text)])
context = util.get_service_context(llm, node_parser)
index = util.get_index(doc, context)
query_engine = util.get_query_engine(index)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# logging.basicConfig(level=logging.INFO)

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
