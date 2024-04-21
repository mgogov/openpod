import util
import json

from llama_index import Document

doc = util.summaries_doc()
node_parser = util.get_node_parser(doc.text)
llm = util.get_openai_llm()
nodes = node_parser.get_nodes_from_documents([Document(text=doc.text)])
context = util.get_service_context(llm, node_parser)
index = util.get_index(doc, context)