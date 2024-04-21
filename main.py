import util
import openai

openai.api_key = util.get_openai_api_key()

doc = util.summaries_doc()

print(doc.text)