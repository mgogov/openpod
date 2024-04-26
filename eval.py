import json
import nest_asyncio
import numpy as np

from trulens_eval import Tru, Feedback, TruLlama, OpenAI
from trulens_eval.feedback import Groundedness

nest_asyncio.apply()

def get_trulens_recorder(query_engine, id):
    openai = OpenAI()

    qa_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    #grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
    grounded = Groundedness(groundedness_provider=openai)

    groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
            .on(TruLlama.select_source_nodes().node.text)
            .on_output()
            .aggregate(grounded.grounded_statements_aggregator)
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]

    tru_recorder = TruLlama(
        query_engine,
        app_id=id,
        feedbacks=feedbacks
    )

    return tru_recorder

def read_evals():
    evals = []

    with open('evals/practicalai.json', 'r') as file:
        data = json.load(file)
        return data['evals']

def run_evals(evals, query_engine, id):
    tru_recorder = get_trulens_recorder(query_engine, id)

    for eval in evals:
        question = eval['q']
        with tru_recorder as recording:
            response = query_engine.query(question)
            print("==========")
            print(f"Question: {question}")
            print(f"Actual response: {response}")
            print(f"Reference response: {eval['a']}")

def run(query_engine, reset_eval_db=False, id="<default>"):
    evals = read_evals()

    if reset_eval_db:
        Tru().reset_database()

    run_evals(evals, query_engine, id=id)
