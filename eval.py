import json

def read_evals():
    evals = []

    with open('evals/practicalai.json', 'r') as file:
        data = json.load(file)
        return data['evals']

def run_evals(evals, query_engine):
    for eval in evals:
        question = eval['q']
        response = query_engine.query(question)
        print("==========")
        print(f"Question: {question}")
        print(f"Actual response: {response}")
        print(f"Reference response: {eval['a']}")


def run(query_engine):
    evals = read_evals()
    run_evals(evals, query_engine)