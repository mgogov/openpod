def run(query_engine):
    try:
        while True:
            question = input("Ask a question: ")
            if question == "exit":
                break

            result = query_engine.query(question)
            print(result)
    except KeyboardInterrupt:
        return
