def run(query_engine):
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