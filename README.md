# openpod
## Description
Talk to your podcast transcripts.

## Installation
To install and run this project, follow these steps:

1. [Install Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) in order  to install reqired dependencies.
2. Clone the repository: `git clone https://github.com/mgogov/openpod.git`
3. Navigate to the project directory: `cd openpod`
4. Create a mamba environment from `requirements.txt` and activate it:

```
mamba create -n openpod
mamba activate openpod
pip install -r requirements.txt
```

## Available LLMs and related setup
Currently only OpenAI's `gpt-3.5-turbo` and `gpt-4` are available.

Provide a `.env` with the following content in the home dir:
```
OPENAI_API_KEY=<your_API_key>
```

## Usage
```
$ python openpod.py usage: openpod [-h] [--chat] [--reindex] [--llm {openai-gpt-3.5-turbo,openai-gpt-4}] [--eval] [--eval-id EVAL_ID]
               [--eval-reset-db]

OpenPod command line interface.

options:
  -h, --help            show this help message and exit
  --chat                Chat with the podcast
  --reindex             Force reindexing of the podcast data
  --llm {openai-gpt-3.5-turbo,openai-gpt-4}
                        Specify the LLM to use
  --eval                Evaluate how well the LLM is answering the benchmark questions
  --eval-id EVAL_ID     The ID to be used for the evaluation
  --eval-reset-db       Whether to reset the evaluation DB

```

## Contributing
Contributions are welcome! If you would like to contribute to this project, just open a pull request.

## License
This project is licensed under the [Apache2 License](LICENSE).

## Contact
If you have any questions or suggestions, send me an email (check out my profile).