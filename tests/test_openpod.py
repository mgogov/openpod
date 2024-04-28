import pytest
import argparse
from openpod import setup_argparser

def test_setup_argparser():
    parser = setup_argparser()
    assert isinstance(parser, argparse.ArgumentParser)

    assert parser.get_default('chat') == False
    assert parser.get_default('reindex') == False
    assert parser.get_default('llm') == 'openai-gpt-3.5-turbo'
    assert parser.get_default('eval') == False
    assert parser.get_default('eval_id') == '<default_id>'
    assert parser.get_default('eval_reset_db') == False

@pytest.mark.parametrize("valid_value", ["openai-gpt-3.5-turbo", "openai-gpt-4"])
def test_llm_arg_valid_values(valid_value):
    parser = setup_argparser()
    args = parser.parse_args(['--llm', valid_value])

@pytest.mark.parametrize("invalid_value", ["llama-5-mega-turbo-500B", "closedai-skynet-20-10T"])
def test_llm_arg_invalid_values(invalid_value):
    parser = setup_argparser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(['--llm', invalid_value])