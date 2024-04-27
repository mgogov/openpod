import pytest
from llm import create, Providers

def test_create_with_openai_gpt_35_turbo():
    llm = Providers.OPENAI_GPT_35_TURBO
    result = create(llm)
    assert result.model == "gpt-3.5-turbo"
    assert result.temperature == 0.1

def test_create_with_openai_gpt_4():
    llm = Providers.OPENAI_GPT_4
    result = create(llm)
    assert result.model == "gpt-4"
    assert result.temperature == 0.1

def test_create_with_unknown_provider():
    llm = "genisys"
    with pytest.raises(ValueError):
        create(llm)