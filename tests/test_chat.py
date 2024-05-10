from unittest.mock import patch
from chat import run

@patch('builtins.input', side_effect=["What is the capital of France?", "exit"])
def test_run_with_valid_question(mock_input):
    class MockQueryEngine:
        def query(self, question):
            if question == "What is the capital of France?":
                return "Of course it's Pernik!"
    
    query_engine = MockQueryEngine()
    
    with patch('builtins.print') as mock_print:
        run(query_engine)
        mock_print.assert_called_with("Of course it's Pernik!")

@patch('builtins.input', side_effect=["exit"])
def test_run_with_exit_command(mock_input):
    class MockQueryEngine:
        def query(self, question):
            return "This should not be printed."
  
    query_engine = MockQueryEngine()
  
    with patch('builtins.print') as mock_print:
        run(query_engine)
        mock_print.assert_not_called()

def test_run_with_keyboard_interrupt():
    class MockQueryEngine:
        def query(self, question):
            return "This should not be printed."
  
    query_engine = MockQueryEngine()
  
    with patch('builtins.print') as mock_print:
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            run(query_engine)
            mock_print.assert_not_called()