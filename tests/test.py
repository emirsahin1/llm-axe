import unittest
from unittest.mock import MagicMock, patch
from llm_axe import Agent, AgentType, OnlineAgent, DataExtractor, PdfReader, FunctionCaller

class TestAgent(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM object
        self.llm_mock = MagicMock()

    def test_ask_with_prompt(self):
        mock_resp = "This is a response from the llm"
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = Agent(self.llm_mock, agent_type=AgentType.GENERIC_RESPONDER)
        response = agent.ask(prompt)
        
        self.assertEqual(response, mock_resp)
        self.assertEqual(agent.chat_history[0]["role"], "user")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertEqual(agent.chat_history[1]["content"], mock_resp)

    def test_ask_with_history(self):
        mock_resp = "This is a response from the llm"
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = Agent(self.llm_mock, agent_type=AgentType.GENERIC_RESPONDER)
        response = agent.ask(prompt, history=[{"role": "user", "content": "Hello"}])
        self.assertEqual(response, mock_resp)
        self.assertEqual(agent.chat_history[0]["role"], "user")
        self.assertEqual(agent.chat_history[0]["content"], "What is the meaning of life?")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertEqual(agent.chat_history[1]["content"], mock_resp)

    def test_get_prompt(self):
        prompt = "What is the meaning of life?"
        agent = Agent(self.llm_mock, agent_type=AgentType.GENERIC_RESPONDER)
        prompts = agent.get_prompt(prompt)
        self.assertEqual(prompts, [agent.system_prompt, {"role": "user", "content": prompt}])


class TestDataExtractor(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM object
        self.llm_mock = MagicMock()

    def test_ask(self):
        mock_resp = "This is a response from the llm"
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = DataExtractor(self.llm_mock)
        response = agent.ask(prompt, ["Name"])
        
        self.assertEqual(response, mock_resp)
        self.assertEqual(agent.chat_history[0]["role"], "user")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertEqual(agent.chat_history[1]["content"], mock_resp)

    def test_get_prompt(self):
        prompt = "What is the meaning of life?"
        agent = DataExtractor(self.llm_mock)
        prompts = agent.get_prompt(prompt, ["Name"])
        self.assertEqual(prompts[0]["content"], agent.system_prompt["content"])
        self.assertEqual(prompts[1]["role"], "user")

class TestPdfReader(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM object
        self.llm_mock = MagicMock()

    def test_ask(self):
        mock_resp = "This is a response from the llm"
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
        agent = PdfReader(self.llm_mock)
    
        with patch('llm_axe.agents.read_pdf') as read_pdf_mock:
            read_pdf_mock.return_value = " HELLLLO\n\n\n\n\n"
            response = agent.ask(prompt, ["pdf1.pdf"])
        
            self.assertEqual(response, mock_resp)
            self.assertEqual(agent.chat_history[0]["role"], "user")
            self.assertEqual(agent.chat_history[1]["role"], "assistant")

    def test_ask_with_history(self):
        mock_resp = "This is a response from the llm"
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = PdfReader(self.llm_mock)

        with patch('llm_axe.agents.read_pdf') as read_pdf_mock:
            read_pdf_mock.return_value = "Pdf output"
            response = agent.ask(prompt, ["pdf1.pdf"], history=[{"role": "user", "content": "Hello"}])
    
            self.assertEqual(response, mock_resp)
            self.assertEqual(agent.chat_history[0]["role"], "user")
            self.assertEqual(agent.chat_history[1]["role"], "assistant")


class TestFunctionCaller(unittest.TestCase):

    def setUp(self):
        # Create a mock LLM object
        self.llm_mock = MagicMock()

    def test_ask(self):
        mock_resp = '{"function": "setUp", "parameters": {"question": "What is the meaning of life?"}}'
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = FunctionCaller(self.llm_mock, [self.setUp])
        response = agent.get_function(prompt)

        self.assertEqual(response["function"], self.setUp)
        self.assertEqual(agent.chat_history[0]["role"], "user")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertEqual(agent.chat_history[1]["content"], mock_resp)

    def test_ask_with_history(self):
        mock_resp = '{"function": "setUp", "parameters": {"question": "What is the meaning of life?"}}'
        prompt = "What is the meaning of life?"
        self.llm_mock.ask.return_value = mock_resp
    
        agent = FunctionCaller(self.llm_mock, [self.setUp])

        response = agent.get_function(prompt, history=[{"role": "user", "content": "Hello"}])

        self.assertEqual(response["function"], self.setUp)
        self.assertEqual(agent.chat_history[0]["role"], "user")
        self.assertEqual(agent.chat_history[1]["role"], "assistant")
        self.assertEqual(agent.chat_history[1]["content"], mock_resp)

class TestOnlineAgent(unittest.TestCase):
    def setUp(self):
        # Create a mock LLM object
        self.llm_mock = MagicMock()

    def test_ask(self):
        mock_resp = '{"url": "This is a response from the llm"}'
        prompt = "What is the meaning of life?"

        self.llm_mock.ask.return_value = mock_resp
    
        with patch('llm_axe.agents.internet_search') as search_mock:
            with patch('llm_axe.agents.read_website') as read_mock:
                with patch('llm_axe.agents.OnlineAgent.get_search_query') as get_query_mock:
                    search_mock.return_value = {"url": "This is a response from the llm"}
                    read_mock.return_value = "Website Info"
                    get_query_mock.return_value = "url"

                    agent = OnlineAgent(self.llm_mock)
                    response = agent.search(prompt)

                    self.assertEqual(response, mock_resp)
                    self.assertEqual(agent.chat_history[0]["role"], "user")
                    self.assertEqual(agent.chat_history[1]["role"], "assistant")
                    self.assertEqual(agent.chat_history[2]["role"], "user")
                    self.assertEqual(agent.chat_history[3]["role"], "assistant")


if __name__ == '__main__':
    unittest.main()