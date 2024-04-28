from llm_axe import Agent, AgentType

# To use llm-axe with your own LLM, all you need is a class with an ask function
# Example:
class MyCustomLLM():

    # Your ask function will always receive a list of prompts
    # The prompts are in open ai prompt format
    #  example: {"role": "system", "content": "You are a helpful assistant."}
    def ask(self, prompts:list):
        return "Your llms response to the prompts goes here!"        

# example usage:
llm = MyCustomLLM()
fc = Agent(llm, agent_type=AgentType.GENERIC_RESPONDER)
fc.ask("Hi how are you today?")