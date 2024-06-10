from llm_axe import Agent, AgentType

# To use llm-axe with your own LLM, all you need is a class with an ask function
# Example:
class MyCustomLLM():

    # Your ask function will always receive a list of prompts
    # The prompts are in open ai prompt format
    #  example: {"role": "system", "content": "You are a helpful assistant."}
    # If your model supports json format, use the format parameter to specify that to your model.
    def ask(self, prompts:list, format:str="", temperature:float=0.8):
        """
        Args:
            prompts (list): A list of prompts to ask.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        return "Your llms response to the prompts goes here!"        



# ------------------------------------------------------------------------------
# example usage:
llm = MyCustomLLM()
agent = Agent(llm, agent_type=AgentType.GENERIC_RESPONDER)
print(agent.ask("Hi how are you today?"))
