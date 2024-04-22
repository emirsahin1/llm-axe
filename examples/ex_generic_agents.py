from llm_axe.core import AgentType
from llm_axe.agents import Agent
from llm_axe.models import OllamaChat

 
llm = OllamaChat(model="llama3:instruct")

# Example showing how to use a generic premade agents
prompt = "I have 500 coins, I lost half of them. How many coins do I have now?"

# Different agents can be created using the AgentType enum
# PLANNER, SUMMARIZER, GENERIC_RESPONDER
#Simple example showing how to use different agent roles together

# We create a planner agent, whose role is to provide a plan to the system.
planner = Agent(llm, agent_type=AgentType.PLANNER)
resp = planner.ask(prompt)
print(resp)

# We can then make a generic responder agent, whose role is to give a nice response to the user prompt
generic_responder = Agent(llm, agent_type=AgentType.GENERIC_RESPONDER)
resp = generic_responder.ask(resp)
print(resp)

