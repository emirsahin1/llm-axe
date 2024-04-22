from llm_axe.agents import OnlineAgent
from llm_axe.models import OllamaChat

# Example showing how to use an online agent
# The online agent will use the internet to try and best answer the user prompt
prompt = "Tell me a bit about this website:  https://toscrape.com/?"
llm = OllamaChat(model="llama3:instruct")
searcher = OnlineAgent(llm)
resp = searcher.search(prompt)
print(resp)

   
# You may provide the OnlineAgent with a custom searcher
# The searcher must take in a search query and return a list of string URLS
# example hard coded searcher:
def internet_search(query):
    return ["https://toscrape.com/", "https://toscrape.com/"]
searcher = OnlineAgent(llm, custom_searcher=internet_search)
resp = searcher.search(prompt)
print(resp)

