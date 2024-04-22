from llm_axe.agents import PdfReader
from llm_axe.models import OllamaChat

llm = OllamaChat(model="llama3:instruct")

# We specify the files that we want the llm to be able to read.
# Note: The files should fit within your LLM's context window.
files = ["../FileOne.pdf", "../FileTwo.pdf"]
agent = PdfReader(llm)
resp = agent.ask("Extract all phone numbers found in any of these documents.", files)
print(resp)

