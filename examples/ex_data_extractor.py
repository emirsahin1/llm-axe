from llm_axe.core import read_pdf, safe_read_json
from llm_axe.models import OllamaChat
from llm_axe.agents import DataExtractor


llm = OllamaChat(model="llama3:instruct")
info = read_pdf("../Example.pdf")

# It will reply in proper json since we set reply_as_json to True
de = DataExtractor(llm, reply_as_json=True)

resp = de.ask(info, ["name", "email", "phone", "address"])
print(resp)

# We can then convert to a proper python object if we wish
resp_json = safe_read_json(resp)
print(resp_json)