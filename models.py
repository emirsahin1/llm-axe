from ollama import Client

class OllamaChat():
    def __init__(self, host:str="http://localhost:11434", model:str=None):

        if model is None:
            raise ValueError('''You must provide a model to use OllamaChat. 
                                example: OllamaChat(model='llama3:instruct')''')

        self._host = host
        self._model = model
        self._ollama = Client(host)


    def ask(self, prompts:list, format:str=""):
        return self._ollama.chat(model=self._model, messages=prompts, format=format)["message"]["content"]        

