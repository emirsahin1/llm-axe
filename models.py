from ollama import Client

class OllamaChat():
    def __init__(self, host:str="http://localhost:11434", model:str=None):

        if model is None:
            raise ValueError('''You must provide a model to use OllamaChat. 
                                example: OllamaChat(model='llama3:instruct')''')

        self._host = host
        self._model = model
        self._ollama = Client(host='http://localhost:11434')


    def ask(self, prompts:list):
        return self._ollama.chat(model=self._model, messages=prompts)["message"]["content"]        

