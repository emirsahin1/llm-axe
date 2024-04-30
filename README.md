
# llm-axe 🪓

llm-axe is a handy little axe for developing llm powered applications. 

It allows you to quickly implement complex interactions for local LLMs, such as function callers, online agents, pre-made generic agents, and more.


## Installation



```bash
pip install llm-axe
```
    
## Example Snippets

- **Function Calling**

&emsp;&emsp;A function calling LLM can be created with just **3 lines of code**:
<br>
&emsp;&emsp;No need for premade schemas, templates, special prompts, or specialized functions.
```python
prompt = "I have 500 coins, I just got 200 more. How many do I have?"

llm = OllamaChat(model="llama3:instruct")
fc = FunctionCaller(llm, [get_time, get_date, get_location, add, multiply])
result = fc.get_function(prompt)
```
- **Online Agent**
```python
prompt = "Tell me a bit about this website:  https://toscrape.com/?"
llm = OllamaChat(model="llama3:instruct")
searcher = OnlineAgent(llm)
resp = searcher.search(prompt)

#output: Based on information from the internet, it appears that https://toscrape.com/ is a website dedicated to web scraping.
# It provides a sandbox environment for beginners and developers to learn and validate their web scraping technologies...
```
- **PDF Reader**
```python
llm = OllamaChat(model="llama3:instruct")
files = ["../FileOne.pdf", "../FileTwo.pdf"]
agent = PdfReader(llm)
resp = agent.ask("Summarize these documents for me", files)
```

- **Data Extractor**
```python
llm = OllamaChat(model="llama3:instruct")
info = read_pdf("../Example.pdf")
de = DataExtractor(llm, reply_as_json=True)
resp = de.ask(info, ["name", "email", "phone", "address"])

#output: {'Name': 'Frodo Baggins', 'Email': 'frodo@gmail.com', 'Phone': '555-555-5555', 'Address': 'Bag-End, Hobbiton, The Shire'}
```
[**See more complete examples**](https://github.com/emirsahin1/llm-axe/tree/main/examples)

[**How to setup llm-axe with your own LLM**](https://github.com/emirsahin1/llm-axe/blob/main/examples/ex_llm_setup.py)


## Features

- Local LLM internet access with Online Agent
- PDF Document Reader Agent
- Premade utility Agents for common tasks
- Compatible with any LLM, local or externally hosted
- Built-in support for Ollama



## Important Notes

The results you get from the agents are highly dependent on the capability of your LLM. An inadequate LLM will not be able to provide results that are usable with llm-axe

**Testing in development was done using llama3 8b:instruct 4 bit quant**
