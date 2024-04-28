
# llm-axe 🪓

llm-axe is a handy little axe for developing llm powered applications. 

It allows you to quickly implement complex interactions for local LLMs, such as function callers, online agents, pre-made generic agents, and more.


## Installation



```bash
pip install llm-axe
```
    
## Example

A function calling LLM can be created with just **3 lines of code**:

```python
prompt = "I have 500 coins, I just got 200 more. How many do I have?"

llm = OllamaChat(model="llama3:instruct")
fc = FunctionCaller(llm, [get_time, get_date, get_location, add, multiply])
result = fc.get_function(prompt)
```
No need for premade schemas, templates, prompts, or specialized functions.

[**See the examples folder for more usage examples**](https://github.com/emirsahin1/llm-axe/tree/main/examples)

## Features

- Local LLM internet access with Online Agent
- PDF Document Reader Agent
- Premade utility Agents for common tasks
- Compatible with any LLM, local or externally hosted
- Built-in support for Ollama



## Important Notes

The results you get from the agents are highly dependent on the capability of your LLM. An inadequate LLM will not be able to provide results that are usable with llm-axe

**Testing in development was done using llama3 8b:instruct 4 bit quant**
