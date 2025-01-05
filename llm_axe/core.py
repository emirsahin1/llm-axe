import inspect
import json
import os
import warnings
import yaml
import docstring_parser
from enum import Enum
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader as pypdfReader
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


class AgentType(Enum):
    """
    Enum for agent types
    See documentation for explanation of Agents
    """
    PLANNER = "Planner"
    SUMMARIZER = "Summarizer"
    GENERIC_RESPONDER = "GenericResponder"
    VALIDATOR = "Validator"

def stream_and_record(stream, chat_history):
    """Streams the response from the LLM and records the chat history."""
    chunks = []
    for chunk in stream:
        if type(chunk) is dict and "message" in chunk:
            content = chunk["message"]["content"]
        else:
            content = chunk
        chunks.append(content)
        yield content
    if chat_history is not None:
        complete_resp = "".join(chunks)
        chat_history.append(make_prompt("assistant", complete_resp))


def llm_has_ask(llm):
    """
    Checks if the llm object has an ask function
    Args:
        llm (object): The object to check
    Returns:
        bool: True if the llm object has an ask function, False otherwise
    """
    if not hasattr(llm, "ask"):
            warnings.warn("llm object must have an ask function! See OllamaChat class in models.py for an example.")
            return False
    return True


def make_prompt(role:str, content:str, images:list=None):
    """
    Creates a prompt in OpenAI format
    Args:
        role (str): The role of the prompt
        content (str): The content of the prompt
        images (list, optional): A list of images to include in the prompt. Defaults to None.
    Returns:
        dict: The prompt in OpenAI format
    """
    args={
        "role": role,
        "content": content
    }
    if images is not None:
        args["images"] = images

    return {**args}


def read_pdf(file):
    """
    Reads a pdf file and returns the complete text content.
    Args:
        file (str): The path to the pdf file.
    """
    reader = pypdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def get_yaml_prompt(yaml_file_name:str, prompt_name:str):
    """
    Reads a prompt from a yaml file.
    See system_prompts.yaml for a list of prompts.
    Args:
        yaml_file_name (str): The name of the yaml file to load.
        prompt_name (str): The name of the prompt to load.
    """
    dir_of_file = os.path.dirname(os.path.realpath(__file__))
    yaml_path = os.path.join(dir_of_file, yaml_file_name)
    file = open(yaml_path)
    loaded = yaml.safe_load(file)[prompt_name]["prompt"]
    file.close()
    return loaded


def generate_schema(functions):
    """
    Generates a schema for a list of functions.
    Doc string information is used as aid to generate the schema.
    Args:
        functions (list): A list of functions to generate the schema for.
    """
    schema = {}
    for func in functions:
        func_name = func.__name__
        signature = inspect.signature(func)
        params = {}
        default = None
        doc = docstring_parser.parse(func.__doc__)
        i = 0
        for name, param in signature.parameters.items():
            param_desc = doc.params[i].description if i < len(doc.params) else "None"

            if param.default is inspect.Parameter.empty:
                default = "None"
            else:
                default = param.default

            params[name] = {'type': str(param.annotation), 'default value': default, 'description': param_desc}
            i+=1
        schema[func_name] = {'description': doc.short_description, 'parameters': params}

    return json.dumps(schema)


def safe_read_json(response):
    """
    Reads the string as json and checks if it is a valid json.
    Args:
        response (str): The string response from the LLM.
    """
    response_json = None
    try:
            response_json = json.loads(response)
    except json.decoder.JSONDecodeError:
            try:
                response_json = json.loads(clean_json_response(response))
            except json.decoder.JSONDecodeError:
                warnings.warn("llm did not respond with proper json.")
                response_json = None
    return response_json


def clean_json_response(response):
    """
    Removes unnecessary characters from the json response.
    Args:
        response (str): The string response from the LLM.
    """
    brace_count = 0
    start = None

    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start = i  # Mark the start of a JSON object
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start is not None:
                # We've found a complete JSON object
                return response[start:i+1]
    return ""


def internet_search(query):
    """
    Searches the internet for a query.
    Returns top 5 results.
    Args:
        query (str): The query to search for.
    """
    urls = list(search(query, tld="co.in", num=5, stop=10, pause=2))
    urls_detailed = []

    for url in urls:
        urls_detailed.append(fetch_url_info(url))

    return urls_detailed

def read_website(url):
    """
    Reads and returns the body of the website at the given url.
    Args:
        url (str): The url of the website to read.
    Returns:
        str: The body of the website.
        None: If the request fails.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.body
        body_text = body.get_text(strip=True)
        return body_text        
    else:
        warnings.warn("Failed to retrieve the website")
        return None
    

def selenium_reader(url):
    """
    Reads and returns the body of the website at the given url using selenium.
    Args:
        url (str): The url of the website to read.
    Returns:
        str: The body of the website.
        None: If the request fails.
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--incognito')
    options.add_argument('--headless')
    options.add_argument("--enable-javascript")
    options.add_argument('--remote-debugging-pipe')
    
    
    prefs = {
        "download.open_pdf_in_system_reader": False,
        "download.prompt_for_download": True,
        "plugins.always_open_pdf_externally": False,
        "download_restrictions": 3,
        "download.default_directory": 'NUL' if os.name == "nt" else '/dev/null',
    }
    options.add_experimental_option(
        "prefs", prefs
    )

    selenium_executable_path = os.path.join(os.getcwd(), "chromedriver" + (".exe" if os.name == "nt" else ""))
    cService = webdriver.ChromeService(executable_path=selenium_executable_path)
    driver = webdriver.Chrome(options=options, service=cService)
    driver.get(url)
    time.sleep(3)
    body = driver.find_element(By.TAG_NAME, "body")
    body_text = body.get_attribute('innerText')
    driver.quit()
    return body_text

def selenium_hybrid_reader(url):
    """
    Reads and returns the body of the website at the given url using selenium or bs4(mainly).
    Args:
        url (str): The url of the website to read.
    Returns:
        str: The body of the website.
        None: If the request fails.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.body
        body_text = body.get_text(strip=True)
        noscript = soup.find("noscript")

        if noscript and 'enable' in str(noscript.get_text):
            body_text = selenium_reader(url)

        if body_text == "":
            body_text = selenium_reader(url)
        
        return body_text

def fetch_url_info(url):
    """
    Fetches the description and title of the website at the given url.
    Args:
        url (str): The url of the website to read.
    
    Returns:
        dict: A dictionary containing the url, title and description of the website.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers,timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.find('title').text if soup.find('title') else 'No title found'
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc['content'] if meta_desc else 'No description found'
            return {"url": url, 'title': title, 'description': description}
        else:
            return None
    except Exception as e:
        return None

def find_most_relevant(text_embedding_pairs: list, prompt_embedding: list, top_k: int = 5):
    """
    Finds the most relevant embeddings in the embeddings list in relation to the prompt_embedding.
    Args:
        text_embedding_pairs (list): A list of embeddings to search through.
        prompt_embedding (list): The embedding of the prompt.
        top_k (int, optional): The number of texts to return, in descending order. Defaults to 5.
    Returns:
        list: A list of the most relevant text.
    """
    texts, embeddings = zip(*text_embedding_pairs)
    distances = cosine_similarity([prompt_embedding], embeddings)[0]
    top_similar = np.argsort(distances)[-top_k:][::-1]
    return [texts[i] for i in top_similar]

def split_into_sentences(text: str):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    Args:
        text (str): The text to split into sentences.
   
    Returns: 
        list: A list of sentences
    """
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences


def split_into_chunks(text: str, sentences_per_chunk: int):
    """
    Split the text into chunks, limited by the number of sentences per chunk.

    Args:
        text (str): The text to split into chunks.
        sentences_per_chunk (int): The maximum number of sentences per chunk.

    Returns:
        list: A list of chunks.
    """
    sentences = split_into_sentences(text)
    chunks = [" ".join(sentences[i:i+sentences_per_chunk]) for i in range(0, len(sentences), sentences_per_chunk)]
    return chunks
