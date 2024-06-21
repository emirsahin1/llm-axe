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

class AgentType(Enum):
    """
    Enum for agent types
    See documentation for explanation of Agents
    """
    PLANNER = "Planner"
    SUMMARIZER = "Summarizer"
    GENERIC_RESPONDER = "GenericResponder"
    VALIDATOR = "Validator"


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


# TODO:: Add custom html reader option