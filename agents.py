import warnings
import os
import json

from llm_axe.core import AgentType, safe_read_json, generate_schema, get_yaml_prompt, internet_search, read_website, read_pdf, make_prompt, llm_has_ask


class Agent:
    """
    Basic agent that can use premade or custom system prompts.
    """
    def __init__(self, llm:object, agent_type:AgentType=None, additional_system_instructions:str="", custom_system_prompt:str=None):
        """
        Args:
            llm (object): An LLM object with an ask function.
            agent_type (AgentType, optional): The type of agent to use. Choose from AgentType enum. Defaults to None.
            system_prompt (Agent): The name of the system prompt to use from the system_prompts.yaml file. See documentation for list.
            custom_system_prompt (any, optional): An optional string to override and use as the custom system prompt.
        """
        self.llm = llm
        self.chat_history = []
        if custom_system_prompt is None:
            if agent_type is None:
                raise ValueError("You must provide either a system_prompt or a custom_system_prompt.")
            else:
                self.system_prompt = get_yaml_prompt("system_prompts.yaml", agent_type.value)
        else:
            self.system_prompt = custom_system_prompt
        
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))

    def get_prompt(self, question):
        """
        Get the prompt for the given question.
        Args:
            question (str): The question to get the prompt for.
        """
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]
        return prompts

    def ask(self, prompt, history:list=None):
        """
        Ask a question based on the given prompt.
        Args:
            prompt (str): The prompt to use for the question. Will only use system_prompt if prompt is None.
            history (list, optional): The history of the conversation. Defaults to None.
        """
        if llm_has_ask(self.llm) is False:
            return None
        
        prompts = [self.system_prompt]
        if history is not None:
            prompts.extend(history)

        if prompt is not None:
            prompts.append(make_prompt("user", prompt))
            self.chat_history.extend(prompts[1:])
    
        response = self.llm.ask(prompts)
        self.chat_history.append(make_prompt("assistant", response))
        return response


class PythonAgent():
    """
    A PytonAgent agent is used to solve problems by writing Python code.
    It will provide code to execute and the imports used in the code.
    IMPORTANT!!: Code should ALWAYS be executed in a virtual or isolated environment.
    """
    def __init__(self, llm:object):
        self.llm = llm
        self.chat_history = []
        self.system_prompt = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "PythonAgent"))
        self.library_extractor = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "ImportExtractor"))

    def ask(self, prompt, history:list=None):
        """
        Ask a question based on the given prompt.
        Args:
            prompt (str): The prompt to use for the question. Will only use system_prompt if prompt is None.
            history (list, optional): The history of the conversation. Defaults to None.
        Returns:
            dict: A dictionary containing the "code" and "imports" keys.
                "code": The code to execute. In string format. WARNING!:CODE SHOULD NEVER BE EXECUTED OUTSIDE OF A VIRTUAL OR ISOLATED ENVIRONMENT.
                "libraries": The imports used in the code. In json list format.
        """
        if llm_has_ask(self.llm) is False:
            return None
        if history is not None:
            self.chat_history.extend(history)

        user_prompt = make_prompt("user", prompt)
        coder_prompts = [self.system_prompt, user_prompt]
        code_response = self.llm.ask(coder_prompts)
        self.chat_history.extend(coder_prompts[1:])
        self.chat_history.append(make_prompt("assistant", code_response))

        code = code_response.split("```")[1]

        # Clean up code
        if "Python" in code:
            code = code.replace("Python", "")
        
        # Extract imports
        imports = self.llm.ask([self.library_extractor, make_prompt("user", code_response)])
        self.chat_history.append(make_prompt("assistant", imports))
        imports = safe_read_json(imports)

        return {"code":code, "libraries":imports}


class DataExtractor():
    """
    A DataExtractor agent is used to extract information from given content.
    """
    def __init__(self, llm:object=None, reply_as_json:bool=False, additional_system_instructions:str=""):
        """
        Initializes a new DataExtractor.
        Args:
            llm (object): An object that implements the ask() method.
            reply_as_json (bool, optional): If True, the response will be in the format of a JSON object. Defaults to False.
            additional_system_instructions (str, optional): Additional instructions to include in the system prompt. Defaults to "".
        """
        yaml_prompt = "DataExtractorJson" if reply_as_json else "DataExtractor"
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", yaml_prompt)
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))
        self.llm = llm
        self.chat_history = []

    def get_prompt(self, info:str, data_points:list=[]):
        """
        Get the prompt for the given content.
        Args:
            info (str): The content to extract information from.
            data_points (list, optional): A list of data points to extract. Defaults to None.
        """
        prompt = '''
                Extract information from the following content:
                {content}

                Extract the following data:
                {data}
        '''
        prompt = prompt.format(content=info, data=", ".join(data_points))
        return [self.system_prompt, make_prompt("user", prompt)]
    
    def ask(self, info:str, data_points:list=[]):
        """
        Ask a question based on the given content.
        Args:
            info (str): The content to extract information from.
            data_points (list, optional): A list of data points to extract. Defaults to None.
        """
        prompts = self.get_prompt(info, data_points)
        resp = self.llm.ask(self.get_prompt(info, data_points))
        self.chat_history.extend(prompts[1:])
        self.chat_history.append(make_prompt("assistant", resp))
        return resp
    

class PdfReader():
    """
    A PdfReader agent is used to answer questions based on information from given PDF files.
    """

    def __init__(self, llm:object=None, pdf_files:list=None, additional_system_instructions:str="", custom_system_prompt:str=None):
        """
        Initializes a new PdfReader.
        Args:
            llm (object): An object that implements the ask() method.
            pdf_files (list): A list of PDF files to read. Each file should be a string representing the path to the PDF file.
            additional_system_instructions (str, optional): Additional instructions to include in the system prompt.
            custom_system_prompt (str, optional): Custom system prompt. Defaults to None.
        """
        self.chat_history = []
        self.llm = llm
        self.pdf_files = pdf_files
        self.additional_instructions = additional_system_instructions
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", "DocumentReader")
        self.custom_system_prompt = custom_system_prompt


    def ask(self, question:str, pdf_files:list=None, history:list=None):
        """
        Ask a question based on the given PDF files.
        Args:
            question (str): The question to ask.
            pdf_files (list): A list of PDF files to read. Each file should be a string representing the path to the PDF file.
            history (list): A list of previous chat messages in openai format.
        """
        if self.llm is None:
            raise ValueError("No LLM object provided.")
            
        prompts = []
        if history is not None:
            prompts.extend(history)

        question_prompts = self.get_prompt(question, pdf_files)
        prompts.extend(question_prompts)
        response = self.llm.ask(prompts)

        self.chat_history.append(question_prompts[1:]) # dont include the system prompt
        self.chat_history.append(make_prompt("assistant", response))
        return response
    
    def get_prompt(self, question, pdf_files:list=None):
        """
        Generates the prompt for the LLM.
        args:
            question (str): The question to ask.
            pdf_files (list): A list of PDF files to read. Each file should be a string representing the path to the PDF file. 
        """
        pdf_text = ""
        for pdf_file in pdf_files:
            pdf_text += f"Contents of document {os.path.basename(pdf_file)} :\n"
            pdf_text += read_pdf(pdf_file) + "\n\n"
        
        if self.custom_system_prompt is None:
            self.system_prompt = get_yaml_prompt("system_prompts.yaml", "DocumentReader")
        else:
            self.system_prompt = self.custom_system_prompt

        self.system_prompt = make_prompt("system", self.system_prompt.format(documents=pdf_text, additional_instructions=self.additional_instructions))
        
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]

        return prompts
    

class FunctionCaller():
    """
    A FunctionCaller agent is used to call functions using an LLM.
    By default, premade system prompts are used, however you can provide your own if you have issues with the default.
    For best results, your function name and parameters must be given meaningful names, 
    have type annotations doc string descriptions.
    """

    def __init__(self, llm:object=None, functions:list=None, additional_system_instructions:str="", custom_system_prompt:str=None):            
        """
        Initializes a new Function Caller.

        Args:
            llm (object, optional): An LLM object. Must have an ask method. Defaults to None.
            functions (list, optional): A list of functions. Defaults to None.
            additional_system_instructions (str, optional): Instructions in addition to the system prompt. Defaults to "".
            custom_system_prompt (str, optional): A custom system prompt. Will override the default. Defaults to None.
        """
        self.chat_history = []
        self.llm = llm
        self.functions_dict = {func.__name__: func for func in functions}
        self.additional_instructions = additional_system_instructions
        self.schema = generate_schema(functions)

        if custom_system_prompt is None:
            self.system_prompt = get_yaml_prompt("system_prompts.yaml", "FunctionCaller")
        else:
            self.system_prompt = custom_system_prompt

        self.system_prompt = make_prompt("system", self.system_prompt.format(schema=self.schema, additional_instructions=additional_system_instructions))
       
    def get_function(self, question, history:list=None):
        """
        Get the most appropriate function and its parameters based on the provided question.

        Parameters:
            question (str): The question to prompt the function caller with.
            history (list): A list of previous chat messages in openai format.

        Returns:
            tuple: A tuple containing the function, its parameters and the prompts used.
                - function (Callable): The function to be called.
                - parameters (dict): The parameters for the function.
                - prompts (list): The prompts that were used to speak to the llm.

        Raises:
            ValueError: If the llm object is not provided or does not have an ask function.
        """

        if self.llm is None:
            raise ValueError('''You must provide an llm to prompt the function caller!
                                If you wish to use external llms, use the get_prompt function to get the usable prompt.''')
        
        # check if llm has an ask function
        if llm_has_ask(self.llm) is False:
            return None
        
        prompts = []
        if history is not None:
            prompts.extend(history)

        question_prompts = self.get_prompt(question)
        prompts.extend(question_prompts)

        response = self.llm.ask(prompts)
        response_json = safe_read_json(response)

        self.chat_history.extend(question_prompts[1:])
        self.chat_history.append(make_prompt("assistant", response))

        if response_json is None:
            return None

        try:
            function_name = response_json["function"]
            parameters = response_json["parameters"]
        except KeyError:
            warnings.warn("llm did not respond with a function and parameters.")
            return None
        
        if function_name not in self.functions_dict:
            warnings.warn(f"{function_name} is not a valid function.")
            return None

        return {
        'function': self.functions_dict[function_name],
        'parameters': parameters,
        'prompts': prompts,
        'raw_response': response
    }
    

    def get_prompt(self, question):
        """
        Gets the prompt that the llm should use for the provided question to get the most appropriate function.

        Parameters:
            question (str): The question to generate the prompt for.

        Returns:
            list: A list containing the system prompt and the user prompt.
                - system_prompt (dict): The system prompt.
                    - role (str): The role of the prompt (system).
                    - content (str): The content of the system prompt.
                - user_prompt (dict): The user prompt.
                    - role (str): The role of the prompt (user).
                    - content (str): The content of the user prompt.
        """
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]
        return prompts


class OnlineAgent:
    """
    An agent that has internet access. 
    It will use the internet to try and best answer the user prompt
    """

    def __init__(self, llm:object, additional_system_instructions:str="", custom_searcher:callable=None, custom_site_reader:callable=None):
        """
        Args:
            llm (object): An LLM object. Must have an ask method.
            additional_system_instructions (str, optional): Instructions in addition to the system prompt. Defaults to "".
            custom_searcher (function, optional): A custom online searcher function. The searcher function must take a query and return a list of string URLS
            custom_site_reader (function, optional): A custom online site reader function. The site reader function must take a URL and return a string representation of the site.
        """
        self.llm = llm
        self.chat_history = []
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", "OnlineSearcher")
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))
        self.search_function = custom_searcher if custom_searcher else internet_search
        self.site_reader_function = custom_site_reader if custom_site_reader else read_website

    def search(self, prompt, history:list=None):
        """
        Searches the internet and answers the prompt based on the search results.

        Parameters:
            prompt (str): The prompt or question to answer.

        Returns:
            str: The response that answers the prompt.
        """

        # Will first get a good search query
        # The query will be used to find relevant URLS
        # The llm will pick the best URL and read it to answer the prompt
    
        query = self.get_search_query(prompt)
        if query is None:
            return None

        search_results = self.search_function(query)
        search_results =  json.dumps(search_results)

        url_picker_prompts = []
        if history is not None:
            url_picker_prompts.extend(history)

        url_picker_prompt = get_yaml_prompt("system_prompts.yaml", "UrlPicker")
        url_picker_prompt = make_prompt("system", url_picker_prompt.format(question=prompt, urls=search_results))
        url_picker_prompts.append(url_picker_prompt)

        resp = self.llm.ask(url_picker_prompts)
        resp_json = safe_read_json(resp)
        self.chat_history.extend(url_picker_prompts[1:])
        self.chat_history.append(make_prompt("assistant", resp))

        # Check if the response is a valid url
        url = None
        if resp_json is not None and "url" in resp_json:
            url = resp_json["url"]
        else:
            warnings.warn("LLM did not respond with valid url or json response.")
            return None
            
        website_text = self.site_reader_function(url)
        user_prompt = f'''
                    Please read the following information:
                    
                    Information about Website {url}: 
                    {website_text}

                    Answer the following question based on the above information: 
                    {prompt}

                    Start your answer with "Based on information from the internet, "
                    '''
        
        final_responder = Agent(llm=self.llm, agent_type=AgentType.GENERIC_RESPONDER)
        response = final_responder.ask(user_prompt, history)

        self.chat_history.append(make_prompt("user", user_prompt))
        self.chat_history.append(make_prompt("assistant", response))
        
        return response

    def get_search_query(self, question):
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]
        response = self.llm.ask(prompts)
        response_json = safe_read_json(response)
        self.chat_history.extend(prompts[1:])
        self.chat_history.append(make_prompt("assistant", response))
        if response_json is not None and "search_query" in response_json:
            return response_json["search_query"]
        else:
            return None
            