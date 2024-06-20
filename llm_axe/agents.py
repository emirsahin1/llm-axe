import warnings
import os
import json

from llm_axe.core import AgentType, safe_read_json, generate_schema, get_yaml_prompt, internet_search, read_website, read_pdf, make_prompt, llm_has_ask


class Agent:
    """
    Basic agent that can use premade or custom system prompts.
    Custom system prompt will override any premade prompts.
    """
    def __init__(self, llm:object, agent_type:AgentType=None, additional_system_instructions:str="", custom_system_prompt:str=None, format:str="", temperature:float=0.8):
        """
        Args:
            llm (object): An LLM object with an ask function.
            agent_type (AgentType, optional): The type of agent to use. Choose from AgentType enum. Defaults to None.
            additional_system_instructions (str, optional): Additional system instructions to include in the premade system prompt. Defaults to "".
            custom_system_prompt (any, optional): An optional string to totally override and use as the custom system prompt.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.llm = llm
        self.chat_history = []
        if custom_system_prompt is None:
            if agent_type is None:
                raise ValueError("You must provide either a AgentType or a custom_system_prompt.")
            else:
                self.system_prompt = get_yaml_prompt("system_prompts.yaml", agent_type.value)
        else:
            self.system_prompt = custom_system_prompt
        
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))
        self.temperature = temperature
        self.format = format

    def get_prompt(self, question):
        """
        Get the prompt for the given question.
        Args:
            question (str): The question to get the prompt for.
        """
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]
        return prompts

    def ask(self, prompt, images:list=None, history:list=None):
        """
        Ask a question based on the given prompt or images.
        Images require a multimodal LLM.
        Args:
            prompt (str): The prompt to use for the question.
            images (list, optional): The images to include in the prompt. Each image must be a path to an image or base64 data. Defaults to None.
            history (list, optional): The history of the conversation. Defaults to None.
        """
        if llm_has_ask(self.llm) is False:
            return None
        
        prompts = [self.system_prompt]
        if history is not None:
            prompts.extend(history)

        prompts.append(make_prompt("user", prompt, images))
        response = self.llm.ask(prompts, temperature=self.temperature, format=self.format)
    
        self.chat_history.append(prompts[-1])
        self.chat_history.append(make_prompt("assistant", response))
        return response


class ObjectDetectorAgent():
    """
    An ObjectDetectorAgent agent is used to detect objects in an image.
    Requires a multimodal LLM.
    """
    def __init__(self, vision_llm:object, text_llm:object, vision_temperature:float=0.3, text_temperature:float=0.3):
        """
        Initializes a new ObjectDetectorAgent object.

        Args:
            vision_llm (object): A multimodal LLM object that implements the ask() method.
            text_llm (object): An llm that is responsible for the final output
            vision_temperature (float, optional): The temperature of the vision LLM. Defaults to 0.3.
            text_temperature (float, optional): The temperature of the text LLM. Defaults to 0.3.
        """
        self.vision_llm = vision_llm
        self.text_llm = text_llm
        self.__system_prompt = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "ObjectDetector"))
        self.vision_temperature = vision_temperature
        self.text_temperature = text_temperature

    def detect(self, images:list, objects:list=None, detection_criteria:str=None):
        """
        Detects objects in an image.
        If given a list of objects, only the objects in the list will be detected.
        If a detection_criteria is given instead, it will detect according to the criteria's instructions. 
        Args:
            images (list): The images to detect objects in. List of string paths or byte data.
            objects (list, optional): An optional list of objects to detect. Defaults to None.
            detection_criteria (str, optional): An optional detection criteria to give.
        """
        if llm_has_ask(self.vision_llm) is False or llm_has_ask(self.text_llm) is False:
            return None
        
        prompts = make_prompt("user", "Detect all objects in this image", images=images)
        detected_objects = self.vision_llm.ask([self.__system_prompt, prompts], temperature=self.vision_temperature)
        prompts = self.__get_prompt(images, detected_objects, objects, detection_criteria)
        response = self.text_llm.ask(prompts, format="json", temperature=self.text_temperature)

        return response
        
    def __get_prompt(self, detected_objects:list, objects:list=None, detection_criteria:str=None):
        """
        Get the prompt for the given objects and detection_prompt.
        Args:
            detected_objects (list): The detected objects in the images.
            objects (list, optional): An optional list of objects to detect. Defaults to None.
            detection_criteria (str, optional): An optional detection criteria to give.
        """
        prompt = ""
        sys_prompt = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "ObjectFilterer"))

        if objects is not None and detection_criteria is not None:
            raise ValueError("You cannot provide both an object list and a detection_prompt.")
        else:
            if detection_criteria is None:
                if objects is None:
                    raise ValueError("You must provide either an object list or a detection_prompt.")
                else:
                    prompt = "Image Description: " + detected_objects + "\n\nI'm INTERESTED in the following OBJECTS: " + ", ".join(objects)
            else:
                prompt = "Image Description: " + detected_objects + "\n\n I'm INTERESTED in the following: " + detection_criteria
            prompt = prompt + "\n Only return the objects that fit my interests"
            prompt = make_prompt("user", prompt)
            return [sys_prompt, prompt]


class PythonAgent():
    """
    A PythonAgent agent is used to solve problems by writing Python code.
    It will provide code to execute and the imports used in the code.
    IMPORTANT!!: Code should ALWAYS be executed in a virtual or isolated environment.
    """
    def __init__(self, llm:object, temperature:float=0.8):
        """
        Initializes a new PythonAgent object.

        Args:
            llm (object): An object that implements the ask() method.
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.llm = llm
        self.chat_history = []
        self.system_prompt = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "PythonAgent"))
        self.library_extractor_prompt = make_prompt("system", get_yaml_prompt("system_prompts.yaml", "ImportExtractor"))
        self.temperature = temperature

    def ask(self, prompt, history:list=None):
        """
        Ask a question based on the given prompt.
        Args:
            prompt (str): The prompt to use for the question.
            history (list, optional): A list of previous chat messages in openai format. Defaults to None.
        Returns:
            dict: A dictionary containing the "code" and "imports" keys.
                "code": The code to execute. In string format. WARNING!:CODE SHOULD NEVER BE EXECUTED OUTSIDE OF A VIRTUAL OR ISOLATED ENVIRONMENT.
                "libraries": The imports used in the code. In json list format.
        """
        if llm_has_ask(self.llm) is False:
            return None
        
        coder_prompts = []
        coder_prompts.append(self.system_prompt)
        if history is not None:
            coder_prompts.extend(history)

        user_prompt = make_prompt("user", prompt)
        coder_prompts.append(user_prompt)
            
        code_response = self.llm.ask(coder_prompts, temperature=self.temperature)
        self.chat_history.append(user_prompt)
        self.chat_history.append(make_prompt("assistant", code_response))

        code = code_response.split("```")[1]

        # Clean up code
        if "Python" in code:
            code = code.replace("Python", "")
        
        # Extract imports
        imports = self.llm.ask([self.library_extractor_prompt, make_prompt("user", code_response)], format="json", temperature=self.temperature)
        self.chat_history.append(make_prompt("assistant", imports))
        imports = safe_read_json(imports)

        return {"code":code, "libraries":imports}


class DataExtractor():
    """
    A DataExtractor agent is used to extract information from given content.
    """
    def __init__(self, llm:object, reply_as_json:bool=False, additional_system_instructions:str="", temperature:float=0.8):
        """
        Initializes a new DataExtractor.
        Args:
            llm (object): An object that implements the ask() method.
            reply_as_json (bool, optional): If True, the response will be in the format of a JSON object. Defaults to False.
            additional_system_instructions (str, optional): Additional instructions to include in the system prompt. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        yaml_prompt = "DataExtractorJson" if reply_as_json else "DataExtractor"
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", yaml_prompt)
        if reply_as_json:
            self.system_prompt = self.system_prompt + "\n Respond in JSON format"
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))
        self.llm = llm
        self.chat_history = []
        self.temperature = temperature

    def get_prompt(self, info:str, data_points:list=[]):
        """
        Get the prompt for the given content.
        Args:
            info (str): The content to extract information from.
            data_points (list, optional): A string list of data points to extract. Defaults to None.
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
            data_points (list, optional): A string list of data points to extract. 
            Example: ["name", "age", "city"] Defaults to None.
        """
        prompts = self.get_prompt(info, data_points)
        resp = self.llm.ask(self.get_prompt(info, data_points), temperature=self.temperature)
        self.chat_history.append(prompts[1])
        self.chat_history.append(make_prompt("assistant", resp))
        return resp
    

class PdfReader():
    """
    An Agent used to answer questions based on information from given PDF files.
    """

    def __init__(self, llm:object, additional_system_instructions:str="", custom_system_prompt:str=None, temperature:float=0.8):
        """
        Initializes a new PdfReader.
        Args:
            llm (object): An object that implements the ask() method.
            additional_system_instructions (str, optional): Additional instructions to include in the system prompt.
            custom_system_prompt (str, optional): Custom system prompt. Defaults to None.
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.chat_history = []
        self.llm = llm
        self.additional_instructions = additional_system_instructions
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", "DocumentReader")
        self.custom_system_prompt = custom_system_prompt
        self.temperature = temperature


    def ask(self, question:str, pdf_files:list, history:list=None):
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
        question_prompts = self.get_prompt(question, pdf_files)
        prompts.append(question_prompts[0])

        if history is not None:
            prompts.extend(history)

        prompts.append(question_prompts[1])
        response = self.llm.ask(prompts, temperature=self.temperature)

        self.chat_history.append(question_prompts[1]) # dont include the system prompt
        self.chat_history.append(make_prompt("assistant", response))
        return response
    
    def get_prompt(self, question, pdf_files:list=None):
        """
        Generates the prompt for the LLM.
        args:
            question (str): The question to ask.
            pdf_files (list): A list of PDF files to read. Each file should be a string path to the PDF file. 
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

    def __init__(self, llm:object, functions:list, additional_system_instructions:str="", custom_system_prompt:str=None, temperature:float=0.8):            
        """
        Initializes a new Function Caller.

        Args:
            llm (object, optional): An LLM object. Must have an ask method.
            functions (list, optional): A list of functions.
            additional_system_instructions (str, optional): Instructions in addition to the default system prompt. Defaults to "".
            custom_system_prompt (str, optional): A custom system prompt to override the default function picking prompt. Defaults to None.
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.chat_history = []
        self.llm = llm
        self.functions_dict = {func.__name__: func for func in functions}
        self.additional_instructions = additional_system_instructions
        self.schema = generate_schema(functions)
        self.temperature = temperature

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
        question_prompts = self.get_prompt(question)
        prompts.append(question_prompts[0])

        if history is not None:
            prompts.extend(history)
        prompts.append(question_prompts[1])            

        response = self.llm.ask(prompts, format="json", temperature=self.temperature)
        response_json = safe_read_json(response)

        self.chat_history.append(question_prompts[1])
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


class WebsiteReaderAgent:
    """
    An agent that will read a specificwebsite and answer questions based on it.
    """

    def __init__(self, llm:object, additional_system_instructions:str="", custom_site_reader:callable=None, temperature:float=0.8):
        """
        Args:
            llm (object): An LLM object. Must have an ask method.
            additional_system_instructions (str, optional): Instructions in addition to the system prompt. Defaults to "".
            custom_site_reader (function, optional): A custom online site reader function. The site reader function must take a URL and return a string representation of the site.
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.llm = llm
        self.chat_history = []
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", "WebsiteReader")
        self.additional_system_instructions = additional_system_instructions
        self.read_function = custom_site_reader if custom_site_reader else read_website
        self.temperature = temperature

    def ask(self, question:str, url:str, history:list=None):
        """
        Answers the question based on the provided website.
        
        Args:
            question (str): The question to answer.
            url (str): The url of the website to read.
            history (list, optional): A list of previous chat messages in openai format. Defaults to None.
        """
        if llm_has_ask(self.llm) is False:
            return None

        website_content = read_website(url)
        
        if website_content is None:
            website_content = "Website could not be read"

        syst_prompt = get_yaml_prompt("system_prompts.yaml", "WebsiteReader")
        syst_prompt = make_prompt("system", self.system_prompt.format(url=url, additional_instructions=self.additional_system_instructions, content=website_content))

        prompts = []
        prompts.append(syst_prompt)
        if history is not None:
            prompts.extend(history)

        user_prompt = make_prompt("user", question)
        prompts.append(user_prompt)
        
        response = self.llm.ask(prompts, temperature=self.temperature)
        self.chat_history.append(user_prompt)
        self.chat_history.append(make_prompt("assistant", response))
        return response
    

class OnlineAgent:
    """
    An agent that has internet access. 
    It will use the internet to try and best answer the user prompt.
    """

    def __init__(self, llm:object, additional_system_instructions:str="", custom_searcher:callable=None, custom_site_reader:callable=None, temperature:float=0.8):
        """
        Args:
            llm (object): An LLM object. Must have an ask method.
            additional_system_instructions (str, optional): Instructions in addition to the system prompt. Defaults to "".
            custom_searcher (function, optional): A custom online searcher function. The searcher function must take a query and return a list of string URLS
            custom_site_reader (function, optional): A custom online site reader function. The site reader function must take a URL and return a string representation of the site.
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        self.llm = llm
        self.chat_history = []
        self.system_prompt = get_yaml_prompt("system_prompts.yaml", "OnlineSearcher")
        self.system_prompt = make_prompt("system", self.system_prompt.format(additional_instructions=additional_system_instructions))
        self.search_function = custom_searcher if custom_searcher else internet_search
        self.site_reader_function = custom_site_reader if custom_site_reader else read_website
        self.temperature = temperature

    def search(self, prompt, history:list=None):
        """
        Searches the internet and answers the prompt based on the search results.

        Parameters:
            prompt (str): The prompt or question to answer.
            history (list, optional): A list of previous chat messages in openai format. Defaults to None.

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

        # Use system prompt as user, since we have chat history
        url_picker_prompt = get_yaml_prompt("system_prompts.yaml", "UrlPicker")
        url_picker_prompt = make_prompt("user", url_picker_prompt.format(question=prompt, urls=search_results))
        url_picker_prompts.append(url_picker_prompt)

        resp = self.llm.ask(url_picker_prompts, format="json", temperature=self.temperature)
        resp_json = safe_read_json(resp)

        self.chat_history.append(url_picker_prompt)
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
        response = final_responder.ask(user_prompt, history=history)

        self.chat_history.append(make_prompt("user", user_prompt))
        self.chat_history.append(make_prompt("assistant", response))
        
        return response

    def get_search_query(self, question):
        user_prompt = make_prompt("user", question)
        prompts = [self.system_prompt, user_prompt]
        response = self.llm.ask(prompts, format="json", temperature=self.temperature)
        response_json = safe_read_json(response)
        self.chat_history.append(prompts[1])
        self.chat_history.append(make_prompt("assistant", response))
        if response_json is not None and "search_query" in response_json:
            return response_json["search_query"]
        else:
            return None
            
