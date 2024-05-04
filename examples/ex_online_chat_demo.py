# Example Conversation Output:

# User: Who are you?
# AI: I am a helpful chat assistant.

# User: Great. Can you find me a website that I can use to practice web scraping?

# AI: Based on information from the internet, one such website is https://www.scrapethissite.com/pages/. 
# This website provides various examples of pages for practicing web scraping, including country lists, hockey team stats, Oscar-winning films, and more. These examples cover different types of interfaces, such as forms, searching, pagination, AJAX, JavaScript, frames, and iFrames, making it a great resource to practice your web scraping skills.    

# User: Thanks!
# AI: You're welcome!


# This is an example chat session through the terminal
# The AI will use the internet if it is required, otherwise, it will use its own knowledge base

from llm_axe import make_prompt, AgentType, OllamaChat, OnlineAgent, FunctionCaller, Agent

def internet():
    """Choose if the internet is required or if we don't have information about the topic"""
    return ""

def no_internet():
    """Choose if the internet is not required"""
    return ""

prompt = '''
        You are an AI assistant.
        Determine whether or not the internet is required to answer the user's prompt.
        If the internet is not required, answer "no internet required".
        If the internet is required, answer "internet required".

        Internet is required if you don't have information about the topic and the user's prompt is talking about something SPECIFIC that we need solid FACTS for.
        Internet is not required if you believe that the internet is not required to correctly answer the user's question.

        Do not respond with anything else.
'''

def main():
    print('''
        ******************************************************************
        Welcome to the online chat demo. 
        If the AI is unable to answer your question, it will use the internet to answer.
        Otherwise, it will answer using its own knowledge base.
        Type 'exit' to exit.
        ******************************************************************''')

    llm = OllamaChat(model="llama3:instruct")
    online_agent = OnlineAgent(llm)
    plan_agent = Agent(llm, custom_system_prompt=prompt)
    normal_agent = Agent(llm, agent_type=AgentType.GENERIC_RESPONDER)
    function_caller = FunctionCaller(llm, functions=[internet, no_internet])

    chat_history = []
    user_input = ""

    while True:
        print("User: ", end="")
        user_input = input()
        if user_input == "exit":
            exit()

        chat_history.append(make_prompt("user", user_input))

        # First determine if the internet is required or not
        plan = plan_agent.ask("USERS INPUT: " + user_input)
        
        function_name = "no_internet"
        func_resp = function_caller.get_function(plan) 
        if func_resp is not None:
            function_name = func_resp["function"].__name__

        resp = ""
        if function_name == "internet":
            resp = online_agent.search(user_input)
        else:
            resp = normal_agent.ask(user_input, history=chat_history)

        print("\nAI: " + resp + "\n")
        chat_history.append(make_prompt("assistant", resp))
    

if __name__ == '__main__':
    main()
