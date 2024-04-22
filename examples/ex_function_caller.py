from llm_axe.agents import FunctionCaller
from llm_axe.models import OllamaChat
import time

# We define the functions that we want the llm to be able to call. 
# Note that documentation is not required, but should be used 
#   to help the llm for understanding what each function does.
# Specifying parameter types is optional but highly recommended
def get_time():
    return time.strftime("%I:%M %p")

def get_date():
    return time.strftime("%Y-%m-%d")

def get_location():
    return "USA"

def add(num1:int, num2:int):
    return num1 + num2

def multiply(num1:int, num2:int):
    return num1 * num2

def get_distance(lat1:int, lon1:int, lat2:int, lon2:int):
    """
    Calculates the distance between two points on the Earth's surface using the Haversine formula.
    :param lat1: latitude of point 1
    :param lon1: longitude of point 1
    :param lat2: latitude of point 2
    :param lon2: longitude of point 2
    """
    return(lat1, lon1, lat2, lon2)
   

llm = OllamaChat(model="llama3:instruct")
prompt = "I have 500 coins, I just got 200 more. How many do I have?"

# Here we setup and prompt the function caller in just two lines of code.
fc = FunctionCaller(llm, [get_time, get_date, get_location, add, multiply, get_distance])
result = fc.get_function(prompt)

# If no function was found, exit
if(result is None):
    print("No function found")
    exit()

func = result['function']
params = result['parameters']

print(func(**params))
print(result['parameters'])
print(result['prompts'])
print(result['raw_response'])

