#[How to Create your own LLM Agent from Scratch: A Step-by-Step Guide](https://gathnex.medium.com/how-to-create-your-own-llm-agent-from-scratch-a-step-by-step-guide-14b763e5b3b8)
import dotenv
dotenv.load_dotenv()
import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
        SystemMessage,
        AIMessage,
        HumanMessage)
from googleapiclient.discovery import build
from py_expression_eval import Parser
import re, time, os

from agent_prompt_templates import (
    system_prompt_1,
    system_prompt_2,
)
agent_system_prompt = system_prompt_2

def query_model(model, messages):
    if isinstance(model, OpenAI):
        response = model.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            top_p=1,)
        delay = 5 #To prevent the Rate Limit error for free-tier users
        print(f'Waiting for {delay} seconds...')
        time.sleep(delay)
    else:
        response = model.invoke(messages)

    return response

def search(search_term):
    search_result = ""
    service = build("customsearch", "v1", developerKey=os.environ.get("GOOGLE_API_KEY"))
    res = service.cse().list(q=search_term, cx=os.environ.get("GOOGLE_CSE_ID"), num = 10).execute()
    for result in res['items']:
        search_result = search_result + result['snippet']
    return search_result

parser = Parser()
def calculator(str):
    return parser.parse(str).evaluate({})

def pass_through(input):
    return input

def stream_agent(model, prompt):
    messages = [
        SystemMessage(content=agent_system_prompt),
        HumanMessage(content=prompt),
    ]
    if isinstance(model, OpenAI) or isinstance(model, ChatOpenAI):
        messages = [
            { "role": "system", "content": agent_system_prompt },
            { "role": "user", "content": prompt },
        ]

    def extract_action_and_input(text):
          action_pattern = r"Action: (.+?)\n"
          #input_pattern = r"Action Input: \"(.+?)\""
          input_pattern = r"Action Input: (.+)"
          action = re.findall(action_pattern, text)
          action_input = re.findall(input_pattern, text)
          print(f'\n{"="*50}\nText:{text}\n{"-"*30}\nAction:{action}\nAction Input:{action_input}')
          return action, action_input

    while True:
        response = query_model(model, messages)
        if isinstance(model, OpenAI): # or isinstance(model, ChatOpenAI):
            response_text = response.choices[0].message.content
        else:
            response_text = response.content
        print(f"\n{'#'*50}\nresponse_text:{response_text}\n{'#'*50}\n")

        action, action_input = extract_action_and_input(response_text)
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            tool = pass_through
            print(f"# Response: {action_input[-1]}")
            break

        observation = tool(action_input[-1])
        print("Observation: ", observation)
        
        if isinstance(model, OpenAI) or isinstance(model, ChatOpenAI):
            amsg = [{ "role": "system", "content": response_text },
                    { "role": "user", "content": f"Observation: {observation}" },]
        else:
            amsg = [AIMessage(content=response_text),
                    HumanMessage(content=f"Observation: {observation}"),]
        messages.extend(amsg)
        
def get_openai_model():
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def get_model(base_url="http://192.168.0.24:8080"):
    return ChatOpenAI(model="gpt-3.5-turbo", base_url=base_url)

if __name__ == "__main__":
    #model = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    #model = ChatOpenAI(model="gpt-3.5-turbo")#, base_url="http://192.168.0.24:8080")
    model = ChatOpenAI(model="gpt-3.5-turbo", base_url="http://192.168.0.24:8080")
    stream_agent(model, "What is 1+1?")
    #stream_agent(model, "Who is new opanai board member?")    
