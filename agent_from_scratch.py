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

system_prompt = """
Answer the following questions and obey the following commands as best you can.

You have access to the following tools:

Search: Search: useful for when you need to answer questions about current events. You should ask targeted questions.
Calculator: Useful for when you need to answer questions about math. Use python code, eg: 2 + 2
Response To Human: When you need to respond to the human you are talking to.

You will receive a message from the human, then you should start a loop and do one of two things

Option 1: You use a tool to answer the question.
For this, you should use the following format:
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: "the input to the action, to be sent to the tool"

After this, the human will respond with an observation, and you will continue.

Option 2: You respond to the human.
For this, you should use the following format:
Action: Response To Human
Action Input: "your response to the human, summarizing what you did and what you learned"

Begin!
"""

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
    oms = [
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": prompt },
    ]
    lms = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    if isinstance(model, OpenAI):
        messages = oms
    else:
        messages = lms

    def extract_action_and_input(text):
          action_pattern = r"Action: (.+?)\n"
          #input_pattern = r"Action Input: \"(.+?)\""
          input_pattern = r"Action Input: (.+)"
          action = re.findall(action_pattern, text)
          action_input = re.findall(input_pattern, text)
          print(f'\ntext:{text}\nAction: {action}, Action Input: {action_input}')
          return action, action_input

    while True:
        response = query_model(model, messages)
        if isinstance(model, OpenAI):
            response_text = response.choices[0].message.content
        else:
            response_text = response.content
        print(response_text)

        action, action_input = extract_action_and_input(response_text)
        if action[-1] == "Search":
            tool = search
        elif action[-1] == "Calculator":
            tool = calculator
        elif action[-1] == "Response To Human":
            tool = pass_through
            print(f"Response: {action_input[-1]}")
            break

        observation = tool(action_input[-1])
        print("Observation: ", observation)
        
        if isinstance(model, OpenAI):
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
    model = ChatOpenAI(model="gpt-3.5-turbo", base_url="http://192.168.0.24:8080")
    stream_agent(model, "What is 1+1?")
    #stream_agent(model, "Who is new opanai board member?")    
