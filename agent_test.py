import dotenv
dotenv.load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field

from conversational_1st import *

def bind_tools_test():
    llm = ChatOpenAI( api_key="ollama", model="llama3", base_url="http://localhost:11434/v1",)
    llm = llm.bind_tools(tools=[])

def test_agent():
    m = get_model()#base_url="http://192.168.0.24:8080")

    r,_ = init_retriever()
    tool = create_retriever_tool(r, 'blog','retrieve from blog')
    m.bind_tools([tool])
    agent_executor = create_react_agent(m, [tool])

def test_agent2():
    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [create_retriever_tool(llm, 'blog','retrieve from blog')]
    agent_executor = create_react_agent(llm, tools)
    llm.bind_tools(tools)

def test_agent3():
    #llm = ChatOpenAI(model="gpt-4o", max_retries=2,# base_url="...",)
    llm = get_model(base_url="http://192.168.0.24:8080")
    messages = [( "system", "You are a helpful assistant that translates English to French. Translate the user sentence.",),
                ("human", "I love programming."),]

    ai_msg = llm.invoke(messages)
    print(ai_msg.content)

    class GetWeather(BaseModel):
        """Get the current weather in a given location"""
        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")

    llm_with_tools = llm.bind_tools([GetWeather])
    ai_msg = llm_with_tools.invoke("what is the weather like in San Francisco",)
    print(ai_msg)

if __name__ == "__main__":
    test_agent3()
