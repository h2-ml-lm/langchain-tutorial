'''
[Langchain Tutorial: Build a Chatbot](https://python.langchain.com/v0.2/docs/tutorials/chatbot/)
'''
import dotenv
dotenv.load_dotenv()
from pprint import pp
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, trim_messages

def init_model(model="gpt-3.5-turbo", base_url="http://192.168.0.24:8080"):
    return ChatOpenAI(model=model, base_url=base_url)

def invoke_a_model(model, messages):
    return model.invoke(messages)

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def invoke_with_history(model, messages, config):
    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    response = with_message_history.invoke(
        messages,
        config=config,
    )

    return response

def converse_with_history(model):
    config_1 = {"configurable": {"session_id": "abc2"}}
    config_2 = {"configurable": {"session_id": "abc4"}}

    messages = [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
    r = invoke_a_model(model, messages)
    print(f"\n2: {'#'*15} Query: Three messages\n{'='*10} {r.content}")

    messages = [HumanMessage(content="Hi, I'm Bob")]
    r = invoke_with_history(model, messages, config_1)
    print(f"\n3: {'#'*15} Query:history + Hi, I'm Bob.\nsession: abc2\n{'='*10} {r.content}")

    messages = [HumanMessage(content="What's my name?")]
    r = invoke_with_history(model, messages, config_1)
    print(f"\n4: {'#'*15} Query:history + {"What's my name?"}\nsession: abc2\n{'='*10} {r.content}")

    messages = [HumanMessage(content="What's my name?")]
    r = invoke_with_history(model, messages, config_2)
    print(f"\n5: {'#'*15} Query:history + {'What is my name?'}\nsession: abc4\n{'='*10} {r.content}")

    messages = [HumanMessage(content="What's my name?")]
    r = invoke_with_history(model, messages, config_1)
    print(f"\n6: {'#'*15} Query:history + {'What is my name?'}\nsession: abc2\n{'='*10} {r.content}")

def prompt_template_test(model):
    prompt = ChatPromptTemplate.from_messages(
                [ ("system", "You are a helpful assistant. Answer all questions to the best of your ability.",),
                  MessagesPlaceholder(variable_name="messages"),])

    chain = prompt | model
    response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})
    print(f"\n{response.content}")

    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    config = {"configurable": {"session_id": "abc5"}}
    response = with_message_history.invoke([HumanMessage(content="Hi! I'm Jim")], config=config,)
    print(f"\n{response.content}")

    response = with_message_history.invoke( [HumanMessage(content="What's my name?")], config=config, )
    print(f"\n{response.content}")

def prompt_variables(model):
    prompt = ChatPromptTemplate.from_messages(
                [ ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",),
                  MessagesPlaceholder(variable_name="messages"),])

    chain = prompt | model
    response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")], "language": "English"})
    print(f"\n{response.content}")

    with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages",)
    config = {"configurable": {"session_id": "abc6"}}
    response = with_message_history.invoke( {"messages":[HumanMessage(content="Hi! I'm Jim")], "language": "English"}, config=config,)
    print(f"\n{response.content}")

    response = with_message_history.invoke( {"messages":[HumanMessage(content="What's my name?")], "language": "English"}, config=config,)
    print(f"\n{response.content}")

def manage_history(model):

    prompt = ChatPromptTemplate.from_messages(
                [ ("system", "You are a helpful assistant. Answer all questions to the best of your ability.",),
                  MessagesPlaceholder(variable_name="messages"),])

    trimmer = trim_messages(
        max_tokens=95,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]

    r = trimmer.invoke(messages)
    pp(r)

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
    )

    response = chain.invoke(
        {
            "messages": messages + [HumanMessage(content="what's my name?")],
            "language": "English",
        }
    )
    print(f"\n{response.content}")

    new_messages = messages + [HumanMessage(content="what math problem did i ask")]
    r = trimmer.invoke(new_messages)
    pp(r)

    response = chain.invoke(
        {
            "messages": new_messages,
            "language": "English",
        }
    )
    print(f"\n{response.content}")


    with_message_history = RunnableWithMessageHistory(
                                chain,
                                get_session_history,
                                input_messages_key="messages",
                            )
    config = {"configurable": {"session_id": "abc20"}}
    response = with_message_history.invoke(
                    {
                        "messages": messages + [HumanMessage(content="whats my name?")],
                        "language": "English",
                    },
                    config=config,
                )
    print(f"\n{response.content}")

    response = with_message_history.invoke(
                        {
                            "messages": [HumanMessage(content="what math problem did i ask?")],
                            "language": "English",
                        },
                        config=config,
                    )
    print(f"\n{response.content}")

def streaming(model):
    prompt = ChatPromptTemplate.from_messages(
                [ ("system", "You are a helpful assistant. Answer all questions to the best of your ability.",),
                  MessagesPlaceholder(variable_name="messages"),])
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages"))
        | prompt
        | model
    )
    with_message_history = RunnableWithMessageHistory(
                                chain,
                                get_session_history,
                                input_messages_key="messages",
                            )
    config = {"configurable": {"session_id": "abc15"}}
    for r in with_message_history.stream(
        {
            "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
            "language": "English",
        },
        config=config,
    ):
        print(r.content, end="|")

    for r in with_message_history.stream(
        {
            "messages": [HumanMessage(content="안녕? 한국어로 웃긴 이야기를 해줘")],
            "language": "English",
        },
        config=config,
    ):
        print(r.content, end="|")
if __name__ == '__main__':
    model = ChatOpenAI(model="gpt-3.5-turbo", base_url="http://192.168.0.24:8080")

    r = invoke_a_model(model, [HumanMessage(content='Hi')])
    print(f"\n1: {'#'*15} Query:Hi\n{'='*10} {r.content}")
