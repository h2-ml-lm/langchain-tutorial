from operator import itemgetter
from typing import List
from typing import Optional

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from pprint import pp

SYSTEM_PROMPT = "You're an assistant who's good at {ability}"
HUMAN_PROMPT = "{question}"

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {}
def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def init_prompt_1():
    return ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_PROMPT),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", HUMAN_PROMPT),
                ])

def init_prompt_2():
    system_template=PromptTemplate( input_variables=["context"], template=SYSTEM_PROMPT)
    system_prompt = SystemMessagePromptTemplate( prompt=system_template )

    human_template = PromptTemplate( input_variables=["question"], template=HUMAN_PROMPT )
    human_prompt = HumanMessagePromptTemplate(prompt=human_template)

    history_prompt = MessagesPlaceholder(variable_name="history")

    messages = [system_prompt, history_prompt, human_prompt]
    chat_prompt_template = ChatPromptTemplate(  input_variables=["context", "question"],
                                                messages=messages )
    return chat_prompt_template

def build_history_chain_1(prompt, model):
    chain = prompt | model
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history

    
def build_history_chain_2(prompt, model):
    chain = (
        #{ "ability": RunnablePassthrough(), "question": RunnablePassthrough()}
        prompt
        | model
        #| StrOutputParser()
    )
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history

def test_history_chain(chain_with_history):
    response = chain_with_history.invoke(  # noqa: T201
        {"ability": "math", "question": "What does cosine mean?"},
        config={"configurable": {"session_id": "foo"}})
    print(f"{'#'*15}\nQuery: What does cosine mean?\n{'='*10}\n{response.content}")
    print(f"\n{'-'*15} History:\n{store}\n\n")  # noqa: T201

    response = chain_with_history.invoke(  # noqa: T201
        {"ability": "math", "question": "What's its inverse?"},
        config={"configurable": {"session_id": "foo"}} )
    print(f"{'#'*15}\nQuery: What's its inverse?\n{'='*10}\n{response.content}")
    print(f"\n{'-'*15} History:\n{store}\n\n")  # noqa: T201

if __name__ == "__main__":
    history = get_by_session_id("1")
    history.add_message(AIMessage(content="hello"))
    print(f"\n{store}\n\n")  # noqa: T201

    model = ChatOpenAI(model="gpt-4o-mini", base_url="http://192.168.0.24:8080")

    #prompt_1 = init_prompt_1()
    #chain_1 = build_history_chain_1(prompt_1, model)
    #test_history_chain(chain_1)

    prompt_2 = init_prompt_2()
    chain_2 = build_history_chain_2(prompt_2, model)
    test_history_chain(chain_2)