from langchain_core.prompts import (
    MessagesPlaceholder,
    ChatPromptTemplate
)

#prompt = MessagesPlaceholder("history")
#prompt.format_messages() # raises KeyError

prompt = MessagesPlaceholder("history", optional=True)
messages = prompt.format_messages() # returns empty list []
print(messages)

messages = prompt.format_messages(
    history=[
        ("system", "You are an AI assistant."),
        ("human", "Hello!"),
    ]
)
print('\n', messages)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

r = prompt.invoke(
   {
       "history": [("human", "what's 5 + 2"), ("ai", "5 + 2 is 7")],
       "question": "now multiply that by 4"
   }
)
print('\n', r)