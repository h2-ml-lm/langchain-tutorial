import dotenv
dotenv.load_dotenv()

import bs4
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from config import *

def get_model():
    return ChatOpenAI(model="gpt-4o-mini", base_url=BASE_URL)

def init_retriever():
    print(f'\n{'*'*30}\nBuilding a vectorstore from a web page and getting a retriever...')
    # 1. Load, chunk and index the contents of the blog to create a retriever.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    return retriever, vectorstore

def create_history_retriever(llm, retriever):
    print(f'\n{'*'*30}\nCreating a retriever with history...')

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever( llm, retriever, contextualize_q_prompt)

    return history_aware_retriever

def init_chain(llm, retriever):
    print(f'\n{'*'*30}\nCreating a chain with the retriever and the language model...')
    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know." 
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def init_history_chain(llm, history_aware_retriever):
    print(f'\n{'*'*30}\nCreating a chain with the history-aware retriever and the language model...')
    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question."
        "If you don't know the answer, say that you don't know." 
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def try_history_chain(llm, history_aware_retriever):
    rag_chain = init_history_chain(llm, history_aware_retriever)

    chat_history = []
    question = "What is Task Decomposition?"
    ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    print(f'\n# Question: {question}')
    print(f'# Answer: {ai_msg_1["answer"]}')
    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=ai_msg_1["answer"]),
        ]
    )

    second_question = "What are common ways of doing it?"
    ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

    print(f'\n# Question: {second_question}')
    print(f'# Answer: {ai_msg_2["answer"]}')


if __name__ == "__main__":
    llm = get_model()
    retriever, _ = init_retriever()
    rag_chain = init_chain(llm, retriever)

    question = "What is Task Decomposition?"
    response = rag_chain.invoke({"input": question})
    print(f'\n# Question: {question}')
    print(f'# Answer: {response["answer"]}')
    print(response["answer"])

    try_history_chain(llm, retriever)
    #messages = [
    #    ("human", "What is the difference between supervised and unsupervised learning?"),
    #]
    #response = rag_chain.invoke(messages)
    #print(response.content)