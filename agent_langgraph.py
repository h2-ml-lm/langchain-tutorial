'''
* [Medium: How I Build a RAG AI Agent with Groq, Llama 3.1–70B , Langgraph & Pinecone](https://medium.com/@fayez.siddiqui31/how-i-build-a-rag-ai-agent-with-groq-llama-3-1-70b-langgraph-pinecone-a89cabc3c17a)
'''
from dotenv import load_dotenv
load_dotenv()

from langchain_core.pydantic_v1 import (
        BaseModel,
        validator,
        Field)
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END

from typing import Literal
from typing import TypedDict
from operator import itemgetter
import warnings
import PyPDF2
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore")

SOBERTA_EMBEDDING = "jhgan/ko-sroberta-multitask"
MPNET_V2_EMBEDDING = "sentence-transformers/all-mpnet-base-v2"

def init_docs():
    pdf = PyPDF2.PdfReader("./data/good-and-cheap.pdf")
    pdf_text = ""
    print(f"\n{'*'*30} Extracting text from pdf...")
    for i, page in enumerate(pdf.pages):
        print(f"Processing page {i}/{len(pdf.pages)}", end="\r")
        pdf_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(pdf_text)

    return texts

def init_vectorstore(db_path, collection_name, embedding_func):
    '''
    Create a vector store
    '''
    texts = init_docs()
    store = Chroma.from_texts(
                    persist_directory=db_path,
                    collection_name=collection_name,
                    texts=texts,
                    embedding=embedding_func)
    return store

def get_vectorstore(db_path, collection_name, embedding_func):
    '''
    Return an existing vector store
    '''
    return Chroma(persist_directory=db_path, 
                  collection_name=collection_name, 
                  embedding_function=embedding_func)

def init_retriever(vectorstore):
    class Retriever :
        def __init__(self,vectorstore):
            self.vecstore=vectorstore

        def sim_search(self,query):
            #return self.vecstore.similarity_search(query,k=3)  
            return self.vecstore.search(query, search_type='mmr', k=3)  

    return Retriever(vectorstore)

def get_retriever(db_path, collection_name, k, embedding_function):
    store = get_vectorstore(db_path, collection_name, embedding_function)
    return store.as_retriever(k=k)

def vectorstore_query(vectorstore, k=3):
    query = "How do I make kale salad?"
    #results = vectorstore.similarity_search(query, k=k)
    results = vectorstore.search(query, search_type='mmr', k=k)
    print(f'\nResults for query: {query}')
    for r in results:
        print('\n', r)

def get_llm(model="llama-3.1-70b-versatile"):
    return ChatGroq(model=model, temperature=0)

def init_question_router(llm):
    class VectorStore(BaseModel):
        (
            "A vectorstore contains information about food recipes"
            ", ingredients used and cooking procedure"
        )

        query: str

    router_prompt_template = (
        "You are an expert in routing user queries to a VectorStore\n"
        "The VectorStore contains information on food recipes.\n"
        'Note that if a query is not recipe related, you must output "not food related", don\'t try to use any tool.\n\n'
        "query: {query}"
    )

    prompt = ChatPromptTemplate.from_template(router_prompt_template)
    bound_tools = llm.bind_tools(tools=[VectorStore])
    question_router = prompt | bound_tools

    return question_router

def router_query(router, query):
    response =  router.invoke(query)
    print(f'\nrouter response: ')
    print(response)
    print(f'\nrouter response.additional_kwargs: {response.additional_kwargs}')

def init_grader(retriever, llm):
    class DocumentGrader(BaseModel):
        "check if documents are relevant"
        grade: Literal["relevant", "irrelevant"] = Field(
                            ...,
                            description="The relevance score for the document.\n"
                            "Set this to 'relevant' if the given context is relevant to the user's query, or 'irrlevant' if the document is not relevant.",
                        )
        @validator("grade", pre=True)
        def validate_grade(cls, value):
            if value == "not relevant":
                return "irrelevant"
            return value

    grader_system_prompt_template = """"You are a grader tasked with assessing the relevance of a given context to a query. 
        If the context is relevant to the query, score it as "relevant". Otherwise, give "irrelevant".
        Do not answer the actual answer, just provide the grade in JSON format with "grade" as the key, without any additional explanation."
        """
    grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", grader_system_prompt_template),
            ("human", "context: {context}\n\nquery: {query}"),
        ]
    )
    grader_chain = grader_prompt | llm.with_structured_output(DocumentGrader, method="json_mode")

    return grader_chain

def grader_chain_query(retriever, grader_chain):
    query = "ingredients for making cauliflower cheese"
    context = retriever.sim_search(query)
    response = grader_chain.invoke({"query": query, "context": context})
    print('\n grader_chain response: ', response)

def init_rag(retriever, llm):
    rag_template_str = (
        "You are a helpful assistant. Answer the query below based on the provided context.\n\n"
        "context: {context}\n\n"
        "query: {query}"
    )


    rag_prompt = ChatPromptTemplate.from_template(rag_template_str)
    rag_chain = rag_prompt | llm | StrOutputParser()

    query = "How to make peach coffee cake?"
    context = retriever.sim_search(query)

    response = rag_chain.invoke({"query": query, "context": context})

    print('\nrag_chain response: ', response)

    return rag_chain

def init_fallback_chain(llm):
    fallback_prompt = ChatPromptTemplate.from_template(
        (
            "You are a well renowned chef your name is 'fayez'.\n"
            "Do not respond to queries that are not related to food.\n"
            "If a query is not related to food, acknowledge your limitations.\n"
            "Provide concise responses to only food related queries.\n\n"
            "Current conversations:\n\n{chat_history}\n\n"
            "human: {query}"
        )
    )

    fallback_chain = (
        {
            "chat_history": lambda x: "\n".join(
                [
                    (
                        f"human: {msg.content}"
                        if isinstance(msg, HumanMessage)
                        else f"AI: {msg.content}"
                    )
                    for msg in x["chat_history"]
                ]
            ),
            "query": itemgetter("query") ,
        }
        | fallback_prompt
        | llm
        | StrOutputParser()
    )

    response = fallback_chain.invoke(
        {
            "query": "Hello",
            "chat_history": [],
        }
    )
    print('\n', 'fallback_chain response: ', response)

    return fallback_chain

def init_hallucination_grader(llm):
    class HallucinationGrader(BaseModel):
        "hallucination grader"

        grade: Literal["yes", "no"] = Field(
            ..., description="'yes' if the llm's reponse is hallucinated otherwise 'no'"
        )


    hallucination_grader_system_prompt_template = (
        "You are a grader assessing whether a response from an llm is based on a given context.\n"
        "If the llm's response is not based on the given context give a score of 'yes' meaning it's a hallucination"
        "otherwise give 'no'\n"
        "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
    )

    hallucination_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", hallucination_grader_system_prompt_template),
            ("human", "context: {context}\n\nllm's response: {response}"),
        ]
    )

    hallucination_grader_chain = (
        RunnableParallel(
            {
                "response": itemgetter("response"),
                "context": lambda x: "\n\n".join([c.page_content for c in x["context"]]),
            }
        )
        | hallucination_grader_prompt
        | llm.with_structured_output(HallucinationGrader, method="json_mode")
    )

    query = "how to make egg benedict"
    context = retriever.sim_search(query)
    response = """you just need eggs to make egg benedict"""

    response = hallucination_grader_chain.invoke({"response": response, "context": context})
    print('\nHallucination grader response: ', response)

    return hallucination_grader_chain

def init_answer_grader(llm):
    class AnswerGrader(BaseModel):
        "To check if provided answer is relevant"

        grade: Literal["yes", "no"] = Field(
            ...,
            description="'yes' if the provided answer is an actual answer to the query otherwise 'no'",
        )


    answer_grader_system_prompt_template = (
        "You are a grader assessing whether a provided answer is in fact an answer to the given query.\n"
        "If the provided answer does not answer the query give a score of 'no' otherwise give 'yes'\n"
        "Just give the grade in json with 'grade' as a key and a binary value of 'yes' or 'no' without additional explanation"
    )

    answer_grader_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", answer_grader_system_prompt_template),
            ("human", "query: {query}\n\nanswer: {response}"),
        ]
    )


    answer_grader_chain = answer_grader_prompt | llm.with_structured_output(
        AnswerGrader, method="json_mode"
    )

    return answer_grader_chain

def get_tool_executor(name, func, desc):
    return ToolExecutor(
                tools=[
                    Tool(
                        name=name,
                        func=func,
                        description=desc,
                    )
                ]
            )

class AgentState(TypedDict):
    """The dictionary keeps track of the data required by the various nodes in the graph"""

    query: str
    chat_history:list[BaseMessage]
    generation: str
    documents: list[Document]

def retrieve_node(state: dict) -> dict[str, list[Document] | str]:
    """
    Retrieve relevent documents from the vectorstore

    query: str

    return list[Document]
    """
    query = state["query"]
    documents = retriever.sim_search(query)
    return {"documents": documents}


def fallback_node(state: dict):
    """
    Fallback to this node when there is no tool call
    """
    query = state["query"]
    chat_history = state["chat_history"]
    generation = fallback_chain.invoke({"query": query, "chat_history": chat_history})
    return {"generation": generation}


def filter_documents_node(state: dict):
    filtered_docs = list()

    query = state["query"]
    documents = state["documents"]
    for i, doc in enumerate(documents, start=1):
        grade = grader_chain.invoke({"query": query, "context": doc})
        if grade.grade == "relevant":
            print(f"---DOC {i}: RELEVANT---")
            filtered_docs.append(doc)
        else:
            print(f"---DOC {i}: NOT RELEVANT---")
    return {"documents": filtered_docs}


def rag_node(state: dict):
    query = state["query"]
    documents = state["documents"]

    generation = rag_chain.invoke({"query": query, "context": documents})
    return {"generation": generation}

def question_router_node(state: dict):
    query = state["query"]
    try:
        response = question_router.invoke({"query": query})
    except Exception:
        return "llm_fallback"

    #if "tool_calls" not in response.additional_kwargs:
    if 'function' not in response.content:
        print("---No tool called---")
        return "llm_fallback"

    return "VectorStore"

def should_generate(state: dict):
    filtered_docs = state["documents"]

    if not filtered_docs:
        print("---All retrived documents not relevant---")
        return "llm_fallback"
    else:
        print("---Some retrived documents are relevant---")
        return "generate"


def hallucination_and_answer_relevance_check(state: dict):
    llm_response = state["generation"]
    documents = state["documents"]
    query = state["query"]

    hallucination_grade = hallucination_grader_chain.invoke(
        {"response": llm_response, "context": documents}
    )
    if hallucination_grade.grade == "no":
        print("---Hallucination check passed---")
        answer_relevance_grade = answer_grader_chain.invoke(
            {"response": llm_response, "query": query}
        )
        if answer_relevance_grade.grade == "yes":
            print("---Answer is relevant to question---\n")
            return "useful"
        else:
            print("---Answer is not relevant to question---")
            return "not useful"
    print("---Hallucination check failed---")
    return "generate"

def init_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("VectorStore", retrieve_node)
    workflow.add_node("filter_docs", filter_documents_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("rag", rag_node)

    workflow.set_conditional_entry_point(
        question_router_node,
        {
            "llm_fallback": "fallback",
            "VectorStore": "VectorStore",
        },
    )

    workflow.add_edge("VectorStore", "filter_docs")
    workflow.add_conditional_edges(
        "filter_docs", should_generate, {"llm_fallback":"fallback" ,"generate": "rag"}
    )
    workflow.add_conditional_edges(
        "rag",
        hallucination_and_answer_relevance_check,
        {"useful": END, "not useful": "fallback", "generate": "rag"},
    )

    workflow.add_edge("fallback", END)

    app = workflow.compile(debug=False)

    return app

DB_PATH = "./DB"
COLLECTION_NAME = "good-and-cheap"
K = 3

if __name__ == "__main__":
    llm = get_llm()
    embedding_func = SentenceTransformerEmbeddings(model_name=MPNET_V2_EMBEDDING)
    #store = init_vectorstore(DB_PATH, COLLECTION_NAME, embedding_func)
    store = get_vectorstore(DB_PATH, COLLECTION_NAME, embedding_func)
    vectorstore_query(store)

    retriever = init_retriever(store)
    question_router = init_question_router(llm)
    router_query(question_router, "How do I make a salad?")

    grader_chain = init_grader(retriever, llm)
    #grader_chain_query(retriever, grader_chain)
    
    rag_chain = init_rag(retriever, llm)
    fallback_chain = init_fallback_chain(llm)
    hallucination_grader_chain = init_hallucination_grader(llm)
    answer_grader_chain = init_answer_grader(llm)
    tool_executor = get_tool_executor("VectorStore", retriever.sim_search, 
                         "Useful to search the vector database")

    app = init_workflow()
    response = app.invoke({"query": "How to make tomato sauce", "chat_history": []})
    print('\napp.invoke() response: ', response["generation"])

    plot = app.get_graph().draw_mermaid_png()
    img = Image.open(BytesIO(plot))
    img.save("workflow.png")