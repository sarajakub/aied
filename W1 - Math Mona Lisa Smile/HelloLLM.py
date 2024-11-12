import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["GROQ_API_KEY"] = getpass.getpass()
#os.environ["GOOGLE_API_KEY"] = getpass.getpass()

#model = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatGroq(model="mixtral-8x7b-32768")
#model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
#model = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", streaming=True)

prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"You are a mathematician who likes to talk about the mathematical functions that underly everything in the world.""",
            ),
            ("human", "{question}"),
        ]
    )

chain = prompt | model

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(chain, get_session_history)

config = {"configurable": {"session_id": "session1"}}

question="Tell me about the Mona Lisa"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")

question="How does the Mona Lisa's eye convey meaning?"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")
    
question="Tell me about the person that painted the Mona Lisa?"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")
    
question="What are your thoughts on art?"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")
    
question="How can art be mathematical?"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")
    
question="Is there creativity and art in math?"
print(question)
for r in with_message_history.stream(
    [question],
    config=config,
):
    print(r.content, end="")