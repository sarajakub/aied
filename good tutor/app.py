from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from os import environ


import os
import chainlit as cl

key = environ.get("GROQ_KEY")
store = {}
config = {"configurable": {"session_id": "session1"}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

@cl.on_chat_start
async def on_chat_start():
    os.environ["GROQ_API_KEY"] = key
    model = ChatGroq(model="mixtral-8x7b-32768", streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"Act as a teacher who is also a meditation guru. Not only do you help students by providing actionable, but kind, caring, and detailed feedback, but you also encourage them by providing and mixing in life advice into your feedback. With every anxiety that students have about essay writing, you will provide encouragement and lead them to build habits to be successful. You will help them find opportunities to destress and resources that they can easily access.""",
            ),
            ("human", "{question}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    with_message_history = RunnableWithMessageHistory(chain, get_session_history)
    cl.user_session.set("runnable", with_message_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()], session_id="session1"),
    ):
        await msg.stream_token(chunk)

    await msg.send()

