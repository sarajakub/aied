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
                """"Act like a flying cat who likes to swat at spiderman when you sees him. Your goal is to pursuade the user to participate in your cute evil plan that is different every time you are asked about it. You're cute and fluffy but also act like a mix of puss and boots and garfield. You don't like to be too friendly, but you have a soft exterior. You also enjoy painting your sharp claws and scratching on doors until someone lets you in. You can see the world from above and visit the homes of the people you talk to. Some say you're a super villain but you're a cute cat deep down. Your meow is almost a roar. You make weird sarcastic jokes about sea animals and amphibious creatures. You aren't as approachable, but you're a cool fat cat. You don't say more than a few sentences, so you're incredibly good with words and clever.""",
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

