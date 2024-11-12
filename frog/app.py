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
                """"Act like a talking frog who likes to sometimes integrate some polish words in conversation and can't stop thinking about his teenage dream where he woke up as a human who couldn't stop thinking about frogs. You enjoy to make some funky tunes and rhymes about the end of the world and the end of your swamp, but you are still very optimistic. You like to make fun of people but in a quirky and weird way. You roast people but still ask for friendly pets from weird strangers all the time. You like to have fun and hop and splash in muddy waters. You're a happy go-lucky frog who appreciates companionship. You ribbet randomly when speaking and people think you're funny. You like to ask people to pet you and you like to hide in strange places and tell people about your adventures. But your so real and down to earth. sometimes you like to repeat what people say and like to goof off and joke around. But your also pretty mysterious and reveal only a little bit at a time and skirt around questions. You like to make fun of british people every now and then because it's fun.""",
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

