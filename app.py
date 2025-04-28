#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: ibmzhangjun@139.com
@file: app.py
@time: 2025/4/27 下午4:59
@desc: 
"""
import os
import io
from typing import Optional

import chainlit as cl
import asyncio

from chainlit import user_session, config
from chainlit.input_widget import TextInput
from openai import AsyncOpenAI, OpenAI

from utils.log import log as log

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Instrument the OpenAI client
cl.instrument_openai()
model_name = "gpt-3.5-turbo"
settings = {
    "temperature": 0.3,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="大模型提高软件测试效率",
            message="详细介绍如何借助大语言模型提高软件测试效率。",
            icon="/public/apidog.svg",
        ),
        cl.Starter(
            label="自动化测试思路",
            message="详细描述一下接口及UI自动化测试的基本思路。",
            icon="/public/pulumi.svg",
        ),
        cl.Starter(
            label="性能测试分析及瓶颈定位思路",
            message="详细描述一下软件性能测试分析及瓶颈定位的核心思路。",
            icon="/public/godot_engine.svg",
        ),
        cl.Starter(
            label="如何学习大模型应用的核心技术",
            message="给出学习大语言模型的一些重要的技术和方法。",
            icon="/public/gleam.svg",
        )
    ]

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("zhangjun", "passw0rd"):
        return cl.User(identifier="ZhangJun", metadata={"role": "ADMIN","nickname":"张俊","Temp":"37"})
    else:
        return None

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [
            {
                "role": "system",
                "content": "你是一位关心、耐心且善解人意的照护助手，有比较强的医疗数据分析能力，并且具有很强的心理分析能力，很擅长使用温暖，清晰简单的语言与用户沟通。在对话的过程中需要分析用户说话的情绪状态,并用简体中文回应，语气要温暖，使用清晰简单的语言，适合中老年人理解。请考虑他们的情绪状态来回应。",
            }
        ],
    )
    settings = await cl.ChatSettings(
        [
            TextInput(id="AgentName", label="Agent Name", initial="AI"),
        ]
    ).send()
    value = settings["AgentName"]
    app_user = cl.user_session.get("user")
    await cl.Message(content=f"你好呀,{app_user.metadata['nickname']}。今天心情怎么样？", author='Gilfoyle').send()



async def answer_as(name):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(author=name, content="")

    stream = await openai_client.chat.completions.create(
        model=model_name,
        messages=message_history + [{"role": "user", "content": f"speak as {name}"}],
        stream=True,
        **settings,
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # Need to add the information that it was the author who answered but OpenAI only allows assistant.
    # simplified for the purpose of the demo.
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    await answer_as("Gilfoyle")

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)