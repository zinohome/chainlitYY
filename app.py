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

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if (username, password) == ("zhangjun", "passw0rd"):
        return cl.User(identifier="ZhangJun", metadata={"role": "ADMIN","nickname":"张俊","Temp":"37"})
    else:
        return None

@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    if current_user.metadata["role"] != "ADMIN":
        return None

    return [
        cl.ChatProfile(
            name="小研",
            icon="/public/profileimg/XiaoYan2.png",
            markdown_description="我是您的健康助手小研，我总是在这里，随时准备帮助您。如果您有任何不舒服或者有任何需要，随时告诉我。",
            starters=[
                cl.Starter(
                    label="聊天解闷",
                    message="有点无聊，和我聊聊天吧。",
                    icon="/public/startterimg/chat.png",
                ),
                cl.Starter(
                    label="健康报告",
                    message="获取我的健康报告。",
                    icon="/public/startterimg/report.png",
                ),
                cl.Starter(
                    label="健康知识",
                    message="给我介绍一些与我有关的健康知识吧。",
                    icon="/public/startterimg/knowledge.png",
                ),
                cl.Starter(
                    label="询医问药",
                    message="我有点不舒服，帮我查查该吃什么药或者该去哪个医院看医生。",
                    icon="/public/startterimg/doctor.png",
                )
                # TODO 增加健康日程，点击查看日程表等；
                # TODO 我的健康搭子  -- 社交属性
            ],
        )
    ]
@cl.on_chat_start
async def start_chat():
    #TODO：根据用户选择的助手指定system role提示词
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
            TextInput(id="bloodpress", label="血压", initial="120/90"),
            TextInput(id="AgentName", label="体温", initial="37.5"),
            TextInput(id="AgentName", label="血糖", initial="200"),
            TextInput(id="AgentName", label="血脂", initial="100"),
        ]
    ).send()
    value = settings["AgentName"]
    app_user = cl.user_session.get("user")

    #TODO：根据时间、时令等外界条件以及用户选择助手的风格发送问候语
    #await cl.Message(content=f"你好呀,{app_user.metadata['nickname']}。今天心情怎么样？", author='Gilfoyle').send()



async def answer_as(name):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(author=name, content="")
    # TODO：定制助手风格库，与问候语，settings配合使用，后续加上Memeory
    stream = await openai_client.chat.completions.create(
        model=model_name,
        messages=message_history + [{"role": "assistant", "content": f"你是一位关心、耐心且善解人意的照护助手，有比较强的医疗数据分析能力，并且具有很强的心理分析能力，很擅长使用温暖，清晰简单的语言与用户沟通。在对话的过程中需要分析用户说话的情绪状态,并用简体中文回应，语气要温暖，使用清晰简单的语言，适合中老年人理解。speak as {name}"}],
        stream=True,
        **settings,
    )
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    # Need to add the information that it was the author who answered but OpenAI only allows assistant.
    # simplified for the purpose of the demo.
    message_history.append({"role": "user", "content": msg.content})
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    await answer_as("Anna Bates")

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)