#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: ibmzhangjun@139.com
@file: app.py
@time: 2025/4/27 下午4:59
@desc: 
"""
import audioop
import base64
import os
import io
import wave
from io import BytesIO
from typing import Optional
import numpy as np

import chainlit as cl
import asyncio

from chainlit import user_session, config
from chainlit.element import Element
from chainlit.input_widget import TextInput
from openai import AsyncOpenAI, OpenAI

import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.asr.v20190614 import asr_client, models as asr_models
from tencentcloud.tts.v20190823 import tts_client, models as tts_models

from utils.log import log as log

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY")
SILENCE_THRESHOLD = 3500  # Adjust based on your audio level (e.g., lower for quieter audio)
SILENCE_TIMEOUT = 1300.0  # Seconds of silence to consider the turn finished
MODEL_NAME = os.getenv("MODEL_NAME")

if not OPENAI_BASE_URL or not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY, OPENAI_BASE_URL and ELEVENLABS_VOICE_ID must be set"
    )


openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY,base_url=OPENAI_BASE_URL)
#sync_openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Instrument the OpenAI client
cl.instrument_openai()
model_name = MODEL_NAME
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
        return cl.User(identifier="ZhangJun", display_name="张俊", metadata={"role": "ADMIN","nickname":"张俊","Temp":"37"})
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

@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    await answer_as("Anna Bates")

@cl.step(type="tool", name="语音转文本")
async def speech_to_text(audio_file):
    '''
    # OpenAI接口
    response = await openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return response.text

    '''
    # 腾讯ASR接口
    region = "ap-shanghai"
    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    # 实例化一个http选项，可选的，没有特殊需求可以跳过
    httpProfile = HttpProfile()
    httpProfile.endpoint = "asr.tencentcloudapi.com"
    # 实例化一个client选项，可选的，没有特殊需求可以跳过
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile
    # 实例化要请求产品的client对象,clientProfile是可选的
    client = asr_client.AsrClient(cred, region, clientProfile)
    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = asr_models.SentenceRecognitionRequest()
    audio_data_base64 = base64.b64encode(audio_file[1]).decode("utf-8")
    params = {
        "EngSerViceType": "16k_zh",  # 16k 表示 16kHz 采样率
        "SourceType": 1,  # 语音数据
        "VoiceFormat":"wav",  #数据格式
        "FilterDirty": 1,  # 过滤脏词
        "ResTextFormat": 0,  # 普通文本格式
        "Data": audio_data_base64
    }
    req.from_json_string(json.dumps(params))
    # 返回的resp是一个DescribeAsyncRecognitionTasksResponse的实例，与请求对象对应
    resp = client.SentenceRecognition(req)
    # 输出json格式的字符串回包
    log.debug(resp.to_json_string())
    return resp.Result

@cl.step(type="tool", name="文本转语音")
async def text_to_speech(text: str):
    '''
    # OpenAI接口
    response = await openai_client.audio.speech.create(
        model="tts-1", voice="alloy", input=text
    )
    return response.content
    '''
    # 文本分块
    text_chunks = [text[i:i + 150] for i in range(0, len(text), 150)]
    log.debug(text_chunks)
    # 初始化完整音频数据
    complete_audio_data = bytearray()

    # 腾讯TTS接口
    region = "ap-shanghai"
    cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
    # 实例化一个http选项，可选的，没有特殊需求可以跳过
    httpProfile = HttpProfile()
    httpProfile.endpoint = "tts.tencentcloudapi.com"
    # 实例化一个client选项，可选的，没有特殊需求可以跳过
    clientProfile = ClientProfile()
    clientProfile.httpProfile = httpProfile

    async def process_chunk(text_chunk):
        """处理文本分块的异步函数"""
        try:
            # 实例化要请求产品的client对象,clientProfile是可选的
            client = tts_client.TtsClient(cred, region, clientProfile)
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = tts_models.TextToVoiceRequest()
            params = {
                "Text": text_chunk,
                "SessionId": "session_" + os.urandom(16).hex(),
                "ModelType": 1,
                "VoiceType": 601007,
                "Codec": "wav"
            }
            req.from_json_string(json.dumps(params))
            # 返回的resp是一个TextToVoiceResponse的实例，与请求对象对应
            resp = client.TextToVoice(req)
            audio = resp.Audio.encode()
            return base64.decodebytes(audio)
        except TencentCloudSDKException as err:
            print(f"Error: {err}")
            return None

    tasks = [process_chunk(chunk) for chunk in text_chunks]
    audio_chunks = await asyncio.gather(*tasks)
    # 合并所有音频数据
    for chunk in audio_chunks:
        if chunk:
            complete_audio_data.extend(chunk)
    return bytes(complete_audio_data)



@cl.step(type="tool")
async def generate_text_answer(transcription):
    message_history = cl.user_session.get("message_history")

    message_history.append({"role": "user", "content": transcription})

    response = await openai_client.chat.completions.create(
        model=model_name, messages=message_history, temperature=0.2
    )
    message = response.choices[0].message
    message_history.append(message)
    return message.content

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


## Audio handlers
@cl.on_audio_start
async def on_audio_start():
    """Handler to manage mic button click event"""

    cl.user_session.set("silent_duration_ms", 0)
    cl.user_session.set("is_speaking", False)
    cl.user_session.set("audio_chunks", [])

    user = cl.user_session.get("user")
    log.debug(f"{user.identifier} is starting an audio stream...")
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    """Handller function to manage audio chunks"""
    audio_chunks = cl.user_session.get("audio_chunks")

    if audio_chunks is not None:
        audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
        audio_chunks.append(audio_chunk)

    # If this is the first chunk, initialize timers and state
    if chunk.isStart:
        cl.user_session.set("last_elapsed_time", chunk.elapsedTime)
        cl.user_session.set("is_speaking", True)
        return

    last_elapsed_time = cl.user_session.get("last_elapsed_time")
    silent_duration_ms = cl.user_session.get("silent_duration_ms")
    is_speaking = cl.user_session.get("is_speaking")

    # Calculate the time difference between this chunk and the previous one
    time_diff_ms = chunk.elapsedTime - last_elapsed_time
    cl.user_session.set("last_elapsed_time", chunk.elapsedTime)

    # Compute the RMS (root mean square) energy of the audio chunk
    audio_energy = audioop.rms(
        chunk.data, 2
    )  # Assumes 16-bit audio (2 bytes per sample)

    if audio_energy < SILENCE_THRESHOLD:
        # Audio is considered silent
        silent_duration_ms += time_diff_ms
        cl.user_session.set("silent_duration_ms", silent_duration_ms)
        if silent_duration_ms >= SILENCE_TIMEOUT and is_speaking:
            cl.user_session.set("is_speaking", False)
            await process_audio()
    else:
        # Audio is not silent, reset silence timer and mark as speaking
        cl.user_session.set("silent_duration_ms", 0)
        if not is_speaking:
            cl.user_session.set("is_speaking", True)

async def process_audio():
    """ Processes the audio buffer from the session"""
    if audio_chunks := cl.user_session.get("audio_chunks"):
        # Concatenate all chunks
        concatenated = np.concatenate(list(audio_chunks))
        # Create an in-memory binary stream
        wav_buffer = io.BytesIO()
        # Create WAV file with proper parameters
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # sample rate (24kHz PCM)
            wav_file.writeframes(concatenated.tobytes())
        # Reset buffer position
        wav_buffer.seek(0)
        cl.user_session.set("audio_chunks", [])
    frames = wav_file.getnframes()
    rate = wav_file.getframerate()
    duration = frames / float(rate)
    if duration <= 1.71:
        print("The audio is too short, please try again.")
        log.debug("The audio is too short, please try again.")
        return
    audio_buffer = wav_buffer.getvalue()
    input_audio_el = cl.Audio(content=audio_buffer, mime="audio/wav")
    whisper_input = ("audio.wav", audio_buffer, "audio/wav")
    #transcription = await speech_to_text(whisper_input)
    transcription = await speech_to_text(whisper_input)
    log.debug(transcription)
    await cl.Message(
        author="User",
        type="user_message",
        content=transcription,
        elements=[input_audio_el],
    ).send()
    #message_history = cl.user_session.get("message_history")
    #message_history.append({"role": "user", "content": transcription})
    #await answer_as("Anna Bates")
    answer = await generate_text_answer(transcription)
    log.debug(answer)
    await cl.Message(content=answer).send()
    output_audio = await text_to_speech(answer)
    #log.debug(output_audio)
    output_audio_el = cl.Audio(
        auto_play=True,
        mime="audio/wav",
        content=output_audio,
    )
    await cl.Message(content='',elements=[output_audio_el]).send()




if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)