#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: danerlt 
@file: llm
@time: 2024-12-03
@contact: danerlt001@gmail.com
"""
import os
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms.llm import ChatMessage


def main():
    load_dotenv()
    model = os.environ.get("MODEL")
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    llm = OpenAILike(model=model, api_base=api_base, api_key=api_key, is_chat_model=True)
    query = "1+1=?"
    chat_response = llm.chat([
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content=query)
    ])
    res = chat_response.message.content
    print(res)


if __name__ == '__main__':
    main()
