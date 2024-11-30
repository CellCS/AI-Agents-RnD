import ollama
import pprint
import json
import base64
import requests
import yfinance as yfintech
from typing import Dict, Any, Callable
# multi functions
import asyncio
from ollama import ChatResponse

# Reference: https://www.youtube.com/watch?v=QzRPtorrZVo


def decodefunction_response(response,available_functions):
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                print("Calling function: ", tool.function.name)
                print("Arguments: ", tool.function.arguments)
                print("Function out: ", function_to_call(**tool.function.arguments))
            else:
                print("Function", tool.function.name, 'NOT FOUND ERROR.')

def get_stock_price(symbol:str) -> float:
    ticker = yfintech.Ticker(symbol)
    return ticker.info.get('regularMarketPrice') or ticker.fast_info.last_price

def add_two_numbers(a:int, b:int)->int:
    return a+b
def substract_two_numbers(a:int, b:int)->int:
    return a-b

async def testmultiplefunction():
    client= ollama.AsyncClient()
    prompt = "what is three plus one?"
    available_functions = {
        "add_two_numbers":add_two_numbers,
        "substract_two_numbers":substract_two_numbers
    }
    response: ChatResponse = await client.chat(
        'llama3.2',
        messages=[
            {
                'role': 'user',
                'content':prompt
            }
        ],
        tools=[add_two_numbers, substract_two_numbers]
    )
    decodefunction_response(response,available_functions)
    return response


if __name__ == "__main__":
    prompt="What is the current stock price of Apple?"
    available_functions: Dict[str,Callable]={
        'get_stock_price':get_stock_price
    }
    response = ollama.chat(
        'llama3.2',
        messages=[
            {
                'role': 'user',
                'content':prompt
            }
        ],
        tools=[get_stock_price]
    )
    print(response)
    decodefunction_response(response,available_functions)


    
    response = ollama.chat(
        'llama3.2',
        messages=[
            {
                'role': 'user',
                'content':"why the sky is blue?"
            }
        ]
    )
    print(response)

    try:
        response = asyncio.run(testmultiplefunction())
        print(response)
        decodefunction_response(response)
    except:
        print("Error")