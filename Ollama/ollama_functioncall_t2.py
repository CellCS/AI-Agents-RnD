import ollama
import pprint
import json
import base64
import requests
import yfinance as yfintech
from typing import Dict, Any, Callable

def decodefunction_response(response,available_functions):
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                pprint.pprint(f"Calling function: {tool.function.name}")
                pprint.pprint(f"Arguments: {tool.function.arguments}")
                pprint.pprint(f"Function out: {function_to_call(**tool.function.arguments)}")
            else:
                pprint.pprint(f"Function {tool.function.name}, NOT FOUND ERROR.")

def get_stock_price(symbol:str) -> float:
    ticker = yfintech.Ticker(symbol)
    pprint.pprint(f"symbol===={symbol}")
    return ticker.info.get('regularMarketPrice') or ticker.fast_info.last_price


def get_two_stockprice_difference(a:float, b:float)->float:
    return float(a)-float(b)


def testonefunctioncall_getstockprice(prompt):
    pprint.pprint(f"*************prompt: {prompt}**************************")
    available_functions: Dict[str,Callable]={
        'get_stock_price':get_stock_price
    }
    try:
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
        decodefunction_response(response,available_functions)
        pprint.pprint(response)
        return response
    except:
        pprint.pprint("Error")
        



if __name__ == "__main__":
    prompt="What are the current stock price of Microsoft, APPLE and Nvidia?"
    testonefunctioncall_getstockprice(prompt)
    # prompt="Compare the current stock price for META and APPLE."
    # testonefunctioncall_getstockprice(prompt)