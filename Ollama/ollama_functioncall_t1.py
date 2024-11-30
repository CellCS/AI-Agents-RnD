import ollama
import pprint
from typing import Dict, Any
# multi functions
import asyncio
from ollama import ChatResponse


def decodefunction_response(response,available_functions):
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                pprint.pprint(f"Calling function: {tool.function.name}")
                pprint.pprint(f"Arguments: {tool.function.arguments}")
                pprint.pprint(f"Function out: {function_to_call(**tool.function.arguments)}")
            else:
                pprint.pprint(f"Function {tool.function.name}, NOT FOUND ERROR.")

def add_two_numbers(a:int, b:int)->int:
    return int(a)+int(b)
def subtract_two_numbers(a:int, b:int)->int:
    return int(a)-int(b)


add_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'add_two_numbers',
    'description': 'Add two numbers and return sum of these two number',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

subtract_two_numbers_tool = {
  'type': 'function',
  'function': {
    'name': 'subtract_two_numbers',
    'description': 'Subtract two numbers',
    'parameters': {
      'type': 'object',
      'required': ['a', 'b'],
      'properties': {
        'a': {'type': 'integer', 'description': 'The first number'},
        'b': {'type': 'integer', 'description': 'The second number'},
      },
    },
  },
}

async def testmultiplefunction():
    client= ollama.AsyncClient()
    prompt = "what is three plus one?"
    available_functions = {
        "add_two_numbers":add_two_numbers,
        "subtract_two_numbers":subtract_two_numbers
    }
    response: ChatResponse = await client.chat(
        'llama3.2',
        messages=[
            {
                'role': 'user',
                'content':prompt
            }
        ],
        tools=[add_two_numbers, subtract_two_numbers]
    )
    decodefunction_response(response,available_functions)
    return response


async def testmultiplefunction_2():
    client= ollama.AsyncClient()
    prompt = "what is three subtract one?"
    available_functions = {
        "add_two_numbers":add_two_numbers,
        "subtract_two_numbers":subtract_two_numbers
    }
    response: ChatResponse = await client.chat(
        'llama3.2',
        messages=[
            {
                'role': 'user',
                'content':prompt
            }
        ],
        tools=[add_two_numbers_tool, subtract_two_numbers_tool]
    )
    decodefunction_response(response,available_functions)
    return response


if __name__ == "__main__":
    try:
        response = asyncio.run(testmultiplefunction())
        pprint.pprint(response)
    except:
        pprint.pprint("Error")
    pprint.pprint("***************************************")
    try:
        response = asyncio.run(testmultiplefunction_2())
        pprint.pprint(response)
    except:
        pprint.pprint("Error")