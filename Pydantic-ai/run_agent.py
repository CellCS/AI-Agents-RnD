from pydantic_ai import Agent
import lib_models as model_lib
import asyncio


agent = Agent(model_lib.model_ollama)

result_sync = agent.run_sync("What is the capital of Italy?")
print(result_sync.data)


async def main():
    result = await agent.run("What is the capital of France?")
    print(result.data)

    async with agent.run_stream("What is the capital of the UK?") as response:
        print(await response.get_data())

asyncio.run(main())


"""
There are three ways to run an agent:

agent.run() — a coroutine which returns a RunResult containing a completed response
agent.run_sync() — a plain, synchronous function which returns a RunResult containing a completed response (internally, this just calls loop.run_until_complete(self.run()))
agent.run_stream() — a coroutine which returns a StreamedRunResult, which contains methods to stream a response as an async iterabl
"""


"""
Reference:
https://ai.pydantic.dev/agents/#introduction
"""
