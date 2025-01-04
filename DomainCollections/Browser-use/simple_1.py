from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio

llm = ChatOpenAI(model="gpt-4o",api_key="")
async def main():
    agent = Agent(
        task="Find a one-way flight from Bali to Oman on 12 January 2025 on Google Flights. Return me the cheapest option.",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())