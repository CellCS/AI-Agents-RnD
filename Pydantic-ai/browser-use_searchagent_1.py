from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
)  # pip install -qU langchain-nvidia-ai-endpoints
from browser_use import Agent
import asyncio
import lib_models as model_lib
from browser_use.controller.service import Controller

openai_llm = ChatOpenAI(model="gpt-4o", api_key=model_lib.openai_key)

ollama_llm = ChatOllama(model="llama3.2:latest")
nvidia_llm = ChatNVIDIA(
    model="meta/llama-3.1-70b-instruct", nvidia_api_key=model_lib.nvidia_api_key
)

usellm = openai_llm

task_prompt = "Find a one-way flight from Toronto to Shanghai on 31 June 2025 on Google Flights. Return me the cheapest option."


async def main():
    agent = Agent(
        task=task_prompt,
        llm=usellm,
    )
    history = await agent.run()

    # Access (some) useful information
    history.urls()  # List of visited URLs
    history.screenshots()  # List of screenshot paths
    history.action_names()  # Names of executed actions
    history.extracted_content()  # Content extracted during execution
    history.errors()  # Any errors that occurred
    history.model_actions()  # All actions with their parameters
    print(history)


async def main_2():
    # Initialize the controller
    controller = Controller()

    @controller.action("Ask user for information")
    def ask_human(question: str, display_question: bool) -> str:
        return input(f"\n{question}\nInput: ")

    agent = Agent(
        task=task_prompt,
        llm=usellm,
        controller=controller,  # Custom function registry
        use_vision=True,  # Enable vision capabilities
        save_conversation_path="logs/conversation.json",  # Save chat logs
    )
    history = await agent.run()

    # Access (some) useful information
    history.urls()  # List of visited URLs
    history.screenshots()  # List of screenshot paths
    history.action_names()  # Names of executed actions
    history.extracted_content()  # Content extracted during execution
    history.errors()  # Any errors that occurred
    history.model_actions()  # All actions with their parameters
    print(history)


asyncio.run(main())
