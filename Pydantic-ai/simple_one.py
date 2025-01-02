from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncAzureOpenAI
from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel

import pprint

from dotenv import load_dotenv
from pathlib import Path
import os

home_dir = os.path.expanduser("~")
dotenv_path = Path(home_dir) / "Keys" / "OpenAIService" / ".env"
load_dotenv(dotenv_path=dotenv_path)

model_openai = OpenAIModel("gpt-4o", api_key=os.getenv("OPENAI_API_KEY", ""))

client_azure = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
)
model_openaiAzure = OpenAIModel("gpt-4o", openai_client=client_azure)
model_ollama = "ollama:llama3.2"
model_ollama2 = OllamaModel(
    model_name="llama3.2",
    base_url="http://localhost:11434/v1",
)

# Define a very simple agent including the model to use, you can also set the model when running the agent.

modellist = [model_openai, model_openaiAzure, model_ollama, model_ollama2]

for modelitem in modellist:
    agent = Agent(
        modelitem,
        # Register a static system prompt using a keyword argument to the agent.
        # For more complex dynamically-generated system prompts, see the example below.
        system_prompt="Be concise, reply with one sentence.",
    )

    # Run the agent synchronously, conducting a conversation with the LLM.
    # Here the exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM,
    # the model will return a text response. See below for a more complex run.

    pprint.pprint(f"======begin=============")
    result = agent.run_sync('Where does "hello world" come from?')
    # pprint.pprint(result)
    pprint.pprint(f"===========result.usage===========")
    pprint.pprint(result.usage())
    pprint.pprint(f"============result.data===========")
    pprint.pprint(result.data)
    pprint.pprint(f"============ENd===========")
