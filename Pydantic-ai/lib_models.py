from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncAzureOpenAI
from pydantic_ai.models.ollama import OllamaModel
from pydantic import BaseModel

from dotenv import load_dotenv
from pathlib import Path
import os

home_dir = os.path.expanduser("~")
dotenv_path = Path(home_dir) / "Keys" / "OpenAIService" / ".env"
load_dotenv(dotenv_path=dotenv_path)
client_azure = AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
)


model_openai = OpenAIModel("gpt-4o", api_key=os.getenv("OPENAI_API_KEY", ""))
model_openaiAzure = OpenAIModel("gpt-4o", openai_client=client_azure)
model_ollama = OllamaModel(
    model_name="llama3.2",
    base_url="http://localhost:11434/v1",
)


def get_model_openai(modelname: str):
    return OpenAIModel(modelname, api_key=os.getenv("OPENAI_API_KEY", ""))


def get_model_azopenai(modelname: str):
    return OpenAIModel(modelname, openai_client=client_azure)


def get_model_ollama(modelname: str, baseurl: str = "http://localhost:11434/v1"):
    return OllamaModel(
        model_name=modelname,
        base_url=baseurl,
    )
