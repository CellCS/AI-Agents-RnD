from dotenv import load_dotenv
from pathlib import Path
import os

home_dir = os.path.expanduser("~")
dotenv_path = Path(home_dir) / "Keys" / "OpenAIService" / ".env"
load_dotenv(dotenv_path=dotenv_path)

ai_clients=["ollama","openai","azure", "lmstudio"]

openai_key=os.getenv("OPENAI_API_KEY", "")

# LLMs
ollama_host_url = os.getenv('OLLAMA_HOST_URL', 'http://localhost')
ollama_host_port=int(os.getenv('OLLAMA_HOST_PORT', '11434'))
ollama_llm_models=os.getenv('OLLAMA_LLM_MODELS', '').split(',')
ollama_embed_models=os.getenv('OLLAMA_EMBEDDING_MODELS', '').split(',')
ollama_embed_chunksize=int(os.getenv('OLLAMA_EMBEDDING_CHUNKSIZES', '1000'))
# LM Studio
lmstudio_host_url = os.getenv('LMSTUDIO_HOST_URL', 'http://localhost:1234/v1')
lmstudio_apikey=os.getenv('LMSTUDIO_APIKEY', 'lm-studio')
lmstudio_llm_models=os.getenv('LMSTUDIO_LLM_MODELS', '').split(',')
lmstudio_embed_models=os.getenv('LMSTUDIO_EMBEDDING_MODELS', '').split(',')
lmstudio_embed_chunksize=int(os.getenv('LMSTUDIO_EMBEDDING_CHUNKSIZES', '1000'))
# Azure OpenAI Service
azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '')
azure_openai_api_type = os.getenv('AZURE_OPENAI_API_TYPE', 'azure')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
azure_openai_apikey = os.getenv('AZURE_OPENAI_API_KEY', '')
azure_openai_developmentname = os.getenv('AZURE_OPENAI_DELOYMENT_NAME', '')
azure_openai_llm_models=os.getenv('AZURE_OPENAI_LLM_MODELS', '').split(',')
azure_openai_llm_modelversions=os.getenv('AZURE_OPENAI_LLM_MODELS_VERSIONS', '').split(',')
# OpenAI
openai_apikey = os.getenv('OPENAI_API_KEY', '')
openai_llm_models=os.getenv('OPENAI_LLM_MODELS', '').split(',')


### Magentic-One
azure_chatcompletionclient = {
  "api_version": "2024-02-15-preview",
  "azure_endpoint": azure_openai_endpoint,
  "model_capabilities": {
    "function_calling": True,
    "json_output": True,
    "vision": True
  },
  "azure_ad_token_provider": "DEFAULT",
  "model": "gpt-4o-2024-05-13"
}