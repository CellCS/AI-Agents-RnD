# https://github.com/ScrapeGraphAI/Scrapegraph-ai
import json
from scrapegraphai.graphs import SmartScraperGraph
import os
import nest_asyncio

nest_asyncio.apply()

# export TOKENIZERS_PARALLELISM=false
openai_llm = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": "openai/gpt-4o-mini",
}
ollma_llm = {
    "model": "ollama/llama3.1",
    "temperature": 0,
    "format": "json",  # Ollama requires explicitly specifying the format
    "base_url": "http://localhost:11434",  # Set the Ollama URL by default http://localhost:11434/ap
    "model_tokens": 4096,
}
embedding_model = {
    "model": "ollama/nomic-embed-text",
    "base_url": "http://localhost:11434",  # Set the Ollama URL
}
# Define the configuration for the scraping pipeline
graph_config = {
    "llm": ollma_llm,
    "embeddings": embedding_model,
    "verbose": True,
    "headless": False,
}

# Create the SmartScraperGraph instance
urls = ["https://platform.openai.com/docs/guides/text-to-speech"]
smart_scraper_graph = SmartScraperGraph(
    prompt="Extract all Supported output formats from the website",
    source=urls[0],
    config=graph_config,
)

result = smart_scraper_graph.run()
print("==========================")
print(json.dumps(result, indent=4))
