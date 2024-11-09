
import app_consts as appconsts
import ollama as ollama_lib
from openai import OpenAI
from openai import AzureOpenAI
import pprint

#References:
#Ollama: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
#LM-studio: https://lmstudio.ai/docs/basics/server
#OpenAI: https://platform.openai.com/docs/api-reference/audio/createTranslation
#

class LLMSService:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMSService, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def __init__(self):
        self.ollama_client = ollama_lib.Client(host=f'{appconsts.ollama_host_url}:{appconsts.ollama_host_port}')
        self.openai_client =  OpenAI(api_key=appconsts.openai_apikey)
        self.azureopenai_client = AzureOpenAI(
                                    api_key=appconsts.azure_openai_apikey,  
                                    api_version=appconsts.azure_openai_api_version,
                                    azure_endpoint = appconsts.azure_openai_endpoint
                                    )
        self.llm_clients={appconsts.ai_clients[0]:self.ollama_client, 
                          appconsts.ai_clients[1]:self.openai_client,
                          appconsts.ai_clients[2]:self.azureopenai_client,
                          }

    # convert differnt response(res) into uniform one that be easy handled.
    def generate_response(self, clientname, model, res):
        if clientname == appconsts.ai_clients[0]:
            new_res = {
                    "client": clientname,
                    "model": model,
                    "messages":[],
                    "details":"",
                    "status_code":200,
                    "response":res
                    }
        elif clientname == appconsts.ai_clients[1]:
                pprint.pprint(res)
                new_res = {
                    "client": clientname,
                    "model": model,
                    "messages":[res.choices[0].message.content],
                    "details":"",
                    "status_code":200,
                    "response":{}
                    }
        elif clientname == appconsts.ai_clients[2]:
                pprint.pprint(res)
                new_res = {
                    "client": clientname,
                    "model": 'azure',
                    "messages":[res.choices[0].message.content],
                    "details":"",
                    "status_code":200,
                    "response":{}
                    }
        return new_res
    

    def chat(self, client, model, messages,temperature=0.2):
        if client not in appconsts.ai_clients:
            return {"messsages":[], "details": f"there is no {client}", "status_code":401}
        if client == appconsts.ai_clients[0]:
            res = self.ollama_client.chat(model=model, messages=messages)
            return self.generate_response(client, model, res)
        elif client == appconsts.ai_clients[1]:
            return self.chatcompletions_v1(client, model, messages, temperature)
        elif client == appconsts.ai_clients[2]:
            return self.chatcompletions_v1(client, model, messages, temperature)
        
    def generate(self, client, model, prompt, isstream, messages):
        if client not in appconsts.ai_clients:
            return {"messsages":[], "details": f"there is no {client}", "status_code":401}
        if client == appconsts.ai_clients[0]:
            res = self.ollama_client.generate(model=model, prompt=messages.get("prompt", prompt), stream=isstream,
                                              format=messages.get("format", ""),
                                              suffix=messages.get("suffix", ""),
                                              system=messages.get("system", ""),
                                              context=messages.get("context", None)
                                              # add more
                                              )
            return self.generate_response(client, model, res)
        # only like OpenAI
    def chatcompletions_v1(self, client, model, messages, temperature):
        if client not in appconsts.ai_clients or client == appconsts.ai_clients[0]:
            return {"messsages":[], "details": f"there is no {client}", "status_code":401}
        if client == appconsts.ai_clients[1]:
            completion = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                )
            return self.generate_response(client, model, completion)
        else:
            completion = self.azureopenai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                )
            return self.generate_response(client, model, completion)

