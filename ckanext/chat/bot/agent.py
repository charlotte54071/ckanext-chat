# from dotenv import load_dotenv
import json
from typing import List

import ckan.plugins.toolkit as toolkit
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIModel

log = __import__("logging").getLogger(__name__)


client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get("ckanext.chat.completion_url"),
    api_version="2024-05-01-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment")
model = OpenAIModel(deployment, openai_client=client)
agent = Agent(model)


def convert_to_model_messages(history: str) -> List:
    model_messages = None
    if history:
        history_list = json.loads(history)
        model_messages = ModelMessagesTypeAdapter.validate_python(history_list)
    return model_messages


async def agent_response(prompt, history: str):
    """
    Synchronously run the agent on the provided list of messages.
    The messages should be a list of dicts with keys 'role' and 'content'.
    """
    msg_history = convert_to_model_messages(history)
    result = await agent.run(user_prompt=prompt, message_history=msg_history)
    return result
