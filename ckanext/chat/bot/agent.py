# from dotenv import load_dotenv
import ckan.plugins.toolkit as toolkit

from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessagesTypeAdapter
from typing import List
from typing import List
import json
from datetime import datetime

log = __import__("logging").getLogger(__name__)


client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get("ckanext.chat.completion_url"),
    api_version="2024-05-01-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment")
model = OpenAIModel(deployment, openai_client=client)
agent = Agent(model)


def convert_timestamp(rfc2822_str):
    try:
        # Parse the RFC 2822 formatted string
        dt = datetime.strptime(rfc2822_str, "%a, %d %b %Y %H:%M:%S %Z")
        # Return the ISO 8601 formatted string
        return dt.isoformat()
    except ValueError as e:
        print(f"Error parsing date: {rfc2822_str} - {e}")
        return rfc2822_str  # Return the original string if parsing fails


def update_timestamps(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "timestamp" and isinstance(value, str):
                data[key] = convert_timestamp(value)
            else:
                update_timestamps(value)
    elif isinstance(data, list):
        for item in data:
            update_timestamps(item)


def convert_to_model_messages(history: str) -> List:
    model_messages = None
    if history:
        history_list = json.loads(history)
        # reformat dates
        update_timestamps(history_list)
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
