# from dotenv import load_dotenv
import ckan.plugins.toolkit as toolkit

from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get("ckanext.chat.completion_url"),
    api_version="2024-05-01-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment")
model = OpenAIModel(deployment, openai_client=client)
agent = Agent(model)


async def agent_response(messages):
    """
    Synchronously run the agent on the provided list of messages.
    The messages should be a list of dicts with keys 'role' and 'content'.
    """
    result = await agent.run(messages)
    return result
