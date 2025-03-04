# from dotenv import load_dotenv
import json
from typing import List

import ckan.plugins.toolkit as toolkit
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIModel
import nest_asyncio
import ckan.plugins.toolkit as toolkit
import inspect

nest_asyncio.apply()

log = __import__("logging").getLogger(__name__)

_actions=[]

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


def agent_response(prompt, history: str):
    """
    Synchronously run the agent on the provided list of messages.
    The messages should be a list of dicts with keys 'role' and 'content'.
    """
    msg_history = convert_to_model_messages(history)
    result = agent.run_sync(user_prompt=prompt, message_history=msg_history)
    return result

from typing import Dict, Callable
from ckan.logic import _actions
from ckan.types.logic import ActionResult
from ckan.logic.action.get import help_show
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema
from ckan.model import User

class FuncSignature(BaseModel):
    doc: ActionResult.HelpShow

@agent.tool_plain  
def get_ckan_actions()->List[str]:
    from ckan.logic import _actions
    return [key for key in _actions.keys()]
@agent.tool_plain
def get_function_info(action_key: str) -> dict:
    func_model = FuncSignature(doc=help_show({},{"name":action_key}))
    return func_model.model_dump()
@agent.tool_plain
def run_action(action_name: str, parameters: Dict)-> dict:
    context={"user": toolkit.current_user}
    try:
        response = toolkit.get_action(action_name)(context, parameters)
    except Exception as e:
        return e
    return response