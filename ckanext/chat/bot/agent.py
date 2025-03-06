# from dotenv import load_dotenv
import json
from typing import List, Dict

import ckan.plugins.toolkit as toolkit
from openai import AsyncAzureOpenAI
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIModel
import nest_asyncio
import ckan.plugins.toolkit as toolkit


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

# Define the system prompt for the CKAN agent
system_prompt = (
    "As a CKAN Agent, you have access to specialized tools that allow you to interact with the CKAN platform effectively. "
    "Your primary functions include retrieving URL patterns, accessing available CKAN actions, obtaining detailed information about specific functions, "
    "and executing CKAN actions with specified parameters. Here's a detailed overview of your capabilities:\n\n"

    "**Available Tools:**\n\n"

    "1. **Retrieve CKAN URL Patterns:**\n"
    "- **Function:** `get_ckan_url_patterns() -> List[RouteModel]`\n"
    "- **Description:** Fetches a list of URL patterns defined in the CKAN application. Each pattern includes details such as the endpoint, URL rule, and allowed HTTP methods. "
    "This information is crucial for understanding the available routes within the CKAN application and can be used to generate useful links.\n"
    "- **Usage:** Call this function to obtain the current URL mappings in CKAN.\n\n"

    "2. **List CKAN Actions:**\n"
    "- **Function:** `get_ckan_actions() -> List[str]`\n"
    "- **Description:** Retrieves a list of all available CKAN action functions. These actions represent the core operations that can be performed within CKAN, such as creating datasets, updating resources, or managing users.\n"
    "- **Usage:** Use this function to get an overview of the actions you can perform within CKAN.\n\n"

    "3. **Get Function Information:**\n"
    "- **Function:** `get_function_info(action_key: str) -> dict`\n"
    "- **Description:** Provides detailed information about a specific CKAN action function, including its signature and documentation. This helps in understanding the required parameters and the functionality of the action.\n"
    "- **Usage:** Invoke this function with the name of the action to get detailed information about it.\n\n"

    "4. **Execute CKAN Action:**\n"
    "- **Function:** `run_action(action_name: str, parameters: Dict) -> dict`\n"
    "- **Description:** Executes a specified CKAN action with the given parameters. This allows for dynamic interaction with the CKAN backend to perform operations like creating or updating datasets, querying data, or managing organizational details.\n"
    "- **Usage:** Use this function to perform specific actions within CKAN by providing the action name and necessary parameters.\n\n"

    "**Guidelines for Enhancing Outputs with Useful Links:**\n\n"

    "- **Utilize URL Patterns:** Leverage the URL patterns retrieved from `get_ckan_url_patterns()` to replace all names of CKAN objects in ur response by links to the appropiate CKAN View. For example use dataset.read pattern for dataset names.\n\n"

    "- **Use Of Actions:** Run Actions like search and get right away without user confirmation. When suggesting actions or operations that minipulate data, ask for confirmation and reference the corresponding CKAN action functions. This not only clarifies the steps involved but also guides users on how to perform these actions programmatically.\n\n"

    "- **Incorporate Documentation Links:** For complex operations or lesser-known features, include links to the official CKAN documentation or user guides. This offers users additional resources to understand and execute the desired tasks effectively.\n\n"

    "By following these guidelines and utilizing your tools effectively, you can provide comprehensive and actionable responses that are enriched with direct links and references, enhancing the overall user experience within the CKAN platform."
)

agent = Agent(
    model=model,
    system_prompt=system_prompt,
)


def convert_to_model_messages(history: str) -> List:
    model_messages = None
    if history:
        history_list = json.loads(history)
        model_messages = ModelMessagesTypeAdapter.validate_python(history_list)
    return model_messages

from pydantic_ai.exceptions import UsageLimitExceeded, ModelRetry, UnexpectedModelBehavior,AgentRunError,ModelHTTPError,FallbackExceptionGroup

def agent_response(prompt, history: str):
    """
    Synchronously run the agent on the provided list of messages.
    The messages should be a list of dicts with keys 'role' and 'content'.
    """
    msg_history = convert_to_model_messages(history)
    try:
        msg_history = convert_to_model_messages(history)
        result = agent.run_sync(user_prompt=prompt, message_history=msg_history)
        return result
    except Exception as e:
        return e

from ckan.types.logic import ActionResult
from ckan.logic.action.get import help_show
from pydantic import BaseModel, computed_field
from typing import List, Optional, Dict, Any

import re

# Regular expression to capture Flask URL variables, optionally with a converter.
VARIABLE_REGEX = re.compile(r"<(?:(?P<converter>[^:<>]+):)?(?P<variable>[^<>]+)>")

def extract_variables(rule: str) -> List[Dict[str, Optional[str]]]:
    """
    Extracts variables from a Flask URL rule.
    Returns a list of dictionaries with keys 'variable' and 'converter' (converter may be None).
    """
    return [match.groupdict() for match in VARIABLE_REGEX.finditer(rule)]

# Define a Pydantic model for your route
class RouteModel(BaseModel):
    endpoint: str
    rule: str
    methods: List[str]

    @computed_field
    @property
    def variables(self) -> List[Dict[str, Optional[str]]]:
        """
        Returns a list of variables in the URL rule, along with any converter specified.
        Example: For rule '/<path:filename>', returns [{'converter': 'path', 'variable': 'filename'}]
        """
        return extract_variables(self.rule)
    
    @computed_field
    @property
    def full_url_pattern(self) -> str:
        """
        Converts the Flask URL rule to a Python format string.
        For example, '/<path:filename>' becomes '/{filename}'.
        """
        def repl(match):
            var = match.group("variable")
            return f"{{{var}}}"
        return VARIABLE_REGEX.sub(repl, self.rule)

    def build_url(self, base_url: str=toolkit.config.get('ckan.site_url',""), fill: Optional[Dict[str, Any]] = None) -> str:
        """
        Builds a fully qualified URL.
        
        Args:
          base_url: The scheme and domain, e.g. "http://example.com".
          fill: Optional dict mapping variable names to values. Missing values will be replaced
                with default dummy values based on their converter.
                
        Returns:
          A fully qualified URL with all parameters substituted.
        """
        fill = fill or {}
        # Prepare substitutions: for each variable in the rule, use the provided value or a dummy value.
        substitution = {}
        for var in self.variables:
            var_name = var["variable"]
            converter = var.get("converter")
            substitution[var_name] = str(fill.get(var_name))
        # Use Python's format method on the full_url_pattern.
        pattern = self.full_url_pattern
        # Ensure base_url does not end with a trailing slash if pattern starts with one.
        if base_url.endswith("/") and pattern.startswith("/"):
            base_url = base_url[:-1]
        try:
            url_path = pattern.format(**substitution)
        except KeyError as e:
            raise ValueError(f"Missing substitution for variable: {e.args[0]}")
        return f"{base_url}{url_path}"

class FuncSignature(BaseModel):
    doc: ActionResult.HelpShow

routes = {}


@agent.tool_plain
def get_ckan_url_patterns()-> List[RouteModel]:
        global routes
        if not routes:
            from ckanext.chat.views import global_ckan_app
            for rule in global_ckan_app.url_map.iter_rules():
                if not rule.rule.startswith("/_debug_toolbar"):
                    route=RouteModel(
                        endpoint=rule.endpoint,
                        rule=rule.rule,
                        methods=sorted(list(rule.methods))
                    )
                    routes[rule.endpoint]=(route.json())
        return routes

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

    