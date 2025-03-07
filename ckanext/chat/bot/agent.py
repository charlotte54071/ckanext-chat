# from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Optional, Any

import nest_asyncio
import ckan.plugins.toolkit as toolkit
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, computed_field, create_model

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.exceptions import UsageLimitExceeded, ModelRetry, UnexpectedModelBehavior, AgentRunError, ModelHTTPError, FallbackExceptionGroup

# Import CKAN models for datasets and resources
from ckan.model.package import Package
from ckan.model.resource import Resource
from ckan.model import Session  # Import the CKAN SQLAlchemy session

nest_asyncio.apply()

log = __import__("logging").getLogger(__name__)

_actions = []

client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get("ckanext.chat.completion_url"),
    api_version="2024-05-01-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment")
model = OpenAIModel(deployment, openai_client=client)

# Global flag to ensure dynamic models are initialized once.
dynamic_models_initialized = False

def init_dynamic_models():
    """
    Initialization function to 'warm up' the dynamic models.
    For now, it simply loads the CKAN URL patterns.
    (Extend this to fetch a sample Package/Resource if needed.)
    """
    global dynamic_models_initialized
    if not dynamic_models_initialized:
        # Initialize routes by calling get_ckan_url_patterns.
        get_ckan_url_patterns()
        # Optionally fetch a sample package to warm up.
        try:
            package_list = toolkit.get_action("package_list")({}, {})
            if package_list:
                sample_pkg = toolkit.get_action("package_show")({}, {"id": package_list[0]})
                # Convert the sample package dict to a DynamicDataset.
                _ = DynamicDataset(**sample_pkg)
        except Exception as e:
            log.warning(f"Could not initialize sample dynamic models: {e}")
        dynamic_models_initialized = True

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
    "- **Utilize URL Patterns:** Leverage the URL patterns retrieved from `get_ckan_url_patterns()` to replace all names of CKAN objects in your response by links to the appropriate CKAN View. For example, use the dataset.read pattern for dataset names.\n\n"
    "- **Use Of Actions:** Run actions like search and get right away without user confirmation. When suggesting actions or operations that manipulate data, ask for confirmation and reference the corresponding CKAN action functions. This not only clarifies the steps involved but also guides users on how to perform these actions programmatically.\n\n"
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

def agent_response(prompt, history: str):
    msg_history = convert_to_model_messages(history)
    return agent.run_sync(user_prompt=prompt, message_history=msg_history)
# --- CKAN Routing and URL Helpers ---

VARIABLE_REGEX = re.compile(r"<(?:(?P<converter>[^:<>]+):)?(?P<variable>[^<>]+)>")

def extract_variables(rule: str) -> List[Dict[str, Optional[str]]]:
    return [match.groupdict() for match in VARIABLE_REGEX.finditer(rule)]

class RouteModel(BaseModel):
    endpoint: str
    rule: str
    methods: List[str]

    @computed_field
    @property
    def variables(self) -> List[Dict[str, Optional[str]]]:
        return extract_variables(self.rule)
    
    @computed_field
    @property
    def full_url_pattern(self) -> str:
        def repl(match):
            var = match.group("variable")
            return f"{{{var}}}"
        return VARIABLE_REGEX.sub(repl, self.rule)

    def build_url(self, base_url: str = toolkit.config.get('ckan.site_url', ""), fill: Optional[Dict[str, Any]] = None) -> str:
        fill = fill or {}
        substitution = {}
        for var in self.variables:
            var_name = var["variable"]
            substitution[var_name] = str(fill.get(var_name, f"<{var_name}>"))
        pattern = self.full_url_pattern
        if base_url.endswith("/") and pattern.startswith("/"):
            base_url = base_url[:-1]
        try:
            url_path = pattern.format(**substitution)
        except KeyError as e:
            raise ValueError(f"Missing substitution for variable: {e.args[0]}")
        return f"{base_url}{url_path}"

routes: Dict[str, RouteModel] = {}

@agent.tool_plain
def get_ckan_url_patterns() -> List[RouteModel]:
    global routes
    if not routes:
        from ckanext.chat.views import global_ckan_app
        for rule in global_ckan_app.url_map.iter_rules():
            if not rule.rule.startswith("/_debug_toolbar"):
                route = RouteModel(
                    endpoint=rule.endpoint,
                    rule=rule.rule,
                    methods=sorted(list(rule.methods))
                )
                routes[rule.endpoint] = route
    return list(routes.values())

def find_route_by_endpoint(endpoint: str) -> Optional[RouteModel]:
    for route in routes:
        if route.endpoint == endpoint:
            return route
    return None

@agent.tool_plain  
def get_ckan_actions() -> List[str]:
    from ckan.logic import _actions
    return [key for key in _actions.keys()]

@agent.tool_plain
def get_function_info(action_key: str) -> dict:
    from ckan.logic.action.get import help_show
    func_model = FuncSignature(doc=help_show({}, {"name": action_key}))
    return func_model.model_dump()

@agent.tool_plain
def run_action(action_name: str, parameters: Dict) -> Any:
    """
    Executes a CKAN action and converts returned entities (in JSON format)
    into dynamic models when possible.
    """
    context = {
        "user": toolkit.current_user,
        #"auth_user_obj": getattr(toolkit, "c", {}).get("userobj", None),
        #"session": Session,
        "ignore_auth": False,
    }
    try:
        response = toolkit.get_action(action_name)(context, parameters)
    except Exception as e:
        return {"error": str(e)}
    
    def convert_entity(entity: Any) -> Any:
        # If the entity is a dict, try converting to a dynamic model.
        if isinstance(entity, dict):
            # If it's likely a dataset (has a 'resources' key), convert accordingly.
            if "resources" in entity:
                try:
                    return DynamicDataset(**entity).dict()
                except Exception as ex:
                    log.warning(f"Conversion to DynamicDataset failed: {ex}")
                    return entity
            # If it looks like a resource (has 'package_id' or a 'url'), convert.
            elif "package_id" in entity or "url" in entity:
                try:
                    return DynamicResource(**entity).dict()
                except Exception as ex:
                    log.warning(f"Conversion to DynamicResource failed: {ex}")
                    return entity
        return entity

    if isinstance(response, list):
        return [convert_entity(item) for item in response]
    elif isinstance(response, dict):
        return convert_entity(response)
    return response

class FuncSignature(BaseModel):
    doc: Any

# --- Dynamic Models for Datasets and Resources ---

class DynamicDataset(BaseModel):
    id: str  # CKAN dataset id
    class Config:
        extra = "allow"

    @computed_field
    @property
    def view_url(self) -> str:
        route = find_route_by_endpoint("dataset.read")
        if route:
            return route.build_url(fill={"id": self.id})
        return ""

    @classmethod
    def from_ckan(cls, package: Package) -> "DynamicDataset":
        data = package.as_dict() if hasattr(package, "as_dict") else package.__dict__
        return cls(**data)

class DynamicResource(BaseModel):
    id: str  # CKAN resource id
    package_id: Optional[str] = None
    class Config:
        extra = "allow"

    @computed_field
    @property
    def view_url(self) -> str:
        route = find_route_by_endpoint("resource.read")
        if route:
            return route.build_url(fill={"id": self.id})
        return ""

    @classmethod
    def from_ckan(cls, resource: Resource) -> "DynamicResource":
        data = resource.as_dict() if hasattr(resource, "as_dict") else resource.__dict__
        return cls(**data)

# --- (Optional) Dynamic Model Generation from Schema ---
def generate_dynamic_model(model_name: str, schema: dict) -> BaseModel:
    fields = {key: (str, None) for key in schema.keys()}
    return create_model(model_name, **fields)  # type: ignore

# Example usage:
# dataset_schema = {"id": None, "title": None, "notes": None, "extras": None}
# DynamicDatasetModel = generate_dynamic_model("DynamicDatasetModel", dataset_schema)
# instance = DynamicDatasetModel(id="abc123", title="My Dataset")
# print(instance)
#
# resource_schema = {"id": None, "name": None, "url": None, "extras": None}
# DynamicResourceModel = generate_dynamic_model("DynamicResourceModel", resource_schema)
# instance = DynamicResourceModel(id="res456", name="Resource Name")
# print(instance)

# Now, the agent initializes the dynamic models on the first query and run_action
# converts returned JSON entities to dynamic models when available!
