# from dotenv import load_dotenv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ckan.model as CKANmodel
import ckan.plugins.toolkit as toolkit
import nest_asyncio
import tiktoken
from ckan.lib.lazyjson import LazyJSONObject

# Import CKAN models for datasets and resources
from ckan.model.package import Package
from ckan.model.resource import Resource
from openai import AsyncAzureOpenAI
from pydantic import (
    BaseModel,
    HttpUrl,
    ValidationError,
    computed_field,
    create_model,
    root_validator,
)
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import (
    AgentRunError,
    FallbackExceptionGroup,
    ModelHTTPError,
    ModelRetry,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from pymilvus import MilvusClient


def truncate_output_by_token(
    output: str, token_limit: int, encoding_name="cl100k_base"
) -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(output)
    if len(tokens) > token_limit:
        truncated_tokens = tokens[:token_limit]
        output = encoding.decode(truncated_tokens)
    return output


def truncate_value(value, max_length):
    if isinstance(value, str):
        return value[:max_length] + "..." if len(value) > max_length else value
    elif isinstance(value, list):
        truncated_list = [truncate_value(item, max_length) for item in value]
        return (
            truncated_list[:max_length] + ["..."]
            if len(truncated_list) > max_length
            else truncated_list
        )
    return value


def truncate_by_depth(data, max_depth, current_depth=0, placeholder="..."):
    """
    Recursively traverse the JSON-like structure (dicts and lists)
    and replace content beyond max_depth with a placeholder.
    """
    if current_depth >= max_depth:
        return placeholder

    if isinstance(data, dict):
        return {
            key: truncate_by_depth(
                truncate_value(value, max_length=200),
                max_depth,
                current_depth + 1,
                placeholder,
            )
            for key, value in data.items()
        }

    if isinstance(data, list):
        data = truncate_value(data, max_length=200)
        return [
            truncate_by_depth(item, max_depth, current_depth + 1, placeholder)
            for item in data
        ]

    # For any other type (str, int, float, etc.), return the value directly.
    return data


def clean_json(data):
    if isinstance(data, dict):
        return {
            k: clean_json(v)
            for k, v in data.items()
            if clean_json(v) not in ([], {}, "", None)
        }
    elif isinstance(data, list):
        return [
            clean_json(item)
            for item in data
            if clean_json(item) not in ([], {}, "", None)
        ]
    else:
        return data


nest_asyncio.apply()

log = __import__("logging").getLogger(__name__)

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
vector_dim = None
if milvus_url:
    milvus_client = MilvusClient(uri=milvus_url)
    if milvus_client:
        # Get the collection info (schema, etc.)
        collection_info = milvus_client.describe_collection(
            collection_name=collection_name
        )

        # Attempt to find the first field that has a 'dim' parameter
        vector_field = None
        for entry in collection_info["fields"]:
            # Check if this field is a vector field by verifying if it has a 'dim' parameter
            if "params" in entry.keys() and "dim" in entry["params"].keys():
                vector_field = entry
                break
        if vector_field:
            vector_dim = vector_field["params"]["dim"]
            field_name = vector_field["name"]
            log.debug(f"Found vector field: {field_name}")
            log.debug(f"Vector dimension is: {vector_dim}")
        else:
            vector_dim = None
            log.debug("No vector field found in the collection schema.")
    else:
        log.debug("Milvus client not initialized.")

else:
    milvus_client = None


@dataclass
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: AsyncAzureOpenAI = field(default_factory=lambda: azure_client)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim


nest_asyncio.apply()


_actions = []

azure_client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get(
        "ckanext.chat.completion_url", "https://your.chat.api"
    ),
    api_version="2024-02-15-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment", "gpt-4-vision-preview")
model = OpenAIModel(deployment, openai_client=azure_client)


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
                sample_pkg = toolkit.get_action("package_show")(
                    {}, {"id": package_list[0]}
                )
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
    "5. **Retrieve documents:**\n"
    "- **Function:** `rag_search(search_query: List[str]) -> List[RagHit]`\n"
    "- **Description:** Retrieves References to chunks in datasets doing vector search on a list of search strings. Each hit includes details on the dataset_id, the resource with the chunks and simularity metrics. "
    "- **Usage:** Call this function to obtain chunks of documents related to the users interest. Use the resource_show action on hits to enrich the information before presenting it to the user.\n\n"
    "**Guidelines for Enhancing Outputs with Useful Links:**\n\n"
    "- **Utilize URL Patterns:** Leverage the URL patterns retrieved from `get_ckan_url_patterns()` to replace all names of CKAN objects in your response by links to the appropriate CKAN View. For example, use the dataset.read pattern for dataset names.\n\n"
    "- **Use Of Actions:** Run actions like search and get right away without user confirmation. When suggesting actions or operations that manipulate data, ask for confirmation and reference the corresponding CKAN action functions. Allways prefer patch actions to update actions, because update deletes data not set in the action call.This not only clarifies the steps involved but also guides users on how to perform these actions programmatically.\n\n"
    "- **Search:** If the user is asking about datasets or resources use the package_search ation with argument include_private=True, to return all his datasets including private ones but returning only the ids of the datasets.\n\n"
    "- **Incorporate Documentation Links:** For complex operations or lesser-known features, include links to the official CKAN documentation or user guides. This offers users additional resources to understand and execute the desired tasks effectively.\n\n"
    "By following these guidelines and utilizing your tools effectively, you can provide comprehensive and actionable responses that are enriched with direct links and references, enhancing the overall user experience within the CKAN platform."
)


def generate_dynamic_model(model_name: str, schema: dict) -> BaseModel:
    fields = {key: (str, None) for key in schema.keys()}
    return create_model(model_name, **fields)  # type: ignore


agent = Agent(
    model=model,
    deps_type=Deps,
    system_prompt=system_prompt,
    retries=3,
)


def convert_to_model_messages(history: str) -> List:
    model_messages = None
    if history:
        history_list = json.loads(history)
        model_messages = ModelMessagesTypeAdapter.validate_python(history_list)
    return model_messages


def agent_response(prompt, history: str, deps=Deps):
    if not dynamic_models_initialized:
        init_dynamic_models()
    msg_history = convert_to_model_messages(history)
    return agent.run_sync(
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
        usage_limits=UsageLimits(total_tokens_limit=None, response_tokens_limit=None),
        #model_settings={'max_tokens': 400}
        )


# --- CKAN Routing and URL Helpers ---

VARIABLE_REGEX = re.compile(r"<(?:(?P<converter>[^:<>]+):)?(?P<variable>[^<>]+)>")


def extract_variables(rule: str) -> List[Dict[str, Optional[str]]]:
    return [match.groupdict() for match in VARIABLE_REGEX.finditer(rule)]


def repl(match):
    var = match.group("variable")
    return f"{{{var}}}"


class RouteModel(BaseModel):
    endpoint: str
    rule: str
    methods: List[str]
    variables: Optional[List] = []
    full_url_pattern: Optional[str]

    @root_validator(pre=True)
    def calculate_computed_field(cls, values):
        values["variables"] = extract_variables(values["rule"])
        values["full_url_pattern"] = VARIABLE_REGEX.sub(repl, values["rule"])
        return values

    def build_url(
        self,
        base_url: str = toolkit.config.get("ckan.site_url", ""),
        fill: Optional[Dict[str, Any]] = None,
    ) -> str:
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
            raise ValueError(f"Missing substitution for variable: {e.args[0]}") from e
        return f"{base_url}{url_path}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert the RouteModel to a dictionary for JSON serialization."""
        return {
            "endpoint": self.endpoint,
            "rule": self.rule,
            "methods": self.methods,
            "variables": self.variables,
            "full_url_pattern": self.full_url_pattern,
        }


routes: Dict[str, RouteModel] = {}


@agent.tool_plain
def get_ckan_url_patterns(help: bool = True, endpoint: str = "") -> RouteModel:
    """Get CKAN url paterns

    Args:
        help (bool, optional): if True the function returns a list of endpoint keys instead of a RouteModel, default is True
        endpoint (str, optional): when set to a valid endpoint key, function returns RouteModel instance. Defaults to "".

    Returns:
        RouteModel: Route Model with details about the url pattern and its arguments
    """
    global routes
    if not routes:
        from ckanext.chat.views import global_ckan_app

        for rule in global_ckan_app.url_map.iter_rules():
            if not rule.rule.startswith("/_debug_toolbar"):
                route = RouteModel(
                    endpoint=rule.endpoint,
                    rule=rule.rule,
                    methods=sorted(list(rule.methods)),
                )
                routes[rule.endpoint] = route
    if endpoint in routes.keys():
        return routes[endpoint].json()
    if help:
        return [str(key) for key in routes.keys()]


def find_route_by_endpoint(endpoint: str) -> Optional[RouteModel]:
    for key, route in routes.items():
        if route.endpoint == endpoint:
            # log.debug(f"route: {key}:{route}")
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


@agent.tool
def run_action(ctx: RunContext[Deps], action_name: str, parameters: Dict) -> Any:
    """
    Executes a CKAN action and converts returned entities (in JSON format)
    into dynamic models when possible.
    """
    user = CKANmodel.User.get(user_reference=ctx.deps.user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,  # SQLAlchemy session
        "ignore_auth": False,  # Do not bypass authorization checks
    }
    try:
        response = toolkit.get_action(action_name)(context, parameters)
    except Exception as e:
        return {"error": str(e)}

    def unpack_lazy_json(obj):
        if isinstance(obj, dict):
            return {key: unpack_lazy_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [unpack_lazy_json(item) for item in obj]
        elif isinstance(obj, LazyJSONObject):  # Replace with the actual class name
            return obj.encoded_json  # Unpacking the LazyJSONObject instance
        return obj

    def convert_entity(entity: Any) -> Any:
        # If the entity is a dict, try converting to a dynamic model.
        if isinstance(entity, dict):
            # If it's likely a dataset (has a 'resources' key), convert accordingly.
            entity = unpack_lazy_json(entity)  # Unpack LazyJSONObjects first

            if "resources" in entity:
                try:
                    log.debug("DynamicDataset")
                    dataset_dict = DynamicDataset(**entity).model_dump(
                        exclude_unset=True,
                        exclude_defaults=False,
                        exclude_none=True,
                    )
                    dataset_dict = {k: v for k, v in dataset_dict.items() if bool(v)}
                    log.debug(dataset_dict)
                    return dataset_dict
                except ValidationError as validation_error:
                    log.warning(
                        f"Validation error while converting to DynamicDataset: {validation_error.json()}"
                    )
                    return entity
                except Exception as ex:
                    log.warning(f"Conversion to DynamicDataset failed: {ex}")
                    return entity
            # If it looks like a resource (has 'package_id' or a 'url'), convert.
            elif "package_id" in entity or "url" in entity:
                try:
                    log.debug("DynamicResource")
                    resource_dict = DynamicResource(**entity).model_dump(
                        exclude_unset=True,
                        exclude_defaults=False,
                        exclude_none=True,
                    )
                    resource_dict = {k: v for k, v in resource_dict.items() if bool(v)}
                    log.debug(resource_dict)
                    return resource_dict
                except ValidationError as validation_error:
                    log.warning(
                        f"Validation error while converting to DynamicResource: {validation_error.json()}"
                    )
                    return entity
                except Exception as ex:
                    log.warning(f"Conversion to DynamicResource failed: {ex}")
                    return entity
            # For any other dict, iterate over all key-values and recursively convert them.
            new_entity = {}
            for key, value in entity.items():
                if bool(value):
                    if isinstance(value, list):
                        new_entity[key] = [
                            convert_entity(item) for item in value if bool(item)
                        ]
                    elif isinstance(value, dict):
                        new_entity[key] = convert_entity(value)
                    else:
                        new_entity[key] = value
            return new_entity

        return entity

    if action_name == "package_search":
        view_route = find_route_by_endpoint("dataset.read")
        clean_response = response
        clean_response["results"] = [
            {
                "id": result["id"],
                "name": result["name"],
                "view_url": str(view_route.build_url(fill={"id": result["id"]})),
            }
            for result in response["results"]
        ]
    else:
        clean_response = unpack_lazy_json(response)
        clean_response = clean_json(clean_response)
    log.debug("{} -> {}".format(len(str(response)), len(str(clean_response))))
    trunc_out = truncate_by_depth(clean_response, max_depth=5)
    # trunc_out=output
    limited_output = truncate_output_by_token(
        json.dumps(trunc_out, indent=None, separators=(",", ": ")),
        token_limit=400,#int(ctx.deps.max_context_length * 0.1),
    )
    log.debug("{} -> {}".format(len(str(response)), len(str(limited_output))))
    return limited_output


class FuncSignature(BaseModel):
    doc: Any


# --- Dynamic Models for Datasets and Resources ---
class DynamicDataset(BaseModel):
    id: str  # CKAN dataset id

    class Config:
        extra = "allow"

    view_url: Optional[str] = None

    @root_validator(pre=True)
    def calculate_computed_field(cls, values):
        route = find_route_by_endpoint("dataset.read")
        if route:
            values["view_url"] = str(route.build_url(fill={"id": values.get("id")}))
        resources = values.get("resources")
        if not isinstance(resources, list):
            raise ValueError(
                'Input should have a "resources" key with a list of resources.'
            )

        # Validate each resource entry
        validated_resources = []
        for resource in resources:
            validated_resource = DynamicResource(
                **resource
            )  # Validate and create instance
            validated_resources.append(validated_resource)

        values["resources"] = validated_resources
        return values

    @classmethod
    def from_ckan(cls, package: Package) -> "DynamicDataset":
        data = package.as_dict() if hasattr(package, "as_dict") else package.__dict__
        return cls(**data)

    # @classmethod
    # def model_dump(cls, *args, **kwargs):
    #     data = super().model_dump(cls,*args, **kwargs, exclude_none=True)
    #     # Filter out empty lists, dictionaries, and the string "null"
    #     return {k: v for k, v in data.items() if bool(v)}
    # # filtered_data = {k: v for k, v in data.items() if v not in ([], {}, "", '', "null")}


class DynamicResource(BaseModel):
    id: str  # CKAN resource id
    package_id: Optional[str] = None

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def calculate_computed_field(cls, values):
        route = find_route_by_endpoint("resource.read")
        if route:
            values["view_url"] = str(route.build_url(fill={"id": values.get("id")}))
        return values

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
        filtered_data = {
            k: v for k, v in data.items() if v not in ([], {}, "", "", "null")
        }
        return cls(**filtered_data)


def user_input_to_model_request(input=str) -> ModelRequest:
    user_prompt = UserPromptPart(content=input)
    return ModelRequest(
        parts=[user_prompt],
        kind="request",
    )


def exception_to_model_response(exc: Exception) -> ModelResponse:
    # For our known exceptions, we pass along the error text
    if isinstance(
        exc,
        (
            UsageLimitExceeded,
            ModelRetry,
            UnexpectedModelBehavior,
            AgentRunError,
            ModelHTTPError,
            FallbackExceptionGroup,
        ),
    ):
        error_text = str(exc)
    else:
        # For all other errors, create a custom message.
        error_text = f"An unexpected error occurred: {type(exc).__name__}: {exc}"

    # Create a TextPart that carries the error message.
    error_part = TextPart(content=error_text)

    # Construct a ModelResponse using the error part.
    return ModelResponse(
        parts=[error_part],
        model_name="pydanticai",  # Or your specific model identifier
        timestamp=datetime.now(timezone.utc),
        kind="response",
    )


class VectorMeta(BaseModel):
    id: int
    # _vector: List[float]
    # Auftraggeber: Optional[str] = None
    # Berichtart: Optional[str] = None
    # Foerdermittelgeber: Optional[str] = None
    # Foerdernummer: Optional[str] = None
    # Foerderprogramm: Optional[str] = None
    # Projektende: Optional[str] = None
    # Projektleiter: Optional[str] = None
    # Projektname: Optional[str] = None
    # Projektnummer: Optional[str] = None
    # Projektstart: Optional[str] = None
    # author: Optional[str] = None
    chunk_id: Optional[int] = None
    chunks: Optional[HttpUrl] = None
    dataset_id: Optional[str] = None
    dataset_url: Optional[HttpUrl] = None
    # description: Optional[str] = None
    groups: Optional[List[str]] = None
    # owner: Optional[str] = None
    private: Optional[str] = None
    resource_id: Optional[str] = None
    source: Optional[HttpUrl] = None
    # title: Optional[str] = None
    view_url: Optional[List[HttpUrl]] = None


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta


@agent.tool
async def rag_search(
    ctx: RunContext[Deps], search_query: List[str], limit: int = 3
) -> List[RagHit]:
    """Retrieve documentation sections based on a search query.

    Args:
        context: The call context.
        search_query: The search query, can be a list of query strngs.
        limit: The limit of hits to return.
    """
    emb_r = await ctx.deps.openai.embeddings.create(
        input=search_query,
        model="text-embedding-3-small",
        dimensions=ctx.deps.vector_dim,
    )
    query_vectors = [vec.embedding for vec in emb_r.data]
    search_res = ctx.deps.milvus_client.search(
        collection_name=ctx.deps.collection_name,
        data=query_vectors,
        search_params={
            "metric_type": "COSINE",
            "params": {"level": 1},  # Search parameters
        },
        limit=limit,  # Max. number of search results to return
        # output_fields=["*"], # Fields to return in the search results
        output_fields=list(VectorMeta.__fields__.keys()),
        consistency_level="Bounded",
    )
    if search_res:
        hits = []
        for i in range(len(query_vectors)):
            hits += [RagHit(**item) for item in search_res[i]]
        hits = [hit.json() for hit in hits]
        return hits
    else:
        return []
