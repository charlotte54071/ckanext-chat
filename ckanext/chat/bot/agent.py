import asyncio
import json
import re, os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import ckan.model as CKANmodel
import ckan.plugins.toolkit as toolkit
import nest_asyncio
import requests
import tiktoken
from ckan.lib.lazyjson import LazyJSONObject
from ckan.model.package import Package
from ckan.model.resource import Resource
from openai import AsyncAzureOpenAI
from pydantic import (
    BaseModel,
    HttpUrl,
    ValidationError,
    computed_field,
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

# Allow nested event loops.
nest_asyncio.apply()

# --------------------- Helper Functions ---------------------


def truncate_output_by_token(
    output: str, token_limit: int, skip_tokens: int = 0, encoding_name="cl100k_base"
) -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(output)
    
    if len(tokens) > token_limit:
        # Skip the specified number of tokens
        truncated_tokens = tokens[skip_tokens:skip_tokens + token_limit]
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
    return data


def unpack_lazy_json(obj):
    if isinstance(obj, dict):
        return {key: unpack_lazy_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack_lazy_json(item) for item in obj]
    elif isinstance(obj, LazyJSONObject):
        return obj.encoded_json
    return obj


def process_entity(data: Any) -> Any:
    if isinstance(data, dict):
        data = unpack_lazy_json(data)
        if "resources" in data:
            try:
                log.debug("DynamicDataset")
                dataset_dict = DynamicDataset(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                dataset_dict = {k: v for k, v in dataset_dict.items() if bool(v)}
                return process_entity(dataset_dict)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicDataset: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicDataset failed: {ex}")
        elif "package_id" in data or "url" in data:
            try:
                log.debug("DynamicResource")
                resource_dict = DynamicResource(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                resource_dict = {k: v for k, v in resource_dict.items() if bool(v)}
                return process_entity(resource_dict)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicResource: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicResource failed: {ex}")

        new_dict = {}
        for key, value in data.items():
            processed_value = process_entity(value)
            if processed_value not in ([], {}, "", None):
                new_dict[key] = processed_value
        return new_dict
    elif isinstance(data, list):
        new_list = []
        for item in data:
            processed_item = process_entity(item)
            if processed_item not in ([], {}, "", None):
                new_list.append(processed_item)
        return new_list
    else:
        return data


def download_file(url: str, headers: dict={}, verify: bool = True):
    try:
        response = requests.get(url, headers=headers, verify=verify, timeout=5)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return str(e)


# --------------------- Logging Setup ---------------------
log = __import__("logging").getLogger(__name__)

# --------------------- Milvus and CKAN Setup ---------------------

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
vector_dim = None
if milvus_url:
    milvus_client = MilvusClient(uri=milvus_url)
    if milvus_client:
        collection_info = milvus_client.describe_collection(
            collection_name=collection_name
        )
        vector_field = None
        for entry in collection_info["fields"]:
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

# --------------------- Azure & Agent Setup ---------------------

azure_client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get(
        "ckanext.chat.completion_url", "https://your.chat.api"
    ),
    api_version="2024-02-15-preview",
    api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment", "gpt-4-vision-preview")
model = OpenAIModel(deployment, openai_client=azure_client)


@dataclass
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: AsyncAzureOpenAI = field(default_factory=lambda: azure_client)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim


# --------------------- Dynamic Models Initialization ---------------------

dynamic_models_initialized = False


def init_dynamic_models():
    global dynamic_models_initialized
    if not dynamic_models_initialized:
        get_ckan_url_patterns()
        try:
            package_list = toolkit.get_action("package_list")({}, {})
            if package_list:
                sample_pkg = toolkit.get_action("package_show")(
                    {}, {"id": package_list[0]}
                )
                _ = DynamicDataset(**sample_pkg)
        except Exception as e:
            log.warning(f"Could not initialize sample dynamic models: {e}")
        dynamic_models_initialized = True


# --------------------- System Prompt & Agent ---------------------

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
    "- **Description:** Retrieves a list of all available CKAN action functions.\n"
    "- **Usage:** Use this function to get an overview of the actions you can perform within CKAN.\n\n"
    "3. **Get Function Information:**\n"
    "- **Function:** `get_function_info(action_key: str) -> dict`\n"
    "- **Description:** Provides detailed information about a specific CKAN action function, including its signature and documentation.\n"
    "- **Usage:** Invoke this function with the name of the action to get detailed information about it.\n\n"
    "4. **Execute CKAN Action:**\n"
    "- **Function:** `run_action(action_name: str, parameters: Dict) -> dict`\n"
    "- **Description:** Executes a specified CKAN action with the given parameters.\n"
    "- **Usage:** Use this function to perform specific actions within CKAN.\n\n"
    "5. **Download Resources Contents:**\n"
    "- **Function:** `get_resource_file_contents(resource_id: str, resource_url: str, max_token_length: int, skip_tokens: int=0, ssl_verify=True) -> str`\n"
    "- **Description:** Returns the file content string of a given resource.\n"
    "- **Usage:** Use this function retrieve file contents of CKAN or external content. Use skip_tokens and max_token_length to fetch a chunk of the content as needed.\n\n"
    "6. **Retrieve documents:**\n"
    "- **Function:** `rag_search(search_query: List[str]) -> List[RagHit]`\n"
    "- **Description:** Retrieves references to chunks in datasets doing vector search on a list of search strings.\n"
    "- **Usage:** Call this function to hits on document chunks related to the user’s interest.Use get_resource_file_contents tool to retrieve the contents of the documents related to the hits for in depth context.\n\n" \
    "  If the function returns an error that the milvus client is not set up, use the package_search action instead."
    "Follow these guidelines to incorporate useful links and documentation as needed."
)

agent = Agent(
    model=model,
    deps_type=Deps,
    system_prompt=system_prompt,
    retries=3,
)


def convert_to_model_messages(history: str) -> List:
    if history:
        history_list = json.loads(history)
        return ModelMessagesTypeAdapter.validate_python(history_list)
    return None


async def async_agent_response(prompt: str, history: str, deps: Deps) -> Any:
    if not dynamic_models_initialized:
        init_dynamic_models()
    msg_history = convert_to_model_messages(history)
    # Wrap the synchronous run call into a thread so that it can be awaited
    response = await asyncio.to_thread(
        agent.run_sync,
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
        usage_limits=UsageLimits(total_tokens_limit=None, response_tokens_limit=None),
    )
    return response


# --------------------- CKAN Routing and URL Helpers ---------------------

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
        substitution = {
            var["variable"]: str(fill.get(var["variable"], f"<{var['variable']}>"))
            for var in self.variables
        }
        pattern = self.full_url_pattern
        if base_url.endswith("/") and pattern.startswith("/"):
            base_url = base_url[:-1]
        try:
            url_path = pattern.format(**substitution)
        except KeyError as e:
            raise ValueError(f"Missing substitution for variable: {e.args[0]}") from e
        return f"{base_url}{url_path}"

    def to_dict(self) -> Dict[str, Any]:
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
    user = CKANmodel.User.get(user_reference=ctx.deps.user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    try:
        response = toolkit.get_action(action_name)(context, parameters)
    except Exception as e:
        return {"error": str(e)}
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
        clean_response = process_entity(clean_response)
    log.debug("{} -> {}".format(len(str(response)), len(str(clean_response))))
    return clean_response

@agent.tool_plain
def get_resource_file_contents(resource_id: str, resource_url: str, max_token_length: int, skip_tokens: int=0, ssl_verify=True) -> str:
    """Retrieves the content of a resource stored in filetore, allows setting max token of output and skip tokens to extract a chunk

    Args:
        resource_id (str): The UUID of the CKAN resource
        resource_url (str): The download url of the CKAN resource
        max_token_length (int): the maximum length of the string to return 
        skip_tokens (int): ommits the token length given from start of the contents, to retrieve a chunk
    Returns:
        str: the content of the file retrieved
    """
    ckan_url=toolkit.config.get("ckan.site_url")
    if ckan_url in resource_url:
        storage_path=toolkit.config.get('ckan.storage_path',"/var/lib/ckan/default")
        # Generate the folder structure based on the resource_id
        first_level_folder = resource_id[:3]
        second_level_folder = resource_id[3:6]
        file_name = resource_id[6:]

        # Construct the full file path
        file_path = os.path.join(storage_path, 'resources', first_level_folder, second_level_folder, file_name)
        log.debug(file_path)
        # Read and return the file contents
        try:
            with open(file_path, 'r') as file:
                contents = file.read()
            return truncate_output_by_token(contents,token_limit=max_token_length,skip_tokens=skip_tokens)
        except FileNotFoundError:
            return "File not found."
        except Exception as e:
            return str(e)
    else:
        return truncate_output_by_token(download_file(resource_url,verify=ssl_verify),token_limit=max_token_length,skip_tokens=skip_tokens)



class FuncSignature(BaseModel):
    doc: Any


# --------------------- Dynamic Models ---------------------


class DynamicDataset(BaseModel):
    id: str  # CKAN dataset id
    view_url: Optional[str] = None

    class Config:
        extra = "allow"

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
        validated_resources = [DynamicResource(**resource) for resource in resources]
        values["resources"] = validated_resources
        return values

    @classmethod
    def from_ckan(cls, package: Package) -> "DynamicDataset":
        data = package.as_dict() if hasattr(package, "as_dict") else package.__dict__
        return cls(**data)


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


def user_input_to_model_request(input_str: str) -> ModelRequest:
    user_prompt = UserPromptPart(content=input_str)
    return ModelRequest(parts=[user_prompt], kind="request")


def exception_to_model_response(exc: Exception) -> ModelResponse:
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
        error_text = f"An unexpected error occurred: {type(exc).__name__}: {exc}"
    error_part = TextPart(content=error_text)
    return ModelResponse(
        parts=[error_part],
        model_name="pydanticai",
        timestamp=datetime.now(timezone.utc),
        kind="response",
    )


# --------------------- Vector & RAG Models ---------------------


class VectorMeta(BaseModel):
    id: int
    chunk_id: Optional[int] = None
    chunks: Optional[HttpUrl] = None
    dataset_id: Optional[str] = None
    dataset_url: Optional[HttpUrl] = None
    groups: Optional[List[str]] = None
    private: Optional[str] = None
    resource_id: Optional[str] = None
    source: Optional[HttpUrl] = None
    view_url: Optional[List[HttpUrl]] = None


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta


@agent.tool
async def rag_search(
    ctx: RunContext[Deps], search_query: List[str], limit: int = 3
) -> List[RagHit]:
    if not ctx.deps.milvus_client:
        return "The Milvus Client was not setup properly, no rag_serach supported in the moment."
    emb_r = await ctx.deps.openai.embeddings.create(
        input=search_query,
        model="text-embedding-3-small",
        dimensions=ctx.deps.vector_dim,
    )
    query_vectors = [vec.embedding for vec in emb_r.data]
    search_res = ctx.deps.milvus_client.search(
        collection_name=ctx.deps.collection_name,
        data=query_vectors,
        search_params={"metric_type": "COSINE", "params": {"level": 1}},
        limit=limit,
        output_fields=list(VectorMeta.__fields__.keys()),
        consistency_level="Bounded",
    )
    if search_res:
        hits = []
        for i in range(len(query_vectors)):
            hits += [RagHit(**item) for item in search_res[i]]
        return [hit.json() for hit in hits]
    else:
        return []


def get_user_token(user_id: str) -> Optional[str]:
    user = CKANmodel.User.get(user_reference=user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    parameters = {"user": user.name, "name": "chat_agent"}
    try:
        response = toolkit.get_action("api_token_create")(context, parameters)
    except Exception as e:
        return e
    if "token" in response.keys():
        token = response["token"].decode("utf-8")
        return token
    return None
