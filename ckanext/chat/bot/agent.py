import asyncio
import aiofiles
import aiohttp
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import ckan.model as CKANmodel
import ckan.plugins.toolkit as toolkit
import nest_asyncio
import requests
import tiktoken
import regex

from ckan.lib.lazyjson import LazyJSONObject
from ckan.model.package import Package
from ckan.model.resource import Resource
from openai import AsyncAzureOpenAI
from openai.resources.embeddings import Embeddings as OAI_Embeddings
from pydantic import (BaseModel, ConfigDict, HttpUrl, ValidationError,
                      computed_field, root_validator, validator)
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import (AgentRunError, FallbackExceptionGroup,
                                    ModelHTTPError, ModelRetry,
                                    UnexpectedModelBehavior,
                                    UsageLimitExceeded)
from pydantic_ai.messages import (ModelMessagesTypeAdapter, ModelRequest,
                                  ModelResponse, TextPart, UserPromptPart)
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits
from pymilvus import MilvusClient

# Allow nested event loops.
nest_asyncio.apply()

import logfire

os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] = 'http://docker-dev.iwm.fraunhofer.de:4318'  
# logfire.configure(send_to_logfire=False)  
# logfire.instrument_pydantic_ai()
# logfire.instrument_httpx(capture_all=True)
# --------------------- Helper Functions ---------------------


import asyncio
import os
from flask import Flask, request, jsonify
from pydantic_ai import Agent, RunContext
import aiofiles

app = Flask(__name__)


def truncate_output_by_token(
    output: str, token_limit: int, offset: int = 0, encoding_name="cl100k_base"
) -> str:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(output)

    if len(tokens) > token_limit:
        # Skip the specified number of tokens
        truncated_tokens = tokens[offset : offset + token_limit]
        output = encoding.decode(truncated_tokens)
        #if last page of tokens, add a mark
        if len(truncated_tokens)<token_limit:
            output += "\n\n**End of Output**"

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


def download_file(url: str, headers: dict = None, verify: bool = True):
    try:
        response = requests.get(url, headers=headers, verify=verify, timeout=5)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return str(e)


# --------------------- Logging Setup ---------------------
log = __import__("logging").getLogger(__name__)

# --------------------- Model & Agent Setup ---------------------

# Azure Setup
azure_client = AsyncAzureOpenAI(
    azure_endpoint=toolkit.config.get(
        "ckanext.chat.completion_url", "https://your.chat.api"
    ),
    # api_version="2024-02-15-preview",
    api_version="2024-06-01",
    api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
)
deployment = toolkit.config.get("ckanext.chat.deployment", "gpt-4o-mini")
rag_model_settings = OpenAIModelSettings(
    model_name=deployment,
    max_tokens=16384,
    #openai_reasoning_effort= "low"
)
model = OpenAIModel(deployment, provider=OpenAIProvider(openai_client=azure_client))

# #Ollama setup
# model = OpenAIModel(
#     model_name=toolkit.config.get("ckanext.chat.deployment", "llama3.3"),
#     provider=OpenAIProvider(base_url=toolkit.config.get("ckanext.chat.completion_url", "https://ollama.local/v1"))
# )


# --------------------- Milvus and CKAN Setup ---------------------

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
embedding_model = toolkit.config.get("ckanext.chat.embedding_model", 'text-embedding-3-small')
embedding_api = toolkit.config.get("ckanext.chat.embedding_api", azure_client.embeddings)

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


@dataclass
class TextSlice:
    text: str
    offset: int
    length: int

@dataclass
class TextResource:
    url: HttpUrl=None
    text: Optional[str] = field(init=False, default=None)
    
    def extract_substring(self, offset: int, length: int) -> str:
        if self.text is None:
            raise ValueError("Text not loaded")
        text_slice=self.text[offset:offset + length]
        return TextSlice(text=text_slice,offset=offset,length=len(text_slice))

    
@dataclass
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: OpenAIModel = field(default_factory=lambda: model)
    embeddings: Union[OAI_Embeddings, str] = field(default=embedding_api)
    embedding_model: str = field(default_factory=lambda: embedding_model)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim
    file: TextResource = None

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


# --------------------- Vector & RAG Models ---------------------


class VectorMeta(BaseModel):
    id: int
    #chunk_id: Optional[int] = None
    #chunks: Optional[HttpUrl] = None
    dataset_id: Optional[str] = None
    #dataset_url: Optional[HttpUrl] = None
    #groups: Optional[list[str]] = None
    #private: Optional[str] = None
    resource_id: Optional[str] = None
    source: Optional[HttpUrl] = None
    view_url: Optional[list[HttpUrl]] = None


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta

class LitResult(BaseModel):
    title: str = ""
    summary: str = ""
    authors: str = ""
    source: Optional[HttpUrl] = None
    view_url: Optional[list[HttpUrl]] = None


class LitSearchResult(BaseModel):
    answer: str = ""
    search_str: Optional[list[str]] = None
    results: Optional[list[LitResult]] = None
    error: Optional[List[str]] = None

# --------------------- System Prompt & Agent ---------------------

system_prompt = (
    "Role:\n\n"
    "You are an assistant to a CKAN software instance that must execute tool commands and assess their success or failure. Do not provide endless examples; instead focus on running tools and reasoning based on their outputs and execute steps in your chain of thought right away. Reduce Thinking output to a minimum.\n"
    "Key Guidelines:\n\n"
    "- Always present view URLs for resources you present. Utilize the `get_ckan_url_patterns` tool to obtain the URL patterns for constructing these links.\n"
    "- For highlighting text in documents, use the specific `markdown_view.highlight` URL pattern. The pattern is `/dataset/<pkg_id>/resource/<id>/highlight/<int:start>/<int:end>`.\n"
    "- Prefer view URLs for CKAN resources over download URLs, find them by action 'resource_view_list'.\n"
    "- Ensure that you present these URLs in a clear and organized manner, directly linking to the relevant sections of the documents.\n"
    "\n"
    "- For update or patch actions, always present the proposed changes to the user and ask for explicit confirmation before proceeding.\n"
    "- When turning off SSL verification in resource downloads (by setting `ssl_verify=False`), notify the user and request confirmation before proceeding.\n"
    "- For general dataset searches and overviews, prioritize using the `package_search` action with parameter q= to fetch all datasets add parameter include_private=true to also list private datasets.\n"
    "- If you output links, ensure you use CKAN URL patterns and highlight formatting where necessary.\n"
    "- If you need to look up literature, try the `literature_search`. Only use it if no resource is mentioned.\n"
    "- If you need to find relevant parts in documents run get_resource_file_contents with a download url once, the resource will be available as long as you dont use get_resource_file_contents again. Then run 'literature_analyse'. Use 'offset' and 'max_length' parameter to analyze substrings of the document[offset:max_length]. If you ommit max_length the whole text will be loaded at once.\n"
    "- Ensure you select the appropriate tool based on the user's request and available capabilities.\n"
    "\n"
    "Your Toolset:\n\n"
    "1. **List CKAN Actions:**\n"
    "   - **Function:** `get_ckan_actions() -> List[str]`\n"
    "   - **Purpose:** Retrieves a complete list of available CKAN actions by name.\n"
    "\n"
    "2. **Get Function Information:**\n"
    "   - **Function:** `get_action_info(action_key: str) -> dict`\n"
    "   - **Purpose:** Provides detailed information for a specified CKAN action.\n"
    "\n"
    "3. **Execute CKAN Action:**\n"
    "   - **Function:** `run_action(action_name: str, parameters: Dict) -> dict`\n"
    "   - **Purpose:** Executes a specified CKAN action with provided parameters.\n"
    "\n"
    "4. **Retrieve CKAN URL Patterns:**\n"
    "   - **Function:** `get_ckan_url_patterns() -> List[RouteModel]`\n"
    "   - **Purpose:** Fetches all URL patterns in CKAN.\n"
    "\n"
    "5. **Download Resource Contents:**\n"
    "   - **Function:** `get_resource_file_contents`\n"
    "   - **Purpose:** Retrieves file content from CKAN or external sources and makes it permanent availbale to tools, as long as not run again.\n"
    "   - **When to Use:** To fetch the contents of a file resource. If SSL verification is to be disabled (i.e., `ssl_verify=False`), notify the user and ask for confirmation before proceeding.\n\n"
    "\n"
    "6. **Retrieve Documents:**\n"
    "   - **Function:** `literature_search(search_question: str, num_results: int=5) -> list[str]`\n"
    "   - **Purpose:** Performs a literature search.\n"
    "\n"
    "7. **Analyse Document:**\n"
    "   - **Function:** `literature_analyse(question: str, offset: int, max_length: int) -> list[str]`\n"
    "   - **Pagination:** To iterate through content you must increase the offset parameter by the amount of max_length of the previous call.\n"
    "   - **Purpose:** Reads a document part already loaded and analyzes it concerning the question asked.\n"
)

agent = Agent(
    model=model,
    deps_type=Deps,
    system_prompt="".join(system_prompt),
    retries=3,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)


# --------------------- Vector & RAG Models ---------------------

from datetime import datetime
from uuid import UUID


rag_prompt = (
    "Role:\n\n"
    "You are an assistant doing literature research by the question you were ask and looking up a vector store to a CKAN software instance that must execute tool commands and assess their success or failure. Do not provide endless examples; instead focus on running tools and reasoning based on their outputs and execute steps in your chain of tought right away.\n"
    "Key Guidelines:\n\n"
    "- Lookup literature, by 'rag_search'. If it indicates that the Milvus client is not set up, switch to `package_search` with search_str.\n"
    "- Start by using rag_search with exactly the original questions passing 'num_result' of hits we aim for as limit.\n\n"
    "- Create LitResult objects by aggregating RagHits by the same source property. Count the number of valid LitResults. If the number of LitResult s does not match the number of results requested, you must run the rag_search again by rephasing questions, but stay as close as possible to the context of the original question, till the necessary number of results is reached. If you fail to reach this goal, name the reason and add it to the error field.\n\n"
    "- Beside the results also return the phrases you used for the search in the search_str field as list of strings.\n\n"
    "- Access the LitResult documents and formulate a comprehensive answer. Update the LitResult objects adding the title, authors and a summary why its relevant to the answer you formulated.\n\n"
    "Your Toolset:\n\n"
    "1. **Retrieve Document hits:**\n"
    "- **Function:** `rag_search`\n"
    "- **Purpose:** Performs a vector search on document chunks using a list of search strings. Results limit can be set by limit parameter\n"
    "2. **Download Resource Contents:**\n"
    "- **Function:** `get_resource_file_contents`\n"
    "- **Purpose:** Retrieves file content from CKAN or external sources, with options for partial content retrieval using offset and max_length parameters.\n"
    "- **Pagination:** To iter trough content you must increase the offset parameter by the amount of max_length of the previous call.\n"
    "- **When to Use:** To fetch the contents of a file resource. If SSL verification is to be disabled (i.e., `ssl_verify=False`), notify the user and ask for confirmation before proceeding.\n\n"
    "3. **Execute CKAN 'package_search' Action:**\n"
    "   - **Function:** `run_action(action_name: 'package_search', parameters: {'q': search_str}) -> dict`\n"
    "   - **Purpose:** Executes a specified CKAN action with provided parameters.\n"
    "\n"
)

rag_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=LitSearchResult,
    system_prompt="".join(rag_prompt),
    retries=3,
    model_settings=rag_model_settings,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)

class SliceModel(BaseModel):
    start: int
    end: int


class AnalyseResult(BaseModel):
    answer: str = ""
    text_slices: Optional[list[SliceModel]] = None
    error: Optional[List[str]] = None

doc_analyse_prompt = (
    "Role:\n\n"
    "You are an assistant analysing scientific documents by the question you were ask.\n"
    "Key Guidelines:\n\n"
    "- Find the charcter indexes of start and end of the passage by using 'find_text_slice_offsets'. You have to use very close matching sub strings o the document text or it will fail.\n\n"
    "- You must add all relevant passages to the list of 'text_slices' add 'offset' to the tuple returned by 'find_text_slice_offsets' the get the absolute index.\n\n"
    "- Construct highlight URLs for a specific section in a document, use the find_text_slice_offsets tool to determine the start and end indices of the section, then retrieve the URL pattern using the get_ckan_url_patterns tool with endpoint='markdown_view.highlight'; if the pattern does not exist, fall back to providing the download URL of the resource along with the start and end indices.\n"
    "Your Toolset:\n\n"
    "1. **Index of Text Parts:**\n"
    "- **Function:** `find_text_slice_offsets`\n"
    "- **Purpose:** Finds start_str substring and end_str substring that follows start_str and will return the character index of that slice inside the text of the resource, counting from top of document. Use exact short strings 10-20 characters of start and end of the paragraph in scope that must be part of the original text.\n"
    "- **When to Use:**\n"
    "   - For pointing to text parts.\n"
    )

doc_agent = Agent(
    model=model,
    deps_type=TextSlice,
    output_type=AnalyseResult,
    system_prompt="".join(doc_analyse_prompt),
    retries=3,
    model_settings=rag_model_settings,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
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
    methods: Optional[list[str]] = []
    variables: Optional[list] = []
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
def get_ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN if the argument endpoint is None or empty it wil return a list of endpoints. If set to an endpoint it will return the RouteModel containing arguements and the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. If set returns the details of the endpoint. Defaults to "".

    Returns:
        RouteModel: All details on the Route
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
    if endpoint and endpoint in routes.keys():
        return routes[endpoint].json()
    else:
        endpoints = [str(key) for key in routes.keys()]
        return f"route endpoint not found. List of endpoints: {endpoints}"


def find_route_by_endpoint(endpoint: str) -> Optional[RouteModel]:
    if endpoint in routes.keys():
        return routes[endpoint]
    return None


@agent.tool_plain
def get_ckan_actions() -> List[str]:
    """Lists all avalable CKAN actions by action name

    Returns:
        List[str]: List of names of CKAN actions
    """
    from ckan.logic import _actions
    actions=[key for key in _actions.keys() if not "_update" in key]
    return actions


@agent.tool_plain
def get_action_info(action_key: str) -> dict:
    """Get the doc string of an action

    Args:
        action_key (str): the str the acton is named after

    Returns:
        dict: a dictionary containing the doc string and the arguments definitions needed to run the action
    """
    from ckan.logic.action.get import help_show

    func_model = FuncSignature(doc=help_show({}, {"name": action_key}))
    return func_model.model_dump()


@agent.tool
@rag_agent.tool
def run_action(ctx: RunContext[Deps], action_name: str, parameters: Dict) -> Any:
    """Run CKAN actions basd on the action name and parameters as a dict.

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencys at runtime, passed in by agent framework by default
        action_name (str): Name of the action to run
        parameters (Dict): Dict of Parameters to be passed to the action

    Returns:
        Any: Output of the action run
    """
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


@agent.tool
@rag_agent.tool
@doc_agent.tool
async def get_resource_file_contents(
    ctx: RunContext[Deps],
    resource_id: str,
    resource_url: str,
    offset: int,
    max_length: int=0,
    ssl_verify=True,
) -> str:
    """Retrieves the content of a resource stored in filetore, allows setting max_length of output and offset to extract a slice of content

    Args:
        resource_id (str): The UUID of the CKAN resource
        resource_url (str): The download url of the CKAN resource
        max_length (int): the maximum length of the string to return, if zero, tries to fetch whole document
        offset (int): ommits the content indexing from the start of the contents, to enable pagination
    Returns:
        str: the raw string content of the file retrieved
    Pagination Instructions:
        - Use the `offset` parameter to skip a certain number of tokens (words) from the start of the document.
        - If you want to retrieve specific chunks of the content, you can adjust the `max_length` accordingly.
        - To paginate through the document, call this function multiple times with increasing values for `offset`.
        - When the end of the document is reached, the output will include "End of Document" to indicate that there are no more contents to retrieve.
    """

    if ctx.deps.file:
        log.debug(ctx.deps.file.url)
        log.debug(resource_url)
        if str(ctx.deps.file.url)==resource_url:
            msg=f"ressource already loaded"
            log.debug(msg)
            return msg
    ckan_url = toolkit.config.get("ckan.site_url")
    try:
        resource = TextResource(url=resource_url)
        """Asynchronously download and store text from the URL."""
        ckan_url = toolkit.config.get("ckan.site_url")
        if ckan_url in resource_url:
            storage_path = toolkit.config.get("ckan.storage_path", "/var/lib/ckan/default")
            
            # Generate folder structure based on resource_id
            first_level_folder = resource_id[:3]
            second_level_folder = resource_id[3:6]
            file_name = resource_id[6:]

            # Construct full file path
            file_path = os.path.join(
                storage_path,
                "resources",
                first_level_folder,
                second_level_folder,
                file_name,
            )

            log.debug(f"Loading CKAN resource file from: {file_path}")

            try:
                async with aiofiles.open(file_path, "r") as file:
                    resource.text = await file.read()
            except Exception as e:
                raise RuntimeError(f"Failed to read CKAN resource file: {e}")
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(resource_url,verify_ssl=ssl_verify) as response:
                        response.raise_for_status()
                        resource.text = await response.text()
            except Exception as e:
                resource.text = ""
                raise RuntimeError(f"Failed to download from {resource_url}: {e}")
        
        ctx.deps.file = resource
        msg=f"TextResource downloaded form {resource_url}"
    except Exception as e:
        msg=f"Failed to download and add TextResource: {e}"
    log.debug(msg)
    return msg

def fuzzy_search(pattern: str, text: str, threshold: float = 0.8) -> tuple[str, int, int]:
    max_err = max(1, int((1 - threshold) * len(pattern)))

    def try_match(pat: str):
        fuzzy_pat = f"({pat})" + f"{{e<={max_err}}}"
        return regex.search(
            fuzzy_pat,
            text,
            flags=regex.BESTMATCH | regex.IGNORECASE | regex.DOTALL
        ), fuzzy_pat

    # First try: raw pattern
    try:
        match, fuzzy_pat = try_match(pattern)
    except regex.error as e:
        log.warning(f"Initial regex failed for pattern '{pattern}': {e}")
        match = None

    # Fallback: escaped pattern
    if not match:
        escaped = regex.escape(pattern)
        try:
            match, fuzzy_pat = try_match(escaped)
        except regex.error as e:
            log.error(f"Escaped regex also failed for pattern '{escaped}': {e}")
            return "", -1, -1

    log.debug(f"Tried to match: '{pattern}' with pattern: '{fuzzy_pat}' - found: {match}")
    
    if not match:
        return "", -1, -1

    return match.group(1), match.start(1), match.end(1)

@doc_agent.tool
def find_text_slice_offsets(ctx: RunContext[TextSlice],
                            start_str: str,
                            end_str: str,
                            threshold: float = 0.8) -> tuple[int, int] | str:
    log.debug(ctx)
    text=ctx.deps.text
    offset=ctx.deps.offset
    slice_length=ctx.deps.length
    start_match, start_idx, start_end_idx = fuzzy_search(start_str, text, threshold)
    if start_idx < 0:
        log.debug(f"Tried to start pattern: '{start_str}' - but didn't find a match")
        return f"Start string not found: '{start_str}'"

    # Search from the end of the start match
    tail = text[start_end_idx:]
    end_match, rel_end_idx, rel_end_idx_end = fuzzy_search(end_str, tail, threshold)
    if rel_end_idx < 0:
        log.debug(f"Tried to end pattern: '{end_str}' - returning default span")
        return (offset+start_idx, offset+slice_length)

    abs_end_idx = start_end_idx + rel_end_idx_end
    log.debug(f"Matched '{start_str}...{end_str}' at {start_idx}, end at {abs_end_idx}")
    return (offset+start_idx, offset+abs_end_idx)

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



async def get_embedding(chunks: List[str], model: str, api_url, vector_dim: int):
    if not isinstance(api_url,str):
        # must be OAI embeddings
        emb_r = await api_url.create(
            input=chunks,
            model=model,
            dimensions=vector_dim
        )
        return [vec.embedding for vec in emb_r.data]
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        'chunks': chunks,
        'model': model
    }
    response = requests.post(api_url, headers=headers, data=json.dumps(data),verify=False)
    
    if response.status_code == 200:
        return response.json()['embeddings']
    else:
        return {'error': response.status_code, 'message': response.text}


@rag_agent.tool
async def rag_search(
    ctx: RunContext[Deps], search_query: List[str], limit: int = 3
) -> List[RagHit]:
    """Vector rag serach using Milvus vector store

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencys at runtime, passed in by agent framework by default
        search_query (List[str]): A list of strings or which to do the vector search with.
        limit (int, optional): Limit for amount of Hits to be returned for the serach. Defaults to 3.

    Returns:
        List[RagHit]: List of RagHit instances as a result of rag search. the object provided a distance attribute with the metrics of similarity and an entity attribute containing the meta data of the vector entity in store.
    """
    if not ctx.deps.milvus_client or not ctx.deps.embeddings:
        return "The Milvus Client was not setup properly, no rag_search supported in the moment."
    else:
        query_vectors= await get_embedding(search_query,model=ctx.deps.embedding_model,api_url=ctx.deps.embeddings, vector_dim=ctx.deps.vector_dim)
        num_results=0
        hits = []
        filter_ids=[]
        while num_results <limit:
            log.debug(f"{search_query} filtered by: {filter_ids}")
            search_res = ctx.deps.milvus_client.search(
                collection_name=ctx.deps.collection_name,
                data=query_vectors,
                search_params={"metric_type": "COSINE", "params": {"level": 1}},
                limit=6,
                filter_params = {"ids": filter_ids} if filter_ids else None,
                filter="id not in {ids}"if filter_ids else None,
                output_fields=list(VectorMeta.__fields__.keys()),
                consistency_level="Bounded",
            )
            if search_res:
                for i in range(len(query_vectors)):
                    hit = [RagHit(**item) for item in search_res[i]]
                    hits += hit
                    filter_ids+= list(set(hit.id for hit in hits))
                    #log.debug(hits)
                distinct_sources = list(set(hit.entity.source for hit in hits))
                num_results=len(distinct_sources)
                log.debug(f"Rag search for:{search_query} with limit: {limit} returned {num_results} results.")
        return hits
        


@agent.tool
async def literature_search(ctx: RunContext[Deps], search_question: str, num_results: int=5) -> list[str]:
    r = await rag_agent.run(
        f"Search for documents using this question:{search_question}. You must return {num_results} results",
        deps=ctx.deps, usage_limits=UsageLimits(request_limit=10),
    )
    log.debug(r.data)
    return r.data.json()

@agent.tool
async def literature_analyse(ctx: RunContext[Deps], question: str, offset: int, max_length: int) -> list[str]:
    doc=ctx.deps.file
    if not doc:
        return f"no document loaded"
    text_part=doc.extract_substring(offset=offset,length=max_length)
    log.debug(text_part)
    r = await doc_agent.run(
        f"Read trough: {text_part} and find if there is an answer to:{question}.",deps=text_part,
        usage_limits=UsageLimits(request_limit=50),
    )
    log.debug(r.data)
    return r.data.json()


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
