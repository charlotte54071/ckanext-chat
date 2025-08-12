import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import aiofiles
import aiohttp
import ckan.model as CKANmodel
import ckan.plugins.toolkit as toolkit
# import logfire
# import nest_asyncio
import requests

from flask import Flask
from loguru import logger

from openai.resources.embeddings import Embeddings as OAI_Embeddings
from pydantic import (BaseModel, HttpUrl)
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import (AgentRunError, FallbackExceptionGroup,
                                    ModelHTTPError, ModelRetry,
                                    UnexpectedModelBehavior,
                                    UsageLimitExceeded)
from pydantic_ai.messages import (ModelMessagesTypeAdapter, ModelRequest,
                                  ModelResponse, TextPart, UserPromptPart)
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
# from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.usage import UsageLimits
from pymilvus import MilvusClient
from ckanext.chat.bot.utils import process_entity, unpack_lazy_json, RouteModel, get_ckan_url_patterns, CKAN_ACTIONS, get_ckan_action, fuzzy_search_early_cancel, FuncSignature, DynamicDataset, DynamicResource, truncate_output_by_token, get_schema_aware_search_context, enhance_search_query_with_schema_context, filter_datasets_by_schema, get_schema_specific_field_mappings, suggest_schema_based_actions
from difflib import get_close_matches

log = logger.bind(module=__name__)

# # Allow nested event loops.
# nest_asyncio.apply()


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://docker-dev.iwm.fraunhofer.de:4318"
# logfire.configure(send_to_logfire=False)
# logfire.instrument_pydantic_ai()
# logfire.instrument_httpx(capture_all=True)
# --------------------- Helper Functions ---------------------

app = Flask(__name__)

# --------------------- Model & Agent Setup ---------------------

# Azure Setup
deployment = toolkit.config.get("ckanext.chat.deployment", "gpt-4o-mini")
rag_model_settings = OpenAIModelSettings(
    model_name=deployment,
    max_tokens=16384,
    # openai_reasoning_effort= "low"
)
model = OpenAIModel(
    "gpt-4o",
    provider=AzureProvider(
        azure_endpoint=toolkit.config.get(
            "ckanext.chat.completion_url", "https://your.chat.api"
        ),
        api_version="2024-06-01",
        api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
    ),
)

think_model = OpenAIModel(
    "gpt-4.1-mini",
    provider=AzureProvider(
        azure_endpoint=toolkit.config.get(
            "ckanext.chat.completion_url", "https://your.chat.api"
        ),
        api_version="2024-06-01",
        api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
    ),
)

# model = OpenAIModel(deployment, provider=OpenAIProvider(openai_client=azure_client))

# #Ollama setup
# model = OpenAIModel(
#     model_name=toolkit.config.get("ckanext.chat.deployment", "llama3.3"),
#     provider=OpenAIProvider(base_url=toolkit.config.get("ckanext.chat.completion_url", "https://ollama.local/v1"))
# )


# --------------------- Milvus and CKAN Setup ---------------------

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
milvus_token = toolkit.config.get("ckanext.chat.milvus_token", os.getenv("MILVUS_TOKEN", ""))

milvus_client = MilvusClient(uri=milvus_url, token=milvus_token)  # 关键：token

collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
embedding_model = toolkit.config.get(
    "ckanext.chat.embedding_model", "text-embedding-3-small"
)
embedding_api = toolkit.config.get("ckanext.chat.embedding_api", "")

vector_dim = None
if milvus_url:
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
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: OpenAIModel = field(default_factory=lambda: model)
    embeddings: Union[OAI_Embeddings, str] = field(default=embedding_api)
    embedding_model: str = field(default_factory=lambda: embedding_model)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim
    #file: Optional[TextResource] = None

@dataclass
class TextSlice:
    url: HttpUrl
    text: str
    offset: int
    length: int
    doc_position: float
    update_url_on_init: bool = field(default=True, repr=False)

    def __post_init__(self):
        if not self.update_url_on_init:
            return
        res_uuid = extract_resource_uuid(str(self.url))
        pkg_uuid = extract_dataset_uuid(str(self.url))
        ckan_url = toolkit.config.get("ckan.site_url")
        if res_uuid and pkg_uuid:
            endpoint = "markdown_view.highlight"
            route = get_ckan_url_patterns(endpoint)
            # if route_info is a string, it's an error message
            if isinstance(route, RouteModel):
                fill_vars = {"pkg_id": pkg_uuid, "id": res_uuid, "start": self.offset, "end": self.offset+self.length} # Replace with correct variable names for this route
                self.url = route.build_url(base_url=ckan_url,fill=fill_vars)

    
    
@dataclass
class TextResource:
    url: HttpUrl = None
    _text: Optional[str] = field(init=False, default=None)

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, value: Optional[str]):
        self._text = value
        self.length = len(value) if value is not None else 0

    length: int = field(init=False, default=0)

    def extract_substring(self, offset: int, length: int) -> TextSlice:
        if self.text is None:
            raise ValueError("Text not loaded")
        text_slice = self.text[offset:offset + length]
        position=float(offset + len(text_slice)) / float(self.length)
        log.debug(f'extracted substring with offset {offset} and length {length}, end position relativ to document {position}')
        return TextSlice(url=self.url,text=text_slice, offset=offset, length=len(text_slice),doc_position=position)
    
    def __getstate__(self):
        # Exclude _text from serialization
        state = self.__dict__.copy()
        state["_text"] = None  # Don't serialize large text
        return state
    

# --------------------- Vector & RAG Models ---------------------


class VectorMeta(BaseModel):
    id: int
    # chunk_id: Optional[int] = None
    # chunks: Optional[HttpUrl] = None
    dataset_id: Optional[str] = None
    # dataset_url: Optional[HttpUrl] = None
    # groups: Optional[list[str]] = None
    # private: Optional[str] = None
    resource_id: Optional[str] = None
    source: Optional[HttpUrl] = None
    #view_url: Optional[list[HttpUrl]] = None


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta


class LitResult(BaseModel):
    title: str = ""
    summary: str = ""
    authors: str = ""
    source: Optional[HttpUrl] = None
    #view_url: Optional[list[HttpUrl]] = None


class LitSearchResult(BaseModel):
    answer: str = ""
    search_str: Optional[list[str]] = None
    results: Optional[list[LitResult]] = None
    error: Optional[List[str]] = None

class AnalyseResult(BaseModel):
    answer: str = ""
    source: HttpUrl
    text_slices: Optional[list[TextSlice]] = None
    error: Optional[List[str]] = None

class CKANResult(BaseModel):
    status: Literal['success', 'fail']
    action_name: str = ""
    parameters: Optional[Dict[str,Any]] = {} 
    doc: Optional[FuncSignature]
    result: str
    comment: Optional[str]

# --------------------- Updated RAG Agent Prompt ---------------------
rag_prompt = (
    "Role:\n\n"
    "You perform literature retrieval using a vector store and return scientific citations in markdown format.\n"
    "- Use rag_search with the original question.\n"
    "- Aggregate results by `source` into LitResult objects. Use the source field in the vector meta data.\n"
    "- For each source, return a markdown citation in the format: [1](url)\n"
    "- Add a summary why the source is relevant.\n"
    "- Retry search if fewer than N distinct sources are returned.\n"
)

# --------------------- Updated Document Agent Prompt ---------------------

doc_prompt = (
    "Role:\n\n"
    "You are a document analysis agent tasked with answering a question based on a long document `doc`. "
    "Your goal is to find and cite the most relevant passages from anywhere in the document — not just the beginning — "
    "using an adaptive strategy like a human researcher would.\n\n"

    "Instructions:\n\n"
    "1. Begin by searching for a **Table of Contents** (ToC) or **summary sections**.\n"
    "   - Use `get_text_slice(doc, offset=0, length=10000)` to fetch the beginning for this purpose.\n"
    "   - If a ToC exists, extract its structure to guide your navigation.\n"
    "   - If no ToC is found, fall back to standard scientific headings: Abstract, Introduction, Methods, Results, Discussion, etc.\n\n"

    "2. Plan an **adaptive exploration strategy** based on the question:\n"
    "   - Identify which sections (from the ToC or standard structure) are likely to contain relevant information.\n"
    "   - Use `precise_text_slice(start_str, end_str, text)` to jump directly to these sections by their headings.\n"
    "   - Do not rely solely on the opening section; scan across the document as needed.\n\n"

    "3. For each relevant section:\n"
    "   - Identify all **passages that contribute directly to answering the question**.\n"
    "   - Extract them using `precise_text_slice(start_str, end_str, text)` with exact 10–20 character substrings.\n"
    "   - Record them as `text_slice` objects.\n\n"

    "4. Write your answer:\n"
    "   - Synthesize the findings into a coherent response.\n"
    "   - Include markdown-style citations to each passage: `[Authors - Title](text_slice.url)`.\n"
    "   - Only use 'text_slice.url' or citations in the text that u have read to cite.\n"
    "   - Every major claim or quoted content must be cited.\n\n"

    "5. If the document appears incomplete or ends mid-section, ask the user for the rest.\n\n"

    "Important:\n"
    "- Your goal is to **simulate how a skilled researcher would navigate and extract evidence**.\n"
    "- Use the ToC (if available) or section headings to jump around. Avoid linear reading unless the document is very short.\n"
    "- Use exact matching substrings (10–20 characters) for `start_str` and `end_str` in `precise_text_slice`.\n"
    "- Always include the document's `doc.url` as `source` in your output.\n"
)


# --------------------- Updated Front Agent ---------------------
front_agent_prompt = """
When the user asks to register/create/add a dataset and mentions a schema (e.g., “device”), your first two tool calls must be get_schema_context() and then get_schema_field_suggestions('<schema>'). Do not output generic CKAN fields. Only list fields returned by get_schema_field_suggestions, marking required vs optional.

— Search result listing with links —
After ckan_run("package_search"), you MUST present the results as a numbered list. 
For each dataset result:
  • Show the title as a markdown link using result.view_url when available.
  • If view_url is missing, build a fallback link as: <ckan.site_url>/dataset/<name-or-id>.
  • Append privacy in italics, e.g., "(Private)" or "(Public)".
  • If a schema type exists (result.type), show it in backticks.

Example format:
1. [<title>](<view_url>) *(Private)* — `<type>`
2. [<title 2>](<fallback_link_if_needed>) *(Public)* — `<type>`

If the user asks “link/links/URL” (e.g., “can you give me the link”), ALWAYS return the list of clickable links as above. 
Do not just say you found results; show the links.
If there are >10 results, show the top 5–10 and ask whether to show more.


🚨 FIRST PRIORITY: DATASET REGISTRATION 🚨
Before doing ANYTHING else, check if the user wants to register/create/add a dataset.
If YES, immediately call get_schema_context and get_schema_field_suggestions. NEVER show generic dataset fields when schema-specific fields exist.

You are a coordinator agent.

🚨 CRITICAL DATASET REGISTRATION RULE 🚨
If a message contains 'register'/'create'/'add' + 'dataset' (especially 'device dataset', 'project dataset', etc.):
1) Call get_schema_context
2) Call get_schema_field_suggestions(<schema>)
3) Only use the returned fields (clearly mark required vs optional)

— Hard rule: prefer CKAN over RAG for summaries —
For intents like “summarise/summarize”, “show”, “describe”, “details of”, or “tell me about” a dataset:
1) Extract <name> (and optional <schema>).
2) Call:
   ckan_run("package_search", {"q": 'title:"<name>" OR name:"<name>"', "include_private": true, ("fq": f"type:{<schema>}") if <schema> else omit})
3) If count > 0:
   • Choose an exact match on title/name if possible; otherwise ask to disambiguate.
   • Call ckan_run("package_show", {"id": <chosen_id>})
   • Then call get_schema_field_suggestions(<dataset.type>) and use labels to render customized fields.
4) ONLY if count == 0, use literature_search / literature_analyse.

— Dataset summary rendering (markdown) —
After package_show (and schema fetch), render a compact markdown summary. Do NOT show tool-call/debug blocks to the user.
Rules:
- Begin with: `Here is a summary of the dataset **<title>**:`
- Use bullets. Only show fields that exist.
- Base fields (in order):
  • **Title:** <title>
  • **Description:** <notes>
  • **Author:** <author> (Email: <author_email>)
  • **Maintainer:** <maintainer> (Email: <maintainer_email>)
  • **License:** <license_title or id> (link if available)
  • **Tags:** comma-separated
  • **Version:** <version>
  • **Groups:** list names (link if available)
  • **Organization:** <organization.title or name>
  • **Private:** Yes/No
  • **Spatial Area:** from `spatial`, `spatial_text`, bbox or relevant extras if present
- Schema-specific fields:
  • From get_schema_field_suggestions(dataset.type), include non-empty values using the field **label** (not raw key).
  • Prefer up to ~10 that best explain the dataset; arrays → comma-joined; URLs → markdown links.
- End with: `For more details, you can view the dataset [here](<view_url>).` if available.

— “Show more / all / customized fields” —
If the user asks for “show more fields”, “all fields”, or “customized fields” for a dataset:
1) Ensure you have package_show; if not, search → show as above.
2) Call get_schema_field_suggestions(dataset.type).
3) Re-render including **all non-empty schema-specific fields** (label + value). Still skip empty/null.

For any question not clearly about CKAN entities (and not a summary/show/describe intent), you may use literature_search. Rephrase queries for vector search. Use literature_analyse to extract exact passages.

Execution & verification:
- Before any write operation, show the plan and ask for confirmation.
- If ssl_verify=False is needed for downloads, ask for confirmation.

Schema-aware features:
- Use get_schema_context to know supported schemas (dataset, device, digitaltwin, geoobject, method, onlineapplication, onlineservice, project, software).
- When searching, consider fq=type:<schema>.
- When presenting dataset info, include the schema type and relevant schema fields.

Guidelines:
- If ckan_run fails, adapt the next call based on the error hints; add default params if needed.
- CKAN model: Packages contain Resources; each Package belongs to one Organization; may join multiple Groups; Views attach to Resources.
- Use get_ckan_actions to explore available actions.
- For broad searches: ckan_run("package_search", {"q": "", "include_private": true}).
- Include view URLs when available.
- Output formulas as LaTeX inline with $$...$$.

Avoid assumptions:
- Don’t invent formats, content, or links. Don’t output placeholder links or data.
"""



# --------------------- Updated Research Agent ---------------------
research_agent_prompt = (
    "You are a coordinator agent, equipped to think through user questions and perform thorough literature analysis.\n"
    "- Begin by thinking through the user's question: identify key concepts, potential data sources, and related sub‑questions or topics to explore.\n"
    "- Call `literature_search` and `literature_nanalysis` repeatedly to investigate further into your findings, till you found a detailed and throughout answer to the user question."
    "- For any question not directly related to CKAN entities like datasets or resources, call `literature_search` as your first step.\n"
    "- Do NOT assume sources of information! Always try `literature_search` first, unless the user explicitly provides a specific source.\n"
    "- When using `literature_search`, do not pass the user prompt verbatim. Instead, rephrase the query to optimize vector similarity search matching.\n"
    "- If the user asked a specific question, apply `literature_analyse` to each result from `literature_search` to extract precise answers. Use the links returned by `literature_analyse` to point to the most relevant passages.\n"
    "Use the links returned by 'literature_analyse' to point to the passages most relavant in ur answer. They usually end with /highlight/<start:int>/<end:int>.\n"
    "- For every question about a given document, you must use `literature_analyse`. Provide a direct download link to the raw text format of the document.\n"
    "- For CKAN actions, after reasoning about the user's intent, formulate a clear command to `ckan_run`, including all relevant parameters you have identified.\n"
    "- Present your final results with inline markdown citations where appropriate, and suggest related questions or topics the user might explore next.\n"
    "- Execution and Verification:\n"
    "  - Before making any changes to data via CKAN, present your planned updates or actions and request user confirmation.\n"
    "  - If SSL verification is disabled (`ssl_verify=False`), explicitly request confirmation before proceeding.\n"
    "Guidelines:\n"
    "- If a `ckan_run` call fails, incorporate the tool's error hints and suggestions, adding default parameters as needed to retry successfully.\n"
    "- CKAN entities are organized as follows: Packages (datasets) contain Resources (files or links). Every Package belongs to exactly one Organization but may be in multiple Groups. Views attach to Resources based on format and user needs.\n"
    "- Use `get_ckan_actions` to retrieve available CKAN action names and their function signatures.\n"
    "- For broad dataset searches, use `ckan_run` with action `package_search` and parameters `{q: search_str, include_private: true}`. If the user does not specify a query, set `search_str` to an empty string.\n"
    "- If you are unsure how to proceed, ask the user clarifying questions or query a suitable CKAN action with `ckan_run`.\n"
    "- When presenting tool results, always include any available view URLs for direct access.\n"
    "- Output formulas as latex nline without code boxes, use $$ as delimiter.\n"
    "Avoid Assumptions:\n"
    "- Do not assume formats, content, or links without confirming their existence and relevance via primary sources.\n"
    "- Refrain from creating placeholder links or data that could mislead about available resources.\n"
    "\n"
)

# --------------------- System Prompt & Agent ---------------------

ckan_agent_prompt = (
    "Role:\n\n"
    "You are an assistant to a CKAN software instance. You execute CKAN actions, evaluate their success, return the results of 'action_run' directly as 'results' "
    "and suggest improvements or appropriate alternatives when as 'comment'.\n\n"
    # "Before returning the results, try to augment the entities in your answer with links created by 'build_ckan_url', "
    # "available routs you can get with 'ckan_url_patterns' tool.\n\n"

    "Behavior:\n"
    "- Attempt to run the specified CKAN action with the given parameters straight away, do not look up the action.\n"
    "- If the action fails or is invalid:\n"
    "  - If the action fails because of missing parameters, run the actions again with the default parameters form the documentation.\n"
    "  - return the results but mentions the corrections you made and what can be improved on next call."
    "  - Use `get_ckan_actions` to explain what the suggested action does.\n"
    "- If your action returns datasets or other CKAN objects, suggest relevant follow-up actions, e.g., "
    "- **Do not output internal reasoning. Focus only on clean, result-oriented output.**\n\n"
    "Data Search:\n"
    "- When searching for datasets, use `package_search` with `include_private=true` to ensure full visibility.\n\n"
)

agent = Agent(
    model=model,
    deps_type=Deps,
    system_prompt="".join(front_agent_prompt),
    retries=3,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)

research_agent= Agent(
    model=think_model,
    deps_type=Deps,
    system_prompt="".join(research_agent_prompt),
    retries=3,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)
ckan_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=CKANResult,
    system_prompt="".join(ckan_agent_prompt),
    retries=10,
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

doc_agent = Agent(
    model=model,
    deps_type=TextResource,
    output_type=AnalyseResult,
    system_prompt="".join(doc_prompt),
    retries=3,
    model_settings=rag_model_settings,
)


def _split_validators(v):
    if not v:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v]
    return [p for p in re.split(r"[|,\s]+", str(v)) if p]


def convert_to_model_messages(history: str) -> List:
    if history:
        history_list = json.loads(history)
        return ModelMessagesTypeAdapter.validate_python(history_list)
    return None



# --------------------- Front Agent Delegation Tools ---------------------

@agent.tool
@research_agent.tool
async def ckan_run(ctx: RunContext[Deps], command: str, parameters: dict={}) -> str:
    """
    Executes a CKAN action with the provided parameters.

    This function sends a command to the CKAN agent and waits for execution.
    It logs the command and parameters and handles possible timeouts
    and unexpected errors.
    Args:
        ctx (RunContext[Deps]): The context containing the dependencies for execution.
        command (str): The name of the CKAN action to be executed.
        parameters (dict): A dictionary of parameters required for the CKAN action.
    Returns:
        str: The result of the CKAN action as a JSON string, or an error message in case of failure.
    Raises:
        asyncio.TimeoutError: If the execution of the CKAN action exceeds the specified time (30 seconds).
        Exception: For any other unexpected errors during the execution of the CKAN action.
    """
    try:
        r = await asyncio.wait_for(
            ckan_agent.run(
                f"Run the CKAN action: '{command}' with the parameters: {parameters}. "
                "If the action fails, suggest the correct action and explain it using 'get_action_info'.",
                deps=ctx.deps,
            ),
            timeout=600
        )
    except asyncio.TimeoutError:
        msg="Timeout on ckan_run attempt, retrying..."
        log.error(msg)
        return msg
    except Exception as e:
        msg=f"Unexpected error on ckan_run attempt: {str(e)}"
        log.error(msg)
        return msg
    #log.debug(f"ckan_run return: {r.data.json()}")
    return r.data.json()
    

#@ckan_agent.tool_plain
def ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN if the argument endpoint is None or empty it wil return a list of endpoints. If set to an endpoint it will return the RouteModel containing arguements and the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. If set returns the details of the endpoint. Defaults to "".

    Returns:
        RouteModel: All details on the Route
    """
    routes=get_ckan_url_patterns(endpoint=endpoint)
    return routes

#@ckan_agent.tool_plain
def build_ckan_url(route: RouteModel, fill: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a CKAN URL for the given endpoint and fill in URL variables.

    Args:
        endpoint (str): The CKAN endpoint to build a URL for.
        fill (Optional[Dict[str, Any]]): A dictionary mapping URL variable names to their values.
        base_url (Optional[str]): Override the CKAN base site URL.

    Returns:
        str: The fully constructed CKAN URL.

    Raises:
        ValueError: If the endpoint is not found or required variables are missing.
    """
    base_url= toolkit.config.get("ckan.site_url", "")
    return route.build_url(base_url=base_url or toolkit.config.get("ckan.site_url", ""), fill=fill)



def _normalize_schema_type(s: str, candidates: list[str]):
    """standardise 'online application/geoObject dataset' as schema key。"""
    raw = s or ""
    norm = re.sub(r"[\s_-]+", "", raw.strip().lower())
    norm = re.sub(r"(dataset|data\s*set)$", "", norm)

    aliases = {
        "onlineapplication": "onlineapplication",
        "onlineapp": "onlineapplication",
        "online application": "onlineapplication",
        "onlineservice": "onlineservice",
        "online service": "onlineservice",
        "geoobject": "geoobject",
        "geo object": "geoobject",
    }
    if norm in aliases:
        return aliases[norm], f"alias '{raw}' -> '{aliases[norm]}'"

    norm_cands = {re.sub(r'[\s_-]+','', c.lower()): c for c in candidates}
    if norm in norm_cands:
        return norm_cands[norm], f"exact '{norm_cands[norm]}'"

    hit = get_close_matches(norm, list(norm_cands.keys()), n=1, cutoff=0.6)
    if hit:
        return norm_cands[hit[0]], f"fuzzy '{norm_cands[hit[0]]}'"

    return None, f"no match for '{raw}' in {candidates}"

@agent.tool_plain
@research_agent.tool_plain
@ckan_agent.tool_plain
def get_ckan_actions() -> Dict[str, FuncSignature]:
    """Lists all avalable CKAN actions by action name

    Returns:
        List[str]: List of names of CKAN actions
    """
    return get_ckan_action()


@agent.tool_plain
@research_agent.tool_plain
@ckan_agent.tool_plain
def debug_schema_tools() -> Dict[str, Any]:
    """Debug tool to check if schema tools are working correctly"""
    try:
        # Test basic configuration
        schema_aware = toolkit.asbool(toolkit.config.get("ckanext.chat.schema_aware", True))
        schemas_config = toolkit.config.get("ckanext.chat.supported_schemas", "")
        
        # Test schema loading
        test_schema = 'device'
        try:
            schema_info = toolkit.get_action('scheming_dataset_schema_show')({}, {'type': test_schema})
            schema_loaded = True
            schema_fields_count = len(schema_info.get('dataset_fields', []))
        except Exception as e:
            schema_loaded = False
            schema_fields_count = 0
            
        return {
            'schema_aware_enabled': schema_aware,
            'supported_schemas_config': schemas_config,
            'test_schema_loaded': schema_loaded,
            'test_schema_fields_count': schema_fields_count,
            'debug_message': 'Schema tools debug completed'
        }
    except Exception as e:
        return {
            'error': f'Debug failed: {str(e)}',
            'debug_message': 'Schema tools debug failed'
        }

@agent.tool_plain
@research_agent.tool_plain
@ckan_agent.tool_plain
def get_schema_context() -> Dict[str, Any]:
    """Get information about available dataset schemas and their fields.
    
    Returns:
        Dict[str, Any]: Schema context information including supported schemas and their details
    """
    return get_schema_aware_search_context()



@agent.tool_plain
@research_agent.tool_plain
@ckan_agent.tool_plain
def get_schema_field_suggestions(schema_type: str) -> Dict[str, Any]:
    log.info("[schema] called with %s", schema_type)
    schema_context = get_schema_aware_search_context()
    scs = schema_context.get("schema_contexts", {})
    candidates = list(scs.keys())

    # normalization
    normalized, note = _normalize_schema_type(schema_type, candidates)
    if not normalized:
        return {
            "status": "error",
            "error": f"Schema '{schema_type}' not exists",
            "reason": note,
            "available_schemas": candidates
        }
    if normalized != schema_type:
        log.info("[schema-normalize] %s → %s (%s)", schema_type, normalized, note)
    schema_type = normalized

    info = scs.get(schema_type, {})
    dataset_fields = info.get("dataset_fields", [])
    resource_fields = info.get("resource_fields", [])
    field_descriptions = info.get("field_descriptions", {})

    if dataset_fields and isinstance(dataset_fields[0], str):
        try:
            schema_info = toolkit.get_action('scheming_dataset_schema_show')({}, {'type': schema_type})
            dataset_fields = schema_info.get('dataset_fields', []) or []
            resource_fields = schema_info.get('resource_fields', []) or []
            field_descriptions = {
                f.get('field_name'): (f.get('label') or f.get('help_text') or f.get('field_name', ''))
                for f in dataset_fields + resource_fields if f.get('field_name')
            }
            log.info("[schema-hydrate] reloaded full field objects for schema=%s", schema_type)
        except Exception as e:
            log.warning("[schema-hydrate] failed for %s: %s", schema_type, e)

    required_fields, optional_fields = [], []
    for f in dataset_fields:
        if not isinstance(f, dict):
            raise TypeError(f"dataset field is not a dict: {type(f)} -> {f}")

        validators_list = _split_validators(f.get("validators"))
        is_required = ("not_empty" in validators_list) or ("ignore_missing" not in validators_list)

        entry = {
            "field_name": f.get("field_name", ""),
            "label": f.get("label", ""),
            "help_text": f.get("help_text", ""),
            "form_snippet": f.get("form_snippet", ""),
            "validators": f.get("validators", ""),
            "description": field_descriptions.get(f.get("field_name", ""), "")
        }
        (required_fields if is_required else optional_fields).append(entry)

    res= {
        "status": "ok",
        "schema_type": schema_type,
        "about": info.get("about", ""),
        "required_fields": required_fields,
        "optional_fields": optional_fields,
        "resource_fields": resource_fields,  
        "field_descriptions": field_descriptions,
        "creation_example": {
            "type": schema_type,
            "name": f"example-{schema_type}-dataset",
            "title": f"Example {schema_type.title()} Dataset",
            "notes": f"Description for this {schema_type} dataset"
        }
    }
    log.info("[schema] %s required=%d optional=%d",
             schema_type, len(required_fields), len(optional_fields)) 
    return res



# @ckan_agent.tool_plain
# def get_action_info(action_key: str) -> dict:
#     """Get the doc string of an action

#     Args:
#         action_key (str): the str the acton is named after

#     Returns:
#         dict: a dictionary containing the doc string and the arguments definitions needed to run the action
#     """
#     from ckan.logic.action.get import help_show

#     func_model = FuncSignature(doc=help_show({}, {"name": action_key}))
#     return func_model.model_dump()




#@agent.tool
@rag_agent.tool
@ckan_agent.tool
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
    unpacked_response = unpack_lazy_json(response)
    clean_response = process_entity(unpacked_response)
    #log.debug(clean_response)
    #log.debug("{} -> {}".format(len(str(response)), len(str(clean_response))))
    return clean_response

def extract_resource_uuid(input_string: str) -> str:
    # Regulärer Ausdruck für UUID zwischen 'resource/' und '/download'
    pattern = r'resource/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})/download'
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(1)  # Gibt die gefundene UUID zurück
    else:
        return None

def extract_dataset_uuid(input_string: str) -> str:
    pattern = r'dataset/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})/resource'
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(1)  # Gibt die gefundene UUID zurück
    else:
        return None

#@agent.tool_plain
@rag_agent.tool_plain
async def get_resource_file_contents(
    resource_url: str,
    ssl_verify=True,
) -> TextResource:
    """
    Retrieves the content of a resource stored in filetore, allows setting max_length of output and offset to extract a slice of content

    Args:
        resource_url (str): The download url of the CKAN resource

    Returns:
        TextResource: The raw string content of the file retrieved
    """
    ckan_url = toolkit.config.get("ckan.site_url")
    try:
        resource = TextResource(url=resource_url)
        resource_id = extract_resource_uuid(resource_url)
        
        if ckan_url in resource_url and resource_id:
            storage_path = toolkit.config.get("ckan.storage_path", "/var/lib/ckan/default")

            first_level_folder = resource_id[:3]
            second_level_folder = resource_id[3:6]
            file_name = resource_id[6:]

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
                    async with session.get(resource_url, ssl=ssl_verify) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("text/"):
                            raise RuntimeError(f"Unsupported MIME type: {content_type}")
                        resource.text = await response.text()
            except Exception as e:
                resource.text = ""
                raise RuntimeError(f"Failed to download from {resource_url}: {e}")

        log.info(f"TextResource downloaded from {resource_url} with length: {resource.length}")
        return resource

    except Exception as e:
        raise RuntimeError(f"Failed to download and add TextResource: {e}")

@doc_agent.tool
async def get_text_slice(ctx: RunContext[TextResource], offset: int, length: int)->TextSlice:
    return ctx.deps.extract_substring(offset=offset, length=length)


@doc_agent.tool
async def precise_text_slice(
    ctx: RunContext[TextResource],
    start_str: str,
    end_str: str,
    threshold: float = 0.9
) -> TextSlice:
    """
    Finds the start and end offsets of a text slice based on fuzzy matching of start and end strings.

    Args:
        ctx (RunContext[Deps]): The context containing the dependencies.
        start_str (str): The starting string to search for.
        end_str (str): The ending string to search for.
        threshold (float): The threshold for fuzzy matching (default is 0.9).

    Returns:
        Union[Tuple[int, int], str]: A tuple containing:
            - The start and end offsets if both strings are found.
            - An error message if the start string is not found.
    """
    if not ctx.deps.text:
        log.debug("No file loaded in Deps")
        return "No file loaded in Deps"

    text = ctx.deps.text
    offset = 0  # TextResource doesn't have an offset; use 0
    slice_length = ctx.deps.length
    
    # Try exact match first
    lower_text = text.lower()
    lower_start_str = start_str.lower()
    start_idx = lower_text.find(lower_start_str)
    if start_idx != -1:
        start_end_idx = start_idx + len(start_str)
        tail = text[start_end_idx:]
        lower_tail = tail.lower()
        lower_end_str = end_str.lower()
        rel_end_idx = lower_tail.find(lower_end_str)
        if rel_end_idx != -1:
            abs_end_idx = start_end_idx + rel_end_idx + len(end_str)
            # log.debug(
            #     f"Exact match found for '{start_str}...{end_str}' at {start_idx}, end at {abs_end_idx}"
            # )
            return (offset + start_idx, offset + abs_end_idx)

    # Fall back to fuzzy search
    start_match, start_idx, start_end_idx = await fuzzy_search_early_cancel(
        start_str, text, threshold
    )
    if start_idx < 0:
        #log.debug(f"Tried to start pattern: '{start_str}' - but didn't find a match")
        return f"Start string not found: '{start_str}'"

    tail = text[start_end_idx:]
    end_match, rel_end_idx, rel_end_idx_end = await fuzzy_search_early_cancel(
        end_str, tail, threshold
    )
    if rel_end_idx < 0:
        #log.debug(f"Tried to end pattern: '{end_str}' - returning default span")
        return (offset + start_idx, offset + slice_length)

    abs_end_idx = start_end_idx + rel_end_idx_end
    # log.debug(
    #     f"Fuzzy match found for '{start_str}...{end_str}' at {start_idx}, end at {abs_end_idx}"
    # )
    start,end=offset + start_idx, offset + abs_end_idx
    position=float(end) / float(len(text))
    text_slice=TextSlice(url=ctx.deps.url,text=text[start:end],doc_position=position)
    log.debug(f"found: {text_slice}")
    return text_slice



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
    if not isinstance(api_url, str):
        # must be OAI embeddings
        emb_r = await api_url.create(input=chunks, model=model, dimensions=vector_dim)
        return [vec.embedding for vec in emb_r.data]
    
    # Check if it's an Azure OpenAI endpoint
    if "openai.azure.com" in api_url or "cognitiveservices.azure.com" in api_url:
        # Azure OpenAI API format
        headers = {
            "Content-Type": "application/json",
            "api-key": toolkit.config.get("ckanext.chat.api_token", "your-api-token")
        }
        data = {
            "input": chunks,
            "model": model
        }
        if vector_dim:
            data["dimensions"] = vector_dim
            
        response = requests.post(
            api_url, headers=headers, data=json.dumps(data), verify=False
        )
        
        if response.status_code == 200:
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        else:
            return {"error": response.status_code, "message": response.text}
    else:
        # Custom embedding API format
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {"chunks": chunks, "model": model}
        response = requests.post(
            api_url, headers=headers, data=json.dumps(data), verify=False
        )

        if response.status_code == 200:
            return response.json()["embeddings"]
        else:
            return {"error": response.status_code, "message": response.text}


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
        query_vectors = await get_embedding(
            search_query,
            model=ctx.deps.embedding_model,
            api_url=ctx.deps.embeddings,
            vector_dim=ctx.deps.vector_dim,
        )
        num_results = 0
        hits = []
        filter_ids = []
        while num_results < limit:
            log.debug(f"{search_query} filtered by: {filter_ids}")
            search_res = ctx.deps.milvus_client.search(
                collection_name=ctx.deps.collection_name,
                data=query_vectors,
                search_params={"metric_type": "COSINE", "params": {"level": 1}},
                limit=6,
                filter_params={"ids": filter_ids} if filter_ids else None,
                filter="id not in {ids}" if filter_ids else None,
                output_fields=list(VectorMeta.__fields__.keys()),
                consistency_level="Bounded",
            )
            if search_res:
                for i in range(len(query_vectors)):
                    hit = [RagHit(**item) for item in search_res[i]]
                    hits += hit
                    filter_ids += list(set(hit.id for hit in hits))
                    # log.debug(hits)
                distinct_sources = list(set(hit.entity.source for hit in hits))
                num_results = len(distinct_sources)
                log.debug(
                    f"Rag search for:{search_query} with limit: {limit} returned {num_results} results."
                )
        return hits
        





@agent.tool
@research_agent.tool
async def literature_search(
    ctx: RunContext[Deps], search_question: str, num_results: int = 5
) -> list[str]:
    for attempt in range(3):
        try:
            r = await asyncio.wait_for(
                rag_agent.run(
                    f"Search for documents using this question:{search_question}. You must return {num_results} results",
                    deps=ctx.deps,
                    usage_limits=UsageLimits(request_limit=10),
                ),
                timeout=300
                )
            break
        except (asyncio.TimeoutError):
                log.warning(f"Timeout on literature_search attempt {attempt}, retrying...")
                attempt+=1
        except Exception as e:
            log.error(f"Unexpected error on literature_search attempt {attempt}: {str(e)}")
    else:
        raise RuntimeError("All literature_search retries timed out")
    #log.debug(r.data)
    return r.data.json()

@agent.tool_plain
async def literature_analyse(doc: TextResource, question: str, ssl_verify=True) -> list[str]:
    try:
        doc=await get_resource_file_contents(resource_url=str(doc.url),ssl_verify=ssl_verify)
    except Exception as e:
        return f"Error: {str(e)}"
    prompt = (
        f"Analyze the provided TextResource to determine whether it contains an answer to the question below.\n\n"
        f"**Question:** {question}\n\n"
        "Use an intelligent document navigation strategy:\n"
        "- Identify a Table of Contents or structural headings to guide your search.\n"
        "- Explore the full document as needed — do not rely only on the beginning.\n"
        "- Extract exact passages that support your answer, and cite them clearly.\n\n"
        "Return your response with inline citations and a concise conclusion."
    )   
    try:
        r = await asyncio.wait_for(
            doc_agent.run(
                prompt,
                deps=doc,
                usage_limits=UsageLimits(request_limit=50),
            ),
            timeout=600
        )
    except asyncio.TimeoutError:
        msg="Timeout on literature_analyse attempt, retrying..."
        log.error(msg)
        return msg
    except Exception as e:
        msg=f"Unexpected error on literature_analyse attempt: {str(e)}"
        log.error(msg)
        return msg
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
