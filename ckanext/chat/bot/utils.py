from loguru import logger
log = logger.bind(module=__name__)

import tiktoken
import re, regex
from ckan.lib.lazyjson import LazyJSONObject
from ckan.model.package import Package
from ckan.model.resource import Resource
from pydantic import (BaseModel, ConfigDict, HttpUrl, ValidationError,
                      computed_field, root_validator, validator)
from typing import Any, Dict, Optional, List, Tuple
import ckan.plugins.toolkit as toolkit
import asyncio
import time

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

# --------------------- CKAN Actions and URL Helpers ---------------------

class FuncSignature(BaseModel):
    doc: Any

CKAN_ACTIONS: Dict[str, FuncSignature] = {}

def get_ckan_action(action: str = "") -> FuncSignature:
    global CKAN_ACTIONS
    if not CKAN_ACTIONS:
        from ckan.logic import _actions
        from ckan.logic.action.get import help_show
        actions=[key for key in _actions.keys() if not "_update" in key]
        for item in actions:
            doc = help_show({}, {"name": item})
            CKAN_ACTIONS[item] = FuncSignature(doc=doc).model_dump()
    if action in CKAN_ACTIONS.keys():
        return CKAN_ACTIONS[action]
    else:
        return CKAN_ACTIONS
    

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


CKAN_ROUTES: Dict[str, RouteModel] = {}



def find_route_by_endpoint(endpoint: str) -> Optional[RouteModel]:
    if endpoint in CKAN_ROUTES.keys():
        return CKAN_ROUTES[endpoint]
    return None


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
    
def get_ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN if the argument endpoint is None or empty it wil return a list of endpoints. If set to an endpoint it will return the RouteModel containing arguements and the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. If set returns the details of the endpoint. Defaults to "".

    Returns:
        RouteModel: All details on the Route
    """
    global CKAN_ROUTES
    if not CKAN_ROUTES:
        from ckanext.chat.views import global_ckan_app

        for rule in global_ckan_app.url_map.iter_rules():
            if not rule.rule.startswith("/_debug_toolbar"):
                route = RouteModel(
                    endpoint=rule.endpoint,
                    rule=rule.rule,
                    methods=sorted(list(rule.methods)),
                )
                CKAN_ROUTES[rule.endpoint] = route
    if endpoint and endpoint in CKAN_ROUTES.keys():
        return CKAN_ROUTES[endpoint].json()
    else:
        endpoints = [str(key) for key in CKAN_ROUTES.keys()]
        return f"route endpoint not found. List of endpoints: {endpoints}"

# --- functions for pattern matching

def try_match(pat: str, text: str, max_err: int):
        if pat and text:
            fuzzy_pat = f"({pat})" + f"{{e<={max_err}}}"
            if len(text)<=len(pat):
                raise ValueError(f"length of 'text': {text} is smaller then pattern length: {pat}.")
            return regex.search(
                fuzzy_pat,
                text,
                flags=regex.BESTMATCH | regex.IGNORECASE | regex.DOTALL
            ), fuzzy_pat
        else:
            raise ValueError("The 'pat' and 'text' parameters must be a non-empty strings.")
        
def _fuzzy_search_sync(pattern: str, text: str, threshold: float = 0.8) -> Tuple[Optional[str], int, int]:
    max_err = max(1, int((1 - threshold) * len(pattern)))
    try:
        match, fuzzy_pat = try_match(pat=pattern,text=text,max_err=max_err)
    except regex.error as e:
        print(f"Initial regex failed for pattern '{pattern}': {e}")
        match = None
    
    if not match:
        escaped = regex.escape(pattern)
        try:
            match, fuzzy_pat = try_match(pat=pattern,text=text,max_err=max_err)
        except regex.error as e:
            print(f"Escaped regex also failed for pattern '{escaped}': {e}")
            return "", -1, -1
    
    if not match:
        return "", -1, -1
    
    return match.group(1), match.start(1), match.end(1)

def split_text_into_chunks(text, chunk_size, overlap):
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append((chunk, i))
    return chunks

async def fuzzy_search_early_cancel(pattern: str, text: str, threshold: float = 0.8) -> Tuple[Optional[str], int, int]:
    # Überprüfe, ob der Pattern und der Text gültig sind
    if not pattern or not isinstance(pattern, str):
        raise ValueError("The 'pattern' parameter must be a non-empty string.")
    
    if not text or not isinstance(text, str):
        raise ValueError("The 'text' parameter must be a non-empty string.")
    
    start_time = time.perf_counter()
    chunk_size = 10000
    overlap = 1000
    
    if text and len(text) <= chunk_size:
        result = _fuzzy_search_sync(pattern, text, threshold)
        duration = time.perf_counter() - start_time
        log.debug(f"Tried to match: '{pattern}' - found: {result[0] if result[1] >= 0 else 'no match'} - took {duration:.4f} seconds")
        return result
    
    step = chunk_size - overlap
    tasks = []
    chunks=split_text_into_chunks(text,chunk_size,overlap)
    # Erstelle die Tasks direkt als awaitables
    tasks = [asyncio.to_thread(_fuzzy_search_sync, pattern, chunk[0], threshold) for chunk in chunks]
    for completed_task in asyncio.as_completed(tasks):
        try:
            match, start, end = await completed_task
            if completed_task not in tasks:
                continue
            if start >= 0:
                log.debug(f"Completed task: {completed_task}")
                # Finde den Index des abgeschlossenen Tasks in der ursprünglichen Zuordnung
                chunk_idx = tasks.index(completed_task)
                abs_start = chunks[chunk_idx][1] + start
                abs_end = chunks[chunk_idx][1] + end
                duration = time.perf_counter() - start_time
                log.debug(f"Tried to match: '{pattern}' - found: {match} at {abs_start}-{abs_end} - took {duration:.4f} seconds")

                # Cancel all other tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return match, abs_start, abs_end

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Error while processing fuzzy_search_early_cancel task:  {str(e)}")

    duration = time.perf_counter() - start_time
    log.debug(f"Tried to match: '{pattern}' - no match found - took {duration:.4f} seconds")
    return "", -1, -1
