import asyncio
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import ckan.plugins.toolkit as toolkit
import regex
import tiktoken
from ckan.lib.lazyjson import LazyJSONObject
from ckan.model.package import Package
from ckan.model.resource import Resource
from loguru import logger
from pydantic import BaseModel, ValidationError, computed_field, root_validator

log = logger.bind(module=__name__)

# --------------------- Dynamic Models Initialization ---------------------

dynamic_models_initialized = False


def init_dynamic_models():
    """Initialize dynamic models for CKAN datasets and resources."""
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
    type: Optional[str] = "dataset"  # Schema type
    view_url: Optional[str] = None
    schema_context: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def calculate_computed_field(cls, values):
        route = find_route_by_endpoint("dataset.read")
        ckan_url = toolkit.config.get("ckan.site_url")
        if route and ckan_url:
            values["view_url"] = str(
                route.build_url(base_url=ckan_url, fill={"id": values.get("id")})
            )
        
        # Add schema context if schema-aware mode is enabled
        dataset_type = values.get("type", "dataset")
        if toolkit.asbool(toolkit.config.get("ckanext.chat.schema_aware", True)):
            try:
                schema_info = toolkit.get_action('scheming_dataset_schema_show')(
                    {}, {'type': dataset_type}
                )
                values["schema_context"] = {
                    'schema_type': dataset_type,
                    'schema_fields': schema_info.get('dataset_fields', []),
                    'resource_fields': schema_info.get('resource_fields', []),
                    'about': schema_info.get('about', ''),
                    'about_url': schema_info.get('about_url', '')
                }
            except Exception:
                values["schema_context"] = {'schema_type': dataset_type}
        
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
    
    def get_schema_field_info(self, field_name: str) -> Dict[str, Any]:
        """Get information about a specific field from the schema context"""
        if not self.schema_context:
            return {}
        
        for field in self.schema_context.get('schema_fields', []):
            if field.get('field_name') == field_name:
                return field
        
        for field in self.schema_context.get('resource_fields', []):
            if field.get('field_name') == field_name:
                return field
        
        return {}
    
    def get_schema_type_description(self) -> str:
        """Get a human-readable description of the schema type"""
        if not self.schema_context:
            return f"Dataset of type: {self.type}"
        
        about = self.schema_context.get('about', '')
        if about:
            return f"{self.type.title()} Schema: {about}"
        
        return f"Dataset using {self.type} schema"


class DynamicResource(BaseModel):
    id: str  # CKAN resource id
    view_url: Optional[str] = None

    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def calculate_computed_field(cls, values):
        route = find_route_by_endpoint("resource.read")
        ckan_url = toolkit.config.get("ckan.site_url")
        if route and ckan_url:
            values["view_url"] = str(route.build_url(fill={"id": values.get("id")}))
        return values

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

        actions = [key for key in _actions.keys() if "_update" not in key]
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
    """Truncate output by token count with optional offset."""
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(output)

    if len(tokens) > token_limit:
        # Skip the specified number of tokens
        truncated_tokens = tokens[offset:offset + token_limit]
        output = encoding.decode(truncated_tokens)
        # If last page of tokens, add a mark
        if len(truncated_tokens) < token_limit:
            output += "\n\n**End of Output**"

    return output


def truncate_value(value, max_length):
    """Truncate string or list values to specified maximum length."""
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
    """Truncate nested data structures by depth to prevent excessive nesting."""
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
    """Recursively unpack LazyJSONObject instances to regular Python objects."""
    if isinstance(obj, dict):
        return {key: unpack_lazy_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [unpack_lazy_json(item) for item in obj]
    elif isinstance(obj, LazyJSONObject):
        return obj.encoded_json
    return obj


def process_entity(data: Any, depth: int = 0, max_depth: int = 4) -> Any:
    """Process and validate CKAN entities with depth control."""
    if depth > max_depth:
        log.warning("Max recursion depth reached")
        return None

    if isinstance(data, dict):
        data = unpack_lazy_json(data)
        if "resources" in data:
            try:
                dataset_dict = DynamicDataset(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                dataset_dict = {k: v for k, v in dataset_dict.items() if bool(v)}
                return truncate_by_depth(dataset_dict, max_depth - depth)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicDataset: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicDataset failed: {ex}")
        elif "package_id" in data or "url" in data:
            try:
                resource_dict = DynamicResource(**data).model_dump(
                    exclude_unset=True, exclude_defaults=False, exclude_none=True
                )
                resource_dict = {k: v for k, v in resource_dict.items() if bool(v)}
                return truncate_by_depth(resource_dict, max_depth - depth)
            except ValidationError as validation_error:
                log.warning(
                    f"Validation error converting to DynamicResource: {validation_error.json()}"
                )
            except Exception as ex:
                log.warning(f"Conversion to DynamicResource failed: {ex}")
        else:
            new_dict = {}
            for key, value in data.items():
                processed_value = process_entity(value, depth + 1, max_depth)
                if processed_value not in ([], {}, "", None):
                    new_dict[key] = processed_value
            return new_dict

    elif isinstance(data, list):
        new_list = []
        for item in data:
            processed_item = process_entity(item, depth + 1, max_depth)
            if processed_item not in ([], {}, "", None):
                new_list.append(processed_item)
        return new_list
    else:
        return data

def get_ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN.
    
    If the argument endpoint is None or empty it will return a list of endpoints.
    If set to an endpoint it will return the RouteModel containing arguments and 
    the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. 
            If set returns the details of the endpoint. Defaults to "".

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
        return CKAN_ROUTES[endpoint]
    else:
        endpoints = [str(key) for key in CKAN_ROUTES.keys()]
        return f"route endpoint not found. List of endpoints: {endpoints}"


# --- functions for pattern matching


def try_match(pat: str, text: str, max_err: int):
    """Attempt fuzzy pattern matching with specified error tolerance."""
    if pat and text:
        fuzzy_pat = f"({pat})" + f"{{e<={max_err}}}"
        if len(text) <= len(pat):
            raise ValueError(
                f"length of 'text': {text} is smaller than pattern length: {pat}."
            )
        return (
            regex.search(
                fuzzy_pat, text, flags=regex.BESTMATCH | regex.IGNORECASE | regex.DOTALL
            ),
            fuzzy_pat,
        )
    else:
        raise ValueError("The 'pat' and 'text' parameters must be non-empty strings.")


def _fuzzy_search_sync(
    pattern: str, text: str, threshold: float = 0.8
) -> Tuple[Optional[str], int, int]:
    """Synchronous fuzzy search implementation."""
    max_err = max(1, int((1 - threshold) * len(pattern)))
    try:
        match, fuzzy_pat = try_match(pat=pattern, text=text, max_err=max_err)
    except regex.error as e:
        print(f"Initial regex failed for pattern '{pattern}': {e}")
        match = None

    if not match:
        escaped = regex.escape(pattern)
        try:
            match, fuzzy_pat = try_match(pat=pattern, text=text, max_err=max_err)
        except regex.error as e:
            print(f"Escaped regex also failed for pattern '{escaped}': {e}")
            return "", -1, -1

    if not match:
        return "", -1, -1

    return match.group(1), match.start(1), match.end(1)


def split_text_into_chunks(text, chunk_size, overlap):
    """Split text into overlapping chunks for parallel processing."""
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 0:
            chunks.append((chunk, i))
    return chunks


async def fuzzy_search_early_cancel(
    pattern: str, text: str, threshold: float = 0.8
) -> Tuple[Optional[str], int, int]:
    """Perform fuzzy search with early cancellation for better performance."""
    # Validate pattern and text parameters
    if not pattern or not isinstance(pattern, str):
        raise ValueError("The 'pattern' parameter must be a non-empty string.")

    if not text or not isinstance(text, str):
        raise ValueError("The 'text' parameter must be a non-empty string.")

    start_time = time.perf_counter()
    chunk_size = 10000
    overlap = 1000

    if text and len(text) <= chunk_size:
        result = _fuzzy_search_sync(pattern, text, threshold)
        return result

    tasks = []
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    # Create tasks directly as awaitables
    tasks = [
        asyncio.to_thread(_fuzzy_search_sync, pattern, chunk[0], threshold)
        for chunk in chunks
    ]
    for completed_task in asyncio.as_completed(tasks):
        try:
            match, start, end = await completed_task
            if completed_task not in tasks:
                continue
            if start >= 0:
                # Find the index of the completed task in the original mapping
                chunk_idx = tasks.index(completed_task)
                abs_start = chunks[chunk_idx][1] + start
                abs_end = chunks[chunk_idx][1] + end

                # Cancel all other tasks
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return match, abs_start, abs_end

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(
                f"Error while processing fuzzy_search_early_cancel task: {str(e)}"
            )

    return "", -1, -1


# --------------------- Schema-Aware Utility Functions ---------------------

def get_schema_aware_search_context() -> Dict[str, Any]:
    """Get context information for schema-aware searches"""
    if not toolkit.asbool(toolkit.config.get("ckanext.chat.schema_aware", True)):
        return {
            "schema_aware_enabled": False,
            "supported_schemas": [],
            "schema_contexts": {}
            }
    
    schemas_config = toolkit.config.get("ckanext.chat.supported_schemas", "")
    supported_schemas = [schema.strip() for schema in schemas_config.split(",") if schema.strip()]
    
    schema_contexts = {}
    for schema_type in supported_schemas:
        try:
            schema_info = toolkit.get_action('scheming_dataset_schema_show')({}, {'type': schema_type})
            ds_fields = schema_info.get('dataset_fields', []) or []
            res_fields = schema_info.get('resource_fields', []) or []
            field_descriptions = {
                f.get('field_name'): (f.get('label') or f.get('help_text') or f.get('field_name', ''))
                for f in ds_fields + res_fields
                if f.get('field_name')
            }

            # Keep complete fields
            schema_contexts[schema_type] = {
                'about': schema_info.get('about', ''),
                'dataset_fields': ds_fields,
                'resource_fields': res_fields,
                'field_descriptions': field_descriptions
            }
        except Exception as e:
            log.warning(f"Could not load schema info for {schema_type}: {e}")
            schema_contexts[schema_type] = {
                'about': f'Schema type: {schema_type}',
                'dataset_fields': [],
                'resource_fields': [],
                'field_descriptions': {}
            }
    
    return {
        'supported_schemas': supported_schemas,
        'schema_contexts': schema_contexts,
        'schema_aware_enabled': True
    }


def enhance_search_query_with_schema_context(query: str, schema_type: str) -> str:
    """Enhance search query with schema-specific context."""
    try:
        schema_context = get_schema_aware_search_context(schema_type)
        if schema_context:
            # Add schema-specific field mappings to the query
            field_mappings = schema_context.get("field_mappings", {})
            enhanced_terms = []
            
            for term in query.split():
                # Check if term matches any schema field
                for field_name, field_info in field_mappings.items():
                    if (term.lower() in field_name.lower() or 
                        term.lower() in field_info.get("label", "").lower()):
                        enhanced_terms.append(f"{field_name}:{term}")
                        break
                else:
                    enhanced_terms.append(term)
            
            return " ".join(enhanced_terms)
    except Exception as e:
        logger.warning(f"Failed to enhance query with schema context: {e}")
    
    return query


def filter_datasets_by_schema(datasets: List[Dict], schema_type: str) -> List[Dict]:
    """Filter datasets by schema type."""
    filtered = []
    for dataset in datasets:
        dataset_schema = dataset.get("type", "dataset")
        if dataset_schema == schema_type:
            filtered.append(dataset)
    return filtered


def get_schema_specific_field_mappings(schema_type: str) -> Dict[str, Any]:
    """Get field mappings for a specific schema type."""
    try:
        schema_context = get_schema_aware_search_context(schema_type)
        return schema_context.get("field_mappings", {}) if schema_context else {}
    except Exception as e:
        logger.warning(f"Failed to get field mappings for schema {schema_type}: {e}")
        return {}


def suggest_schema_based_actions(query: str, schema_type: str) -> List[str]:
    """Suggest actions based on schema type and query."""
    suggestions = []
    
    try:
        schema_context = get_schema_aware_search_context(schema_type)
        if schema_context:
            field_mappings = schema_context.get("field_mappings", {})
            
            # Suggest field-specific searches
            for field_name, field_info in field_mappings.items():
                if any(term.lower() in field_name.lower() for term in query.split()):
                    suggestions.append(f"Search in {field_info.get('label', field_name)} field")
            
            # Suggest schema-specific actions
            if schema_type == "device":
                suggestions.extend([
                    "Filter by device type",
                    "Search by manufacturer",
                    "Find devices by location"
                ])
            elif schema_type == "dataset":
                suggestions.extend([
                    "Filter by data format",
                    "Search by organization",
                    "Find recent datasets"
                ])
    except Exception as e:
        logger.warning(f"Failed to generate schema-based suggestions: {e}")
    
    return suggestions
