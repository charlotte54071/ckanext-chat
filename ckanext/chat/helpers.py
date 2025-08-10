from typing import Any, Dict, List

import ckan.plugins.toolkit as toolkit
import yaml
import os


def service_available() -> bool:
    """Check if the chat service is available"""
    completion_url = toolkit.config.get("ckanext.chat.completion_url")
    api_token = toolkit.config.get("ckanext.chat.api_token")
    return bool(completion_url and api_token)


def get_supported_schemas() -> List[str]:
    """Get list of supported schemas from configuration"""
    schemas_config = toolkit.config.get("ckanext.chat.supported_schemas", "")
    if schemas_config:
        return [schema.strip() for schema in schemas_config.split(",")]
    return []


def is_schema_aware() -> bool:
    """Check if schema-aware mode is enabled"""
    return toolkit.asbool(toolkit.config.get("ckanext.chat.schema_aware", True))


def get_schema_context_for_dataset(dataset_type: str) -> Dict[str, Any]:
    """Get schema context information for a specific dataset type"""
    if not is_schema_aware():
        return {}
    
    try:
        # Try to get schema information from scheming extension
        schema_info = toolkit.get_action('scheming_dataset_schema_show')({}, {'type': dataset_type})
        return {
            'schema_type': dataset_type,
            'schema_fields': schema_info.get('dataset_fields', []),
            'resource_fields': schema_info.get('resource_fields', []),
            'about': schema_info.get('about', ''),
            'about_url': schema_info.get('about_url', '')
        }
    except Exception:
        # Fallback if scheming is not available or schema not found
        return {'schema_type': dataset_type}


def get_schema_field_info(dataset_type: str, field_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific field in a schema"""
    schema_context = get_schema_context_for_dataset(dataset_type)
    
    for field in schema_context.get('schema_fields', []):
        if field.get('field_name') == field_name:
            return field
    
    for field in schema_context.get('resource_fields', []):
        if field.get('field_name') == field_name:
            return field
    
    return {}


def get_all_schema_contexts() -> Dict[str, Dict[str, Any]]:
    """Get schema contexts for all supported schemas"""
    if not is_schema_aware():
        return {}
    
    contexts = {}
    supported_schemas = get_supported_schemas()
    
    for schema_type in supported_schemas:
        contexts[schema_type] = get_schema_context_for_dataset(schema_type)
    
    return contexts


def enhance_dataset_with_schema_context(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance dataset data with schema context information"""
    if not toolkit.asbool(toolkit.config.get("ckanext.chat.schema_context_enhancement", True)):
        return dataset
    
    dataset_type = dataset.get('type', 'dataset')
    schema_context = get_schema_context_for_dataset(dataset_type)
    
    if schema_context:
        dataset['_schema_context'] = schema_context
    
    return dataset


def get_helpers():
    return {
        "service_available": service_available,
        "get_supported_schemas": get_supported_schemas,
        "is_schema_aware": is_schema_aware,
        "get_schema_context_for_dataset": get_schema_context_for_dataset,
        "get_schema_field_info": get_schema_field_info,
        "get_all_schema_contexts": get_all_schema_contexts,
        "enhance_dataset_with_schema_context": enhance_dataset_with_schema_context,
    }
