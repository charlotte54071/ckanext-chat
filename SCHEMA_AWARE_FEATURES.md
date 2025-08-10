# Schema-Aware Features for CKAN Chat Extension

This document describes the schema-aware enhancements made to the `ckanext-chat` extension to support multiple CKAN dataset schemas.

## Overview

The CKAN Chat extension has been enhanced to work intelligently with multiple dataset schemas, providing context-aware responses and better search capabilities across different schema types.

## Supported Schema Types

The extension now supports the following schema types:
- `dataset` - Standard CKAN datasets
- `device` - IoT devices and hardware components
- `digitaltwin` - Digital twin representations
- `geoobject` - Geographic objects and spatial data
- `method` - Methodologies and procedures
- `onlineapplication` - Web applications
- `onlineservice` - Online services
- `project` - Project datasets
- `software` - Software packages and tools

## Configuration

### New Configuration Options

Add the following to your CKAN configuration file:

```ini
# Enable schema-aware functionality
ckanext.chat.schema_aware = true

# List of supported schemas (comma-separated)
ckanext.chat.supported_schemas = dataset,device,digitaltwin,geoobject,method,onlineapplication,onlineservice,project,software

# Enable schema context enhancement for datasets
ckanext.chat.schema_context_enhancement = true
```

## Features

### 1. Schema-Aware Search

The chat agent can now:
- Filter searches by schema type using `fq=type:schema_name`
- Provide schema-specific search suggestions
- Understand schema-specific field meanings

Example queries:
- "Show me all device datasets"
- "Find digital twin projects related to manufacturing"
- "Search for software packages in the energy domain"

### 2. Enhanced Dataset Context

Datasets are now enriched with:
- Schema type information
- Schema-specific field descriptions
- Context about the schema's purpose
- Links to schema documentation

### 3. Schema-Specific Actions

The agent suggests relevant actions based on dataset schema type:
- Device datasets: status checks, monitoring
- Digital twins: simulation runs, model updates
- Geographic objects: spatial queries, mapping
- Software: version checks, dependency analysis

### 4. Intelligent Field Mapping

The extension provides:
- Field name to human-readable label mapping
- Schema-specific field descriptions
- Context-aware field suggestions

## Helper Functions

### Template Helpers

- `get_supported_schemas()` - Returns list of configured schemas
- `is_schema_aware()` - Checks if schema-aware mode is enabled
- `get_schema_context_for_dataset(dataset_type)` - Gets schema context for a dataset
- `get_schema_field_info(schema_type, field_name)` - Gets field information
- `enhance_dataset_with_schema_context(dataset)` - Enriches dataset with schema info

### Utility Functions

- `get_schema_aware_search_context()` - Gets complete schema context
- `enhance_search_query_with_schema_context(query, schema_type)` - Enhances search queries
- `filter_datasets_by_schema(datasets, schema_types)` - Filters datasets by schema
- `suggest_schema_based_actions(dataset_type)` - Suggests relevant actions

## Agent Tools

### New Agent Tool

- `get_schema_context()` - Available to all agents to retrieve schema information

## Usage Examples

### Basic Schema-Aware Search

```python
# Search for device datasets
ckan_run("package_search", {"q": "sensors", "fq": "type:device"})

# Get schema context
get_schema_context()
```

### Enhanced Dataset Information

When viewing datasets, the chat will now show:
- Schema type (e.g., "Device Dataset")
- Schema-specific fields and their meanings
- Relevant actions for that schema type

### Schema-Specific Queries

The chat agent can now handle queries like:
- "What device datasets are available for temperature monitoring?"
- "Show me digital twin models for building automation"
- "Find software packages related to data processing"

## Implementation Details

### Modified Files

1. **plugin.py** - Added schema-aware configuration options
2. **helpers.py** - Added schema-aware helper functions
3. **bot/utils.py** - Enhanced DynamicDataset class and added utility functions
4. **bot/agent.py** - Updated prompts and added schema context tool

### Key Enhancements

1. **Dynamic Schema Loading**: Schemas are loaded dynamically from CKAN's scheming extension
2. **Context Enhancement**: Datasets are automatically enriched with schema context
3. **Intelligent Search**: Search queries are enhanced with schema-specific context
4. **Action Suggestions**: Relevant actions are suggested based on schema type

## Benefits

1. **Better User Experience**: Users get more relevant and context-aware responses
2. **Improved Search**: Schema-aware filtering provides more precise results
3. **Enhanced Understanding**: The chat agent understands the purpose and structure of different data types
4. **Flexible Configuration**: Easy to add new schema types or modify existing ones

## Future Enhancements

Potential future improvements include:
- Schema-specific validation rules
- Custom prompts per schema type
- Schema-aware data visualization suggestions
- Integration with schema-specific workflows