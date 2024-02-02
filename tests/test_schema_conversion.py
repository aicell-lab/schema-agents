from schema_agents.utils.schema_conversion import get_service_openapi_schema, get_service_json_schema, get_service_function_schema
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pytest

class DocWithScore(BaseModel):
    """A document with an associated relevance score."""
    doc: str = Field(description="The document retrieved.")
    score: float = Field(description="The relevance score of the retrieved document.")
    metadata: Dict[str, Any] = Field(description="The document's metadata.")
    base_url: Optional[str] = Field(
        None,
        description="The documentation's base URL, which will be used to resolve the relative URLs in the retrieved document chunks when producing markdown links.",
    )

def process(doc: DocWithScore) -> str:
    """Process a document with an associated relevance score."""
    return "processed"

def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"

service_config = {
    "name": "Hello World",
    "id": "hello-world",
    "config": {
        "visibility": "public"
    },
    "process": process,
    "hello": {
        "world": hello
    }
}

expected_openapi_schema = {'openapi': '3.1.0', 'info': {'title': 'Hello World', 'version': 'v0.1.0'}, 'servers': [{'url': '/'}], 'paths': {'/process': {'post': {'description': 'Process a document with an associated relevance score.', 'operationId': 'process', 'requestBody': {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/process'}}}, 'required': False}, 'responses': {'200': {'description': 'Successful response', 'content': {'application/json': {'schema': {'type': 'string'}}}}}, 'deprecated': False}}, '/hello.world': {'post': {'description': 'Say hello to someone.', 'operationId': 'hello_world', 'requestBody': {'content': {'application/json': {'schema': {'$ref': '#/components/schemas/hello_world'}}}, 'required': False}, 'responses': {'200': {'description': 'Successful response', 'content': {'application/json': {'schema': {'type': 'string'}}}}}, 'deprecated': False}}}, 'components': {'schemas': {'hello_world': {'properties': {'name': {'type': 'string', 'title': 'Name'}}, 'type': 'object', 'required': ['name'], 'title': 'hello_world', 'description': 'Say hello to someone.'}, 'DocWithScore': {'properties': {'doc': {'type': 'string', 'title': 'Doc', 'description': 'The document retrieved.'}, 'score': {'type': 'number', 'title': 'Score', 'description': 'The relevance score of the retrieved document.'}, 'metadata': {'type': 'object', 'title': 'Metadata', 'description': "The document's metadata."}, 'base_url': {'type': 'string', 'title': 'Base Url', 'description': "The documentation's base URL, which will be used to resolve the relative URLs in the retrieved document chunks when producing markdown links."}}, 'type': 'object', 'required': ['doc', 'score', 'metadata'], 'title': 'DocWithScore', 'description': 'A document with an associated relevance score.'}, 'process': {'properties': {'doc': {'allOf': [{'$ref': '#/components/schemas/DocWithScore'}], 'title': 'Doc', 'description': 'A document with an associated relevance score.'}}, 'type': 'object', 'required': ['doc'], 'title': 'process', 'description': 'Process a document with an associated relevance score.'}}}}

# Define the pytest test function
@pytest.mark.parametrize("field, expected_value", [
    ('openapi', '3.1.0'),
    ('info', {'title': 'Hello World', 'version': 'v0.1.0'}),
    ('servers', [{'url': '/'}]),
    ('paths', expected_openapi_schema['paths']),
    ('components', expected_openapi_schema['components'])
])
def test_validate_openapi_schema(field, expected_value):
    schema = get_service_openapi_schema(service_config)
    assert schema[field] == expected_value


expected_json_schema = {'process': {'input_schema': {'title': 'process', 'description': 'Process a document with an associated relevance score.', 'type': 'object', 'properties': {'doc': {'title': 'Doc', 'description': 'A document with an associated relevance score.', 'allOf': [{'$ref': '#/definitions/DocWithScore'}]}}, 'required': ['doc'], 'definitions': {'DocWithScore': {'title': 'DocWithScore', 'description': 'A document with an associated relevance score.', 'type': 'object', 'properties': {'doc': {'title': 'Doc', 'description': 'The document retrieved.', 'type': 'string'}, 'score': {'title': 'Score', 'description': 'The relevance score of the retrieved document.', 'type': 'number'}, 'metadata': {'title': 'Metadata', 'description': "The document's metadata.", 'type': 'object'}, 'base_url': {'title': 'Base Url', 'description': "The documentation's base URL, which will be used to resolve the relative URLs in the retrieved document chunks when producing markdown links.", 'type': 'string'}}, 'required': ['doc', 'score', 'metadata']}}}, 'output_schema': {'type': 'string'}}, 'hello.world': {'input_schema': {'title': 'hello_world', 'description': 'Say hello to someone.', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}}, 'required': ['name']}, 'output_schema': {'type': 'string'}}}


@pytest.mark.parametrize("field, expected_value", [
    ('process', expected_json_schema['process']),
    ('hello.world', expected_json_schema['hello.world'])
])
def test_validate_json_schema(field, expected_value):
    schema = get_service_json_schema(service_config)
    assert schema[field] == expected_value
    

expected_function_schema = [{'type': 'function', 'function': {'name': 'process', 'description': 'Process a document with an associated relevance score.', 'parameters': {'type': 'object', 'properties': {'doc': {'allOf': [{'$ref': '#/definitions/DocWithScore'}]}}, 'required': ['doc'], 'definitions': {'DocWithScore': {'type': 'object', 'properties': {'doc': {'type': 'string'}, 'score': {'type': 'number'}, 'metadata': {'type': 'object'}, 'base_url': {'type': 'string'}}, 'required': ['doc', 'score', 'metadata']}}}}}, {'type': 'function', 'function': {'name': 'hello.world', 'description': 'Say hello to someone.', 'parameters': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}}}]

@pytest.mark.parametrize("field, expected_value", [
    (0, expected_function_schema[0]),
    (1, expected_function_schema[1])
])
def test_validate_function_schema(field, expected_value):
    schema = get_service_function_schema(service_config)
    assert schema[field] == expected_value