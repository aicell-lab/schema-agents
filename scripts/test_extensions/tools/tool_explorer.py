import yaml
import os
import asyncio
from schema_agents import schema_tool, Role
from pydantic import Field
import ast
from typing import List, Set
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

class SchemaToolVisitor(ast.NodeVisitor):
    def __init__(self):
        self.schema_tools = []
    
    def visit_FunctionDef(self, node):
        self.check_decorators(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.check_decorators(node)
    
    def check_decorators(self, node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "schema_tool":
                self.schema_tools.append(node)
        # Continue to visit the children of the node
        self.generic_visit(node)

def get_docstring_from_async_function(async_function_node):
    """
    Extracts the docstring from an ast.AsyncFunctionDef node if it exists.
    
    :param async_function_node: An instance of ast.AsyncFunctionDef
    :return: The docstring as a string if found, otherwise None
    """
    # Check if the async function has a body and the first element of the body is an expression
    if async_function_node.body and isinstance(async_function_node.body[0], ast.Expr):
        # The first element of the body could be a docstring (as an expression)
        first_element = async_function_node.body[0]
        # Depending on the Python version, the docstring can be in ast.Str (Python <3.8) or ast.Constant (Python >=3.8)
        if hasattr(first_element, 'value') and isinstance(first_element.value, (ast.Str, ast.Constant)):
            return first_element.value.s  # Extract the string value
    return "No docstring provided"  # No docstring found

async def list_schema_tools(top_directory : str, 
                            exclude_funcs : Set[str] = {}, 
                            exclude_dirs : Set[str] = {}, 
                            exclude_files : Set[str] = {}) -> List[str]:
    """List all schema tools in the given directory"""
    schema_tools = {}
    path = Path(top_directory)

    # Recursively search for Python files in the directory
    for py_file in path.rglob("*.py"):

        # For checking if the file is in an excluded directory or is an excluded file
        relative_py_file = py_file.relative_to(path)
        if any(str(relative_py_file).startswith(str(ex_dir) + "/") for ex_dir in exclude_dirs) or \
           any(str(relative_py_file) == str(ex_file) for ex_file in exclude_files):
            continue
        
        print(f"Checking {py_file}")
        usage_yaml = py_file.with_suffix('.yaml')
        if not usage_yaml.exists():
            usage_strings = {}
        else:
            with open(usage_yaml, 'r', encoding='utf-8') as yaml_file:
                usage_strings = yaml.safe_load(yaml_file)
        with open(py_file, 'r', encoding = 'utf-8') as file:
            source = file.read()
            try:
                tree = ast.parse(source)
                visitor = SchemaToolVisitor()
                visitor.visit(tree)
                d = {t.name : {'ast' : t,
                               'usage' : usage_strings.get(t.name, {'usage' : "No usage string provided"})['usage'],
                               'docstring' : get_docstring_from_async_function(t)} 
                    for t in visitor.schema_tools}
                # schema_tools.extend(visitor.schema_tools)
                schema_tools.update(d)
            except Exception as e:
                print(f"Error parsing {py_file}: {e}")
    
    # schema_tools = [tool for tool in schema_tools if tool.name not in exclude_funcs]
    # schema_tools = {k : v for k, v in schema_tools.items() if k not in exclude_funcs}
    return schema_tools

async def create_tool_db(tool_dir,
                         save_path = "tool_index",
                         exclude_funcs = {}, 
                         exclude_dirs = {}, 
                         exclude_files = {}):
    """Create a database of available tools for the agents to use"""
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50,
                        length_function=len,
                        is_separator_regex=False,
                    )
    schema_tools = await list_schema_tools(top_directory = tool_dir,
                                           exclude_funcs = exclude_funcs,
                                           exclude_dirs = exclude_dirs,
                                           exclude_files = exclude_files)
    tool_documents = [Document(page_content = t['docstring'], metadata = {'name' : name, 'usage' : t['usage'], 'ast' : t['ast']}) for name, t in schema_tools.items()]
    db = FAISS.from_documents(tool_documents, OpenAIEmbeddings())
    db.save_local(save_path)
    return db

async def search_tools(query : str, db_path : str) -> List[str]:
    """Search for tool to solve tasks given a tool database"""
    db = FAISS.load_local(db_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    results = db.similarity_search(query)
    return results

def fixed_db_tool_search(fixed_db_path : str):
    @schema_tool
    async def wrapper(query : str) -> List[str]:
        """Search for tool to solve tasks given a tool database"""
        return await search_tools(query, fixed_db_path)
    return wrapper

@schema_tool
async def test_tool(name : str = Field(description = "The name of the agent running this tool")) -> str:
    """A test tool for testing tool exploration"""
    return f"Hi, {name}. This is a tool test"
# Usage: test_tool(name="Agent Name")


async def main():
    top_dir = "/Users/gkreder/schema-agents/scripts/test_extensions/tools/"
    db = await create_tool_db(tool_dir = top_dir,
                              save_path="tool_index",
                               exclude_funcs = {"test_tool"},
                               exclude_dirs={},
                               exclude_files={"tool_explorer.py"})
    res = db.similarity_search("NCBI")
    # print('hello world')
    tool_search = fixed_db_tool_search(fixed_db_path = "tool_index")
    res = await tool_search("NCBI")
    print(res)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()
    


