import pytest
import os
import time
from schema_agents.hyphaichat import HyphaAIChat
from schema_agents.tools.code_interpreter import CodeInterpreter
from pydantic import BaseModel
from simpleaichat.utils import fd

params = {"temperature": 0.0, "max_tokens": 1000}

class Author(BaseModel):
    name: str = fd("The name of the author.")
    email: str = fd("The email of the author.")

class Story(BaseModel):
    """Story."""
    name: str = fd("The name of the story.")
    year: int = fd("The year of the story.")
    text: str = fd("If selected, arguments to pass to the tool.")
    pages: int = fd("The number of pages in the story.")
    author: Author = fd("The author of the story.")

def write_file(fancy_name: str, content: str):
    """Write content to file."""
    print("=======> Writing to file: ", fancy_name)
    with open(fancy_name, "w") as f:
        f.write(content)
    # print("Error occurred while writing file: " + file_name)


def read_file(file_name: str):
    """Read content from file."""
    print("=======> Reading from file: ", file_name)
    with open(file_name, "r") as f:
        content = f.read()
    
    return content

class TestCodeInterpreter:
    @pytest.fixture(autouse=True)
    def init_code_interpreter(self):
        self.work_dir_root = "./.data"
        os.makedirs(self.work_dir_root, exist_ok=True)
        self.code_interpreter = CodeInterpreter(work_dir_root=self.work_dir_root)
        self.bot = HyphaAIChat(params=params, console=True, system="""You are Story-GPT, an AI designed to autonomously write stories.""")
        
    def test_functions(self):
        initial_message_length = len(self.bot.get_session().messages)
        response = self.bot("Write a short story in 100 words about yourself and save it into a file.", functions=[write_file, read_file])
        assert response.get('function') == 'write_file'
        assert len(self.bot.get_session().messages) == initial_message_length + 3
        
        response = self.bot("Read the story and generate a short title for it.", functions=[write_file, read_file])
        assert response.get('function') == 'read_file'
        assert len(self.bot.get_session().messages) == initial_message_length + 6

        response = self.bot("Execute Python script to count the #words in the story file", functions=[write_file, read_file, self.code_interpreter.execute_code])
        assert response.get('function') == 'execute_code'
        assert len(self.bot.get_session().messages) == initial_message_length + 9
        
        response = self.bot("Tell me a word which is interesting to learn", functions=[write_file, read_file])
        assert response.get('function') == None
        assert len(self.bot.get_session().messages) == initial_message_length + 11
        
        
    def test_io_schema(self):
        initial_message_length = len(self.bot.get_session().messages)
        response = self.bot("write a short story in 100 words about yourself and save it into a file named 'my_story.txt'.", functions=[write_file, read_file], output_schema=Story)
        assert response.get('function') == 'write_file'
        story = response.get('response')
        Story.model_validate(story)
        # If output_schema is specified, the length of the messages won't increase
        assert len(self.bot.get_session().messages) == initial_message_length + 3
        
