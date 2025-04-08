# docstyle-ignore
INSTALL_CONTENT = """
# Installation
! pip install schema_agents
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/schema_agents.git
"""

notebook_first_cells = [{"type": "code", "content": INSTALL_CONTENT}]
black_avoid_patterns = {
    "{processor_class}": "FakeProcessorClass",
    "{model_class}": "FakeModelClass",
    "{object_class}": "FakeObjectClass",
}
