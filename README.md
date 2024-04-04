## Schema Agents

A schema-based LLM framework for building multi-agent collaborative systems.

## Development
Create a conda environment and install dependencies:
```
conda create -n schema-agents python=3.10
conda activate schema-agents
```

```
conda install faiss-cpu -c conda-forge
pip install -r requirements_test.txt
pip install -e .
```

### Ollama Support

Start the Ollama container using docker:


```
# expose all gpus
docker run --gpus=all -d -v ollama:/root/.ollama -p 11434:11434 --rm --name ollama ollama/ollama

# expose specific gpus
# docker run --gpus device=2 -d -v ollama:/root/.ollama -p 11434:11434 --rm --name ollama ollama/ollama
```
