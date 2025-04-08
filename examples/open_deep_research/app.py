from run import create_agent

from schema_agents.gradio_ui import GradioUI


agent = create_agent()

demo = GradioUI(agent)

if __name__ == "__main__":
    demo.launch()
