import os
import asyncio
import importlib
import sys
from schema_agents.schema import Message
import argparse
import logging

class DualLogger:
    def __init__(self, filepath, mode = 'w'):
        self.terminal = sys.stdout
        self.log = open(filepath, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    
async def run_team(module_name, user_request):
    module = importlib.import_module(module_name)
    team = module.make_team()
    responses = await team.handle(Message(content=user_request, role="User"))
    print(responses)

def setup_logging(log_filepath):
    sys.stdout = DualLogger(log_filepath)
    sys.stderr = sys.stdout

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filepath, 'a'), logging.StreamHandler(sys.stdout)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a team')
    parser.add_argument('module', type=str, help='The module to run')
    parser.add_argument("user_request", type=str, help="The user request")
    args = parser.parse_args()

    log_dir = "team_runner_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.module}.log")
    setup_logging(log_file)
    loop = asyncio.get_event_loop()
    loop.create_task(run_team(args.module, args.user_request))
    loop.run_forever()
