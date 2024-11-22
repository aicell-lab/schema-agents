import sys
import os
import asyncio
import argparse
from pydantic import BaseModel, Field
from schema_agents.schema import Message
from typing import List, Dict
from xml.etree import ElementTree as ET
import importlib
import logging
import datetime

class DualLogger:
    def __init__(self, filepath, mode = 'w'):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    
    def silent_write(self, message):
        self.log.write(message)
        self.flush()

    def dump_test(self, team_name, user_input):
        self.log.write(f"Team Name: {team_name}\n")
        self.log.write(f"User Input: {user_input}\n")
        self.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(log_filepath):
    logger = DualLogger(log_filepath)
    sys.stdout = logger
    sys.stderr = sys.stdout

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_filepath, 'a'), logging.StreamHandler(sys.stdout)])
    return logger

async def run_test(team_name, user_input, logger = None):
    team_module = importlib.import_module("teams")
    team_class = getattr(team_module, team_name)
    team = team_class()
    if logger:
        logger.silent_write(f"\n\nRUN PARAMS\n")
        logger.dump_test(team_name, user_input)
        logger.silent_write("\n")
    event_bus = team.get_event_bus()
    event_bus.register_default_events()
    responses = await team.handle(Message(content=user_input, role="User"))
    print(responses)


async def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    date_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{date_prefix}_{args.team_name}.log"
    log_file = os.path.join(args.log_dir, log_name)
    logger = setup_logging(log_file)
    await run_test(args.team_name, args.user_input, logger = logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a team test")
    parser.add_argument("team_name", help="The name of the team to run")
    parser.add_argument("user_input", help="The user input to provide to the team")
    parser.add_argument("--log_dir", default = 'run_logs', help="The directory to store logs")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.create_task(main(args))
    loop.run_forever()