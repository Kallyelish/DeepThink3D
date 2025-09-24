import sys
from tqdm import tqdm
import json
import os
import argparse
import sys
import signal

sys.path.append("../src")
from LLMTPC_agent.LLMTPC_SOP import LLMTPC_SOP as SOP
from LLMTPC_agent.agent import LLMTPC_Agent as Agent
from LLMTPC_agent.environment import LLMTPC_Environment as Environment
from dataset.scene_dataset import SceneDataset
from LLMTPC_executor.api.api import Match_SQA3D_Attr
import re


def get_previous_question_ids(config):
    res = []
    log_dir = config["log_path"]
    if not os.path.exists(log_dir):
        return res
    log_file_names = os.listdir(log_dir)
    r = r"scene(\d+)_(\d+)_(\d+)-(\d+-\d+-\d+-\d+:\d+:\d+)"
    for log_file_name in log_file_names:
        m = re.match(r, log_file_name)
        if m:
            question_id = m.group(3)
            res.append(str(question_id))
    return res


def init(config):
    sop = SOP.from_config(config)
    agents, roles_to_names, names_to_roles = Agent.from_config(config)
    environment = Environment.from_config(config)
    environment.agents = agents
    environment.roles_to_names, environment.names_to_roles = (
        roles_to_names,
        names_to_roles,
    )
    sop.roles_to_names, sop.names_to_roles = roles_to_names, names_to_roles
    for name, agent in agents.items():
        agent.environment = environment
    return agents, sop, environment


def init_scene_dataset(config):
    scene_dataset = SceneDataset.from_config(config["config"]["scene_info"])
    return scene_dataset


def init_openshape_matcher(config):
    openshape_matcher = Match_SQA3D_Attr(
        config_path=config["config"]["openshape_config"]["meta_info"][
            "openshape_config_path"
        ],
        openclip_model_path=config["config"]["openshape_config"]["meta_info"][
            "openclip_model_path"
        ],
        openshape_model_path=config["config"]["openshape_config"]["meta_info"][
            "openshape_model_path"
        ],
    )
    openshape_matcher.init_model()
    return openshape_matcher


def run(agents: Agent, sop: SOP, environment: Environment):
    i = 0
    action = None
    while True:
        current_state, current_agent = sop.next(
            environment, agents, action
        )  # State, Agent
        if sop.finished:
            os.environ.clear()
            break
        user_input = input(f"{current_agent.name}:") if current_agent.is_user else ""

        action = current_agent.step(current_state, user_input, action)

        memory = action.process()

        environment.update_memory(memory, current_state, current_agent)


def handler(signum, frame):
    print(
        "\n----------Timeout: 10 minutes exceeded for this question. Exiting.----------\n"
    )
    sys.exit(1)


signal.signal(signal.SIGALRM, handler)

parser = argparse.ArgumentParser(description="LLM-TPC")
parser.add_argument("--agent", type=str, help="path to SOP json")
parser.add_argument("--split", type=str, help="split")
args = parser.parse_args()

with open(args.agent, "r") as f:
    config = json.load(f)

with open(args.split, "r") as f:
    split = json.load(f)

if config["config"]["setting"]["use_openshape"]:
    config["config"]["openshape_config"]["attr_matcher"] = init_openshape_matcher(
        config
    )

scene_dataset = init_scene_dataset(config)
config["config"]["scene_info"]["scene_dataset"] = scene_dataset

previous_questions = get_previous_question_ids(config)
print(f"{len(previous_questions)=}")

for qa_scene in tqdm(scene_dataset.qa.keys()):
    for qa_question in tqdm(scene_dataset.qa[qa_scene].keys(), leave=False):
        signal.alarm(600)
        try:
            if qa_question in previous_questions:
                if "LOG_ID" in os.environ:
                    tqdm.write(f"{qa_scene=}, {qa_question=}, exists")
                continue
            if split and qa_question not in split:
                if "LOG_ID" in os.environ:
                    tqdm.write(f"{qa_scene=}, {qa_question=}, not in split")
                continue
            if "LOG_ID" in os.environ:
                tqdm.write(f"{qa_scene=}, {qa_question=}, start")
            config["config"]["scene_info"]["qa_info"]["scene_id"] = qa_scene
            config["config"]["scene_info"]["qa_info"]["question_id"] = qa_question
            scene = scene_dataset.get_scene(qa_scene, qa_question)
            config["config"]["scene_info"]["scene"] = scene
            agents, sop, environment = init(config)
            try:
                run(agents, sop, environment)
            except Exception as e:
                tqdm.write(f"Error: {e}")
                continue

            if "LOG_ID" in os.environ:
                tqdm.write(f"{qa_scene=}, {qa_question=}, end")
        finally:
            signal.alarm(0)
