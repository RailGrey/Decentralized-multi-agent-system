import os
from datasets import load_dataset
from dotenv import load_dotenv
import json

from settings.agent_config import AGENTS
from settings.logger import logger
from rag_agent import MistralRAGAgent
from brainstorm_solver import BrainstormSolver
from knowledge_builder import build_all_agent_knowledge, load_agent_knowledge

import logging

logging.basicConfig(level=logging.INFO)  # or DEBUG, WARNING, ERROR
logging.getLogger('httpx').setLevel(logging.WARNING)

load_dotenv("env/credentials.env")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

ds_train = load_dataset(
    "deepmind/code_contests",
    cache_dir="dataset",
    split='valid'
)

print("Dataset is loaded")

filepaths = build_all_agent_knowledge(
    dataset=ds_train,
    agent_configs=AGENTS,
    output_dir="./agent_knowledge",
    count=10,  # 10 problems per agent
    language=3  # Python
)

print("\nKnowledge files created:")
for name, path in filepaths.items():
    print(f"  {name}: {path}")

agent_system = {}
for config in AGENTS:
    name = config["name"]
    expertise = config["expertise"]
    agent = MistralRAGAgent(
        name=name,
        expertise=expertise,
        api_key=MISTRAL_API_KEY,
        qdrant_path=os.path.join("qdrant_storage", name),
        cache_dir=os.path.join("embedding_cache", name)
    )
    
    # Load pre-built knowledge
    knowledge = load_agent_knowledge(name, "agent_knowledge")
    
    print(f"Loading {len(knowledge)} problems for {name}...")
    
    # Add to agent (uses Qdrant + embedding cache)
    for item in knowledge:
        agent.add_knowledge(
            content=item['content'],
            metadata=item['metadata']
        )
        #time.sleep(1) #to avoid rate limits
    
    print(f"✓ {name} ready ({agent.get_knowledge_count()} items)")

    agent_system[name] = agent

task = "Given an array of integers, find two numbers that add up to a target."
task = ds_train[-2]['description']
solver = BrainstormSolver(agent_system, max_review_rounds=1)
result = solver.solve(task)

serializable_output = {
    "task": task,
    "confidences": result.get("confidences", {}),
    "core_agents": result.get("core_agents", []),
    "ideas": [
        {
            "idea_id": idea.idea_id,
            "text": idea.text,
            "author": idea.author,
            "score": idea.score,
            "supporters": idea.supporters,
            "critics": idea.critics
        }
        for idea in result.get("ideas", [])
    ],
    "solutions": result.get("solutions", {}),
    "votes": result.get("votes", {}),
    "winner": result.get("winner"),
    "final_summary": result.get("final_summary", {})
}

filepath = "solution_example.json"
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(serializable_output, f, indent=2, ensure_ascii=False)

logger.info(f"Solution output saved to {filepath}")