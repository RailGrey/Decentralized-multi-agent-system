"""
Knowledge builder utility for saving coding problems to agent-specific folders.
This avoids searching the dataset every time agents are initialized.
"""

import json
import os
from pathlib import Path
from settings.logger import logger
from typing import List, Dict, Any
from tqdm import tqdm


def get_problems(
    dataset,
    topic: str,
    count: int,
    language: int = 2
) -> List[Dict[str, str]]:
    """
    Extract problems from dataset for a specific topic.
    
    Args:
        dataset: The dataset object (e.g., ds_train)
        topic: CF tag to filter by (e.g., 'greedy', 'trees', 'dp')
        count: Number of problems to extract
        language: Programming language code (2 = Python typically)
        
    Returns:
        List of dicts with 'description' and 'solution' keys
    """
    problems = []
    
    for i in tqdm(range(len(dataset)), desc=f"Searching {topic}"):
        row = dataset[i]
        
        # Check if topic matches and language is available
        if topic in row['cf_tags'] and language in row['solutions']['language']:
            # Get all solutions in the target language
            sol_inds = [
                i for i, lang in enumerate(row['solutions']['language']) 
                if lang == language
            ]
            
            # Take shortest solution (usually cleaner)
            solution = min(
                [row['solutions']['solution'][ind] for ind in sol_inds], 
                key=len
            )
            
            problems.append({
                "description": row['description'],
                "solution": solution
            })
            
            if len(problems) == count:
                break
    
    return problems


def save_agent_knowledge(
    dataset,
    agent_config: Dict[str, Any],
    output_dir: str = "./agent_knowledge",
    count: int = 10,
    language: int = 2
):
    """
    Save problems for a single agent to a JSON file.
    
    Args:
        dataset: Dataset to extract from
        agent_config: Dict with 'name' and 'cf_tag' keys
        output_dir: Base directory for saving knowledge
        count: Number of problems per agent
        language: Programming language code
        
    Returns:
        Path to saved file
    """
    agent_name = agent_config['name']
    cf_tag = agent_config['cf_tag']
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    filename = f"{agent_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Check if already exists
    if os.path.exists(filepath):
        logger.info(f"Knowledge already exists for {agent_name} at {filepath}")
        return filepath
    
    # Extract problems
    logger.info(f"\nExtracting knowledge for {agent_name} (tag: {cf_tag})...")
    problems = get_problems(dataset, cf_tag, count, language)
    
    # Format for agent
    knowledge_items = []
    for i, problem in enumerate(problems, 1):
        knowledge_items.append({
            "content": problem['description'],
            "metadata": {
                "solution": problem['solution'],
                "cf_tag": cf_tag,
                "problem_id": i,
                "agent_name": agent_name
            }
        })
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(knowledge_items, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Saved {len(knowledge_items)} problems to {filepath}")
    
    return filepath


def build_all_agent_knowledge(
    dataset,
    agent_configs: List[Dict[str, Any]],
    output_dir: str = "./agent_knowledge",
    count: int = 10,
    language: int = 2
) -> Dict[str, str]:
    """
    Build knowledge files for all agents.
    
    Args:
        dataset: Dataset to extract from
        agent_configs: List of dicts with 'name' and 'cf_tag' keys
        output_dir: Base directory for saving knowledge
        count: Number of problems per agent
        language: Programming language code
        
    Returns:
        Dict mapping agent name to filepath
    """
    logger.info("=" * 70)
    logger.info("BUILDING AGENT KNOWLEDGE BASE")
    logger.info("=" * 70)
    
    filepaths = {}
    
    for config in agent_configs:
        filepath = save_agent_knowledge(
            dataset=dataset,
            agent_config=config,
            output_dir=output_dir,
            count=count,
            language=language
        )
        filepaths[config['name']] = filepath
    
    logger.info("\n" + "=" * 70)
    logger.info("KNOWLEDGE BASE BUILD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total agents: {len(filepaths)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Problems per agent: {count}")
    
    return filepaths


def load_agent_knowledge(
    agent_name: str,
    knowledge_dir: str = "./agent_knowledge"
) -> List[Dict[str, Any]]:
    """
    Load pre-built knowledge for an agent.
    
    Args:
        agent_name: Name of the agent
        knowledge_dir: Directory containing knowledge files
        
    Returns:
        List of knowledge items
    """
    filename = f"{agent_name.replace(' ', '_').lower()}.json"
    filepath = os.path.join(knowledge_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Knowledge file not found for {agent_name} at {filepath}. "
            f"Run build_all_agent_knowledge() first."
        )
    
    with open(filepath, 'r', encoding='utf-8') as f:
        knowledge = json.load(f)
    
    return knowledge