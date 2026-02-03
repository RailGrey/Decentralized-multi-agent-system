"""
Summarizer Agent using Mistral API to combine multiple agent outputs.
"""

from mistralai import Mistral
from typing import List, Dict, Any
from settings.logger import logger
import json


class MistralSummarizerAgent:
    """Agent that summarizes and combines outputs from multiple agents."""
    
    def __init__(
        self,
        api_key: str,
        chat_model: str = "mistral-small-latest"
    ):
        """
        Initialize the Summarizer Agent.
        
        Args:
            api_key: Mistral API key
            chat_model: Model for chat completion
        """
        self.api_key = api_key
        self.chat_model = chat_model
        self.client = Mistral(api_key=api_key)
        self.response_schema = self._build_response_schema()
    
    def _build_response_schema(self) -> Dict[str, Any]:
        """Build JSON schema for summarized response."""
        return {
            "type": "object",
            "properties": {
                "final_solution": {
                    "type": "string",
                    "description": "The winning solution, potentially enhanced with insights from other solutions"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Overall confidence in the final solution based on voting consensus"
                },
                "winner_rationale": {
                    "type": "string",
                    "description": "Why this solution won - what made it better than alternatives"
                },
                "key_strengths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key strengths of the winning solution"
                },
                "potential_improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Potential improvements or alternative approaches from other solutions"
                },
                "voting_consensus": {
                    "type": "string",
                    "description": "Summary of voting results and consensus level"
                },
                "brainstorm_impact": {
                    "type": "string",
                    "description": "How the brainstorm phase influenced the final solution"
                }
            },
            "required": ["final_solution", "confidence", "winner_rationale", "key_strengths"],
            "additionalProperties": False
        }
    
    def summarize_winner(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize the winning solution with voting context.
        
        Args:
            context: Dictionary containing:
                - task: Original task
                - winner: Name of winning agent
                - winning_solution: The winning solution dict
                - all_solutions: All solutions for comparison
                - votes: Voting results from all agents
                - ideas: Brainstorm ideas
                
        Returns:
            Structured summary focused on winner
        """
        # Build formatted context
        formatted_context = self._build_winner_context(context)
        
        # Create system prompt
        system_prompt = """You are a Summarizer Agent. Your job is to present the winning solution from a multi-agent voting process.

You will receive:
1. The original task
2. The winning solution (chosen by agent voting)
3. All solutions for comparison
4. Complete voting results
5. Brainstorm ideas that informed the solutions

Your response must:
- Present the winning solution clearly
- Explain why it won based on voting feedback
- Highlight its key strengths
- Suggest potential improvements from other solutions if relevant
- Provide confidence based on voting consensus
- Explain how brainstorming contributed to the solution

RESPOND IN JSON FORMAT with this structure:
{
  "final_solution": "<the winning solution, potentially enhanced>",
  "confidence": 0.85,
  "winner_rationale": "<why this solution won>",
  "key_strengths": ["strength 1", "strength 2"],
  "potential_improvements": ["improvement 1", "improvement 2"],
  "voting_consensus": "<strong/moderate/weak consensus explanation>",
  "brainstorm_impact": "<how brainstorming helped>"
}"""
        
        user_prompt = f"""{formatted_context}

Analyze the winning solution and voting results, then provide a comprehensive summary."""
        
        # Get structured response from Mistral
        response = self.client.chat.complete(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            response_format={
                "type": "json_object",
                "json_schema": {
                    "name": "winner_summary",
                    "schema": self.response_schema,
                    "strict": True
                }
            }
        )
        
        # Parse and return the structured response
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _build_winner_context(self, context: Dict[str, Any]) -> str:
        """Build formatted context for winner summarization."""
        task = context.get("task", "")
        winner = context.get("winner", "Unknown")
        winning_sol = context.get("winning_solution", {})
        all_solutions = context.get("all_solutions", {})
        votes = context.get("votes", {})
        ideas = context.get("ideas", [])
        
        lines = []
        lines.append("=" * 70)
        lines.append("ORIGINAL TASK")
        lines.append("=" * 70)
        lines.append(task)
        lines.append("")
        
        lines.append("=" * 70)
        lines.append(f"WINNING SOLUTION - {winner}")
        lines.append("=" * 70)
        lines.append(f"Ideas used: {winning_sol.get('ideas_used', [])}")
        lines.append(f"Time complexity: {winning_sol.get('time_complexity', 'N/A')}")
        lines.append(f"Space complexity: {winning_sol.get('space_complexity', 'N/A')}")
        lines.append(f"Edge cases: {winning_sol.get('edge_cases', [])}")
        lines.append(f"\nCode:\n{winning_sol.get('solution', 'N/A')}")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("OTHER SOLUTIONS (for comparison)")
        lines.append("=" * 70)
        for agent_name, sol in all_solutions.items():
            if agent_name != winner:
                lines.append(f"\n[{agent_name}]")
                lines.append(f"  Time: {sol.get('time_complexity', 'N/A')}")
                lines.append(f"  Space: {sol.get('space_complexity', 'N/A')}")
                lines.append(f"  Ideas: {sol.get('ideas_used', [])}")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("VOTING RESULTS")
        lines.append("=" * 70)
        for voter, voter_votes in votes.items():
            lines.append(f"\n{voter}'s votes:")
            for vote in voter_votes:
                agent = vote.get("agent_name", "?")
                score = vote.get("score", 0)
                reasoning = vote.get("reasoning", "")
                lines.append(f"  {agent}: {score}/10.0 - {reasoning}")
        lines.append("")
        
        lines.append("=" * 70)
        lines.append("BRAINSTORM IDEAS")
        lines.append("=" * 70)
        for idea in ideas:
            lines.append(f"\n[{idea.get('idea_id')}] (score: {idea.get('score', 0)})")
            lines.append(f"  {idea.get('text', '')}")
            lines.append(f"  Author: {idea.get('author', '?')}")
            lines.append(f"  Supporters: {', '.join(idea.get('supporters', []))}")
        
        return "\n".join(lines)
    
    def summarize(self, task: str, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize multiple agent outputs into a single coherent response.
        
        Args:
            task: Original task description
            agent_results: Results from multi-agent solver
            
        Returns:
            Structured summary with final solution
        """
        # Extract all responses and solutions
        responses = agent_results.get('responses', [])
        
        # Build context from all agent interactions
        context = self._build_context(task, agent_results, responses)
        
        # Create system prompt
        system_prompt = """You are a Summarizer Agent. Your job is to analyze outputs from multiple specialized agents and create a single, comprehensive solution.

You will receive:
1. The original task
2. All agent responses and their actions
3. Final solution(s) from executing agent(s)

Your response must:
- Combine insights from all agents into a coherent solution
- Maintain technical accuracy from specialized agents
- Provide a confidence score based on agent agreements and solution quality
- Highlight key insights from different agents
- Summarize each agent's contribution

RESPOND IN JSON FORMAT with this structure:
{
  "final_solution": "<unified comprehensive solution>",
  "confidence": 0.85,
  "key_insights": ["insight 1", "insight 2"],
  "agent_contributions": {
    "Agent1": "contribution description",
    "Agent2": "contribution description"
  }
}"""
        
        user_prompt = f"""{context}

Analyze all the agent interactions and outputs above, then provide a comprehensive summary with:
1. A unified final solution that combines the best insights
2. An overall confidence score
3. Key insights from the process
4. Summary of each agent's contribution"""
        
        # Get structured response from Mistral
        response = self.client.chat.complete(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=False,
            response_format={
                "type": "json_object",
                "json_schema": {
                    "name": "summary_response",
                    "schema": self.response_schema,
                    "strict": True
                }
            }
        )
        
        # Parse and return the structured response
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _build_context(self, task: str, agent_results: Dict[str, Any], responses: List[Dict[str, Any]]) -> str:
        """Build context string from all agent interactions."""
        context = f"ORIGINAL TASK:\n{task}\n\n"
        context += "="*70 + "\n"
        context += "AGENT INTERACTION HISTORY:\n"
        context += "="*70 + "\n\n"
        
        # Add each agent's response
        for resp in responses:
            agent_name = resp.get('agent', 'Unknown')
            action = resp.get('action', 'Unknown')
            iteration = resp.get('iteration', 0)
            
            context += f"Step {iteration} - {agent_name}:\n"
            context += f"Action: {action}\n"
            
            if action == "Pass":
                context += f"Passed to: {resp.get('pass_to')}\n"
            elif action == "Split":
                context += f"Split into:\n"
                context += f"  Task 1: {resp.get('task1')}\n"
                context += f"  Task 2: {resp.get('task2')}\n"
                context += f"  Task 1 → {resp.get('task1_to')}\n"
                context += f"  Task 2 → {resp.get('task2_to')}\n"
            elif action == "Execute":
                context += f"Confidence: {resp.get('confidence')}\n"
                context += f"Solution:\n{resp.get('solution')}\n"
            
            context += "\n" + "-"*70 + "\n\n"
        
        # Add final result summary
        context += "="*70 + "\n"
        context += "FINAL RESULT:\n"
        context += "="*70 + "\n"
        
        if agent_results.get('success'):
            if agent_results.get('action') == 'Split':
                context += "Task was split into subtasks.\n"
                # Add subtask results
                subtasks = agent_results.get('subtask_results', {})
                for task_name, task_result in subtasks.items():
                    if task_result.get('success'):
                        context += f"\n{task_name.upper()}:\n"
                        context += f"Agent: {task_result.get('final_agent')}\n"
                        context += f"Confidence: {task_result.get('confidence')}\n"
                        context += f"Solution: {task_result.get('solution')}\n"
            else:
                context += f"Final Agent: {agent_results.get('final_agent')}\n"
                context += f"Confidence: {agent_results.get('confidence')}\n"
                context += f"Solution:\n{agent_results.get('solution')}\n"
        else:
            context += f"Error: {agent_results.get('error')}\n"
        
        return context
    
    @staticmethod
    def print_summary(summary: Dict[str, Any]):
        """Pretty print the summary."""
        logger.info("="*70)
        logger.info("FINAL SUMMARY")
        logger.info("="*70)
        
        # Check if this is winner-focused summary or old format
        if 'winner_rationale' in summary:
            # New winner-focused format
            logger.info(f"Confidence: {summary.get('confidence', 'N/A')}")
            
            logger.info("-"*70)
            logger.info("WHY THIS SOLUTION WON:")
            logger.info("-"*70)
            logger.info(summary.get('winner_rationale', 'No rationale provided'))
            
            logger.info("-"*70)
            logger.info("FINAL SOLUTION:")
            logger.info("-"*70)
            logger.info(summary.get('final_solution', 'No solution provided'))
            
            if 'key_strengths' in summary and summary['key_strengths']:
                logger.info("-"*70)
                logger.info("KEY STRENGTHS:")
                logger.info("-"*70)
                for i, strength in enumerate(summary['key_strengths'], 1):
                    logger.info(f"{i}. {strength}")
            
            if 'potential_improvements' in summary and summary['potential_improvements']:
                logger.info("-"*70)
                logger.info("POTENTIAL IMPROVEMENTS:")
                logger.info("-"*70)
                for i, improvement in enumerate(summary['potential_improvements'], 1):
                    logger.info(f"{i}. {improvement}")
            
            if 'voting_consensus' in summary:
                logger.info("-"*70)
                logger.info("VOTING CONSENSUS:")
                logger.info("-"*70)
                logger.info(summary['voting_consensus'])
            
            if 'brainstorm_impact' in summary:
                logger.info("-"*70)
                logger.info("BRAINSTORM IMPACT:")
                logger.info("-"*70)
                logger.info(summary['brainstorm_impact'])
        else:
            # Old format (backward compatibility)
            logger.info(f"Overall Confidence: {summary.get('confidence', 'N/A')}")
            
            logger.info("-"*70)
            logger.info("FINAL SOLUTION:")
            logger.info("-"*70)
            logger.info(summary.get('final_solution', 'No solution provided'))
            
            if 'key_insights' in summary and summary['key_insights']:
                logger.info("-"*70)
                logger.info("KEY INSIGHTS:")
                logger.info("-"*70)
                for i, insight in enumerate(summary['key_insights'], 1):
                    logger.info(f"{i}. {insight}")
            
            if 'agent_contributions' in summary and summary['agent_contributions']:
                logger.info("-"*70)
                logger.info("AGENT CONTRIBUTIONS:")
                logger.info("-"*70)
                for agent, contribution in summary['agent_contributions'].items():
                    logger.info(f"• {agent}: {contribution}")
        
        logger.info("="*70)
