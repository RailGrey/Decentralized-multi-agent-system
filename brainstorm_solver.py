"""
Brainstorm-based multi-agent solver.

Pipeline (no moderator — coordination emerges from confidence thresholds,
shared scored memory, and peer review):

    1. Confidence estimation  →  every agent self-reports
    2. Core-agent selection   →  keep agents at or above the mean
    3. Brainstorm             →  shared, append-only idea memory; ideas are
                                 scored by the sum of supporters' confidences
                                 and re-sorted after every agent call
    4. Code generation        →  each core agent independently writes a full
                                 solution grounded in the brainstorm ideas
    5. Peer review & revision →  fixed-round loop: every agent reviews every
                                 other agent's code, then every agent revises
                                 its own code.  Stops early when all verdicts
                                 are 'approve'.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from settings.logger import logger
from rag_agent import MistralRAGAgent
from summarizer_agent import MistralSummarizerAgent
from settings.prompts import ESTIMATE_CONFIDENCE_SYSTEM_PROMPT, BRAINSTORM_SYSTEM_PROMPT, GENERATE_SOLUTION_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT, REVISE_SYSTEM_PROMPT, VOTING_SYSTEM_PROMPT
from settings.schemas import CONFIDENCE_SCHEMA, BRAINSTORM_SCHEMA, CODE_GEN_SCHEMA, REVIEW_SCHEMA, REVISION_SCHEMA, VOTING_SCHEMA
import random
import time


# ─── Data Structures ────────────────────────────────────────────────────────


@dataclass
class Idea:
    """Single idea in the shared brainstorm memory."""
    idea_id: str
    text: str
    author: str
    score: float                                          # Σ confidence of supporters
    supporters: List[str] = field(default_factory=list)  # agent names
    critics: List[Dict[str, str]] = field(default_factory=list)  # {"agent", "reason"}


# ─── Solver ─────────────────────────────────────────────────────────────────


class BrainstormSolver:
    """
    Decentralised multi-agent solver.  No moderator agent.

    Coordination comes from:
        • mean-based confidence threshold   (who participates)
        • shared, score-sorted idea memory  (what the group knows)
        • fixed-round peer review           (convergence toward correctness)
    """

    def __init__(
        self,
        agent_system: Dict[str, MistralRAGAgent],
        max_review_rounds: int = 3,
        sleep_between_phases: int = 20
    ):
        self.agent_system = agent_system
        self.max_review_rounds = max_review_rounds
        self.sleep_between_phases = sleep_between_phases

    # ── Main entry point ────────────────────────────────────────────────

    def solve(self, task: str) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Returns
        -------
        {
            "confidences":  {agent_name: float},
            "core_agents":  [agent_name, ...],          # sorted desc by confidence
            "ideas":        [Idea, ...],                 # sorted desc by score
            "solutions":    {agent_name: solution_dict}, # after final review round
            "votes":        {agent_name: [vote_dict]},   # voting results
            "winner":       agent_name,                  # winning solution
            "final_summary": summary_dict                # from summarizer
        }
        """
        logger.info("=" * 70)
        logger.info("PHASE 1 — Confidence Estimation")
        logger.info("=" * 70)
        confidences = self._estimate_confidence(task)
        time.sleep(self.sleep_between_phases)

        logger.info("=" * 70)
        logger.info("PHASE 2 — Core Agent Selection  (threshold = mean)")
        logger.info("=" * 70)
        core_agents = self._select_core_agents(confidences)

        logger.info("=" * 70)
        logger.info("PHASE 3 — Brainstorm")
        logger.info("=" * 70)
        ideas = self._brainstorm(task, core_agents, confidences)
        time.sleep(self.sleep_between_phases)

        logger.info("=" * 70)
        logger.info("PHASE 4 — Code Generation")
        logger.info("=" * 70)
        solutions = self._generate_solutions(task, core_agents, ideas)
        time.sleep(self.sleep_between_phases)

        logger.info("=" * 70)
        logger.info("PHASE 5 — Peer Review & Revision")
        logger.info("=" * 70)
        final_solutions = self._review_and_revise(task, core_agents, solutions, ideas)
        time.sleep(self.sleep_between_phases)

        logger.info("=" * 70)
        logger.info("PHASE 6 — Voting")
        logger.info("=" * 70)
        votes, winner = self._voting_phase(task, core_agents, final_solutions)
        time.sleep(self.sleep_between_phases)

        logger.info("=" * 70)
        logger.info("PHASE 7 — Final Summary")
        logger.info ("=" * 70)
        final_summary = self._create_final_summary(task, winner, final_solutions, votes, ideas)

        return {
            "confidences": confidences,
            "core_agents": core_agents,
            "ideas": ideas,
            "solutions": final_solutions,
            "votes": votes,
            "winner": winner,
            "final_summary": final_summary
        }

    # ── Phase 1 ─────────────────────────────────────────────────────────

    def _estimate_confidence(self, task: str) -> Dict[str, float]:
        """Every agent self-reports confidence. Returns {name: float}."""
        confidences: Dict[str, float] = {}

        for agent_name, agent in self.agent_system.items():
            result = agent.call(
                task=task,
                system_prompt=ESTIMATE_CONFIDENCE_SYSTEM_PROMPT.substitute(
                    agent_name=agent_name,
                    agent_expertise=agent.expertise,
                ),
                response_schema=CONFIDENCE_SCHEMA,
                schema_name="confidence_estimation"
            )

            # Handle both possible formats the API might return
            try:
                # Try to parse confidence directly
                confidence_raw = result.get("confidence")
                
                # If it's a string like "high"/"medium"/"low", convert it
                if isinstance(confidence_raw, str):
                    confidence_raw = confidence_raw.strip().lower()
                    mapping = {
                        "high": 0.85,
                        "medium": 0.5,
                        "low": 0.2,
                        "very high": 0.95,
                        "very low": 0.1
                    }
                    if confidence_raw in mapping:
                        confidence_value = mapping[confidence_raw]
                        logger.warning(f"  WARNING: {agent_name} returned '{confidence_raw}' → converted to {confidence_value}")
                    else:
                        # Try to parse as number string
                        confidence_value = float(confidence_raw)
                else:
                    confidence_value = float(confidence_raw)
                
                # Clamp to valid range
                confidence_value = max(0.0, min(1.0, confidence_value))
                
            except (ValueError, TypeError) as e:
                logger.error(f"  ERROR: {agent_name} returned invalid confidence: {result.get('confidence')} → defaulting to 0.5")
                confidence_value = 0.5
            
            confidences[agent_name] = confidence_value
            
            # Handle field name variation (reason vs reasoning)
            reason_text = result.get('reason') or result.get('reasoning', 'No reason provided')
            logger.info(f"  {agent_name:22} {confidence_value:.2f}  — {reason_text}")

        return confidences

    # ── Phase 2 ─────────────────────────────────────────────────────────

    def _select_core_agents(self, confidences: Dict[str, float]) -> List[str]:
        """Agents at or above the mean, sorted by descending confidence."""
        mean = sum(confidences.values()) / len(confidences)

        core = sorted(
            ((name, c) for name, c in confidences.items() if c >= mean),
            key=lambda x: x[1],
            reverse=True
        )

        logger.info(f"  Mean threshold : {mean:.2f}")
        logger.info(f"  Core agents    : {[n for n, _ in core]}")
        return [name for name, _ in core]

    # ── Phase 3 ─────────────────────────────────────────────────────────

    def _brainstorm(
        self,
        task: str,
        core_agents: List[str],
        confidences: Dict[str, float]
    ) -> List[Idea]:
        """
        Sequential pass in descending-confidence order.
        The shared idea list is mutated and re-sorted after every agent call
        so the next agent sees the latest state.
        """
        ideas: List[Idea] = []
        idea_counter = 0

        for agent_name in core_agents:
            agent = self.agent_system[agent_name]
            agent_conf = confidences[agent_name]

            result = agent.call(
                task=task,
                system_prompt=BRAINSTORM_SYSTEM_PROMPT.substitute(
                    agent_name=agent_name,
                    agent_expertise=agent.expertise,
                    agent_conf=f"{agent_conf:.2f}",
                    ideas=self._format_ideas(ideas)
                ),
                response_schema=BRAINSTORM_SCHEMA,
                schema_name="brainstorm_contribution",
                retrieval_query=task                # RAG still searches on original task
            )

            for c in result.get("contributions", []):
                action = c["action"]

                if action == "add" and c["text"]:
                    idea_counter += 1
                    idea_id = f"idea_{idea_counter}"
                    ideas.append(Idea(
                        idea_id=idea_id,
                        text=c["text"],
                        author=agent_name,
                        score=agent_conf,
                        supporters=[agent_name]
                    ))
                    logger.info(f"  {agent_name:22} ADD       [{idea_id}] {c['text'][:52]}…")

                elif action == "support" and c["idea_id"]:
                    target = self._find_idea(ideas, c["idea_id"])
                    if target:
                        target.score += agent_conf
                        target.supporters.append(agent_name)
                        logger.info(f"  {agent_name:22} SUPPORT   [{target.idea_id}] score → {target.score:.2f}")

                elif action == "criticize" and c["idea_id"]:
                    target = self._find_idea(ideas, c["idea_id"])
                    if target:
                        target.critics.append({"agent": agent_name, "reason": c["reason"]})
                        logger.info(f"  {agent_name:22} CRITICIZE [{target.idea_id}] {c['reason'][:52]}…")

            # Re-sort so the next agent sees the current ranking
            ideas.sort(key=lambda x: x.score, reverse=True)

        return ideas

    # ── Phase 4 ─────────────────────────────────────────────────────────

    def _generate_solutions(
        self,
        task: str,
        core_agents: List[str],
        ideas: List[Idea]
    ) -> Dict[str, Dict[str, Any]]:
        """Each core agent independently writes a complete solution."""
        solutions: Dict[str, Dict[str, Any]] = {}
        ideas_text = self._format_ideas(ideas)

        for agent_name in core_agents:
            agent = self.agent_system[agent_name]

            result = agent.call(
                task=task,
                system_prompt=GENERATE_SOLUTION_SYSTEM_PROMPT.substitute(
                    agent_name=agent_name,
                    agent_expertise=agent.expertise,
                    ideas_text=ideas_text
                ),
                response_schema=CODE_GEN_SCHEMA,
                schema_name="code_generation",
                retrieval_query=task
            )

            solutions[agent_name] = result
            logger.info(f"  {agent_name:22} ideas={result.get('ideas_used', [])}  "
                        f"time={result.get('time_complexity', '?')}")

        return solutions

    # ── Phase 5 ─────────────────────────────────────────────────────────

    def _review_and_revise(
        self,
        task: str,
        core_agents: List[str],
        solutions: Dict[str, Dict[str, Any]],
        ideas: List[Idea]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fixed-round loop.  Each round:
            A)  every agent reviews every *other* agent's code
            B)  every agent revises its own code based on the feedback it received
        Stops early when every single verdict in a round is 'approve'.
        """
        current = dict(solutions)           # working copy
        ideas_text = self._format_ideas(ideas)

        for round_num in range(1, self.max_review_rounds + 1):
            logger.info(f"  ── Review round {round_num}/{self.max_review_rounds} ──")

            # A) Review
            reviews: Dict[str, Dict] = {}
            for agent_name in core_agents:
                agent = self.agent_system[agent_name]

                result = agent.call(
                    task=task,
                    system_prompt=REVIEW_SYSTEM_PROMPT.substitute(
                        agent_name=agent_name,
                        agent_expertise=agent.expertise,
                        own_solution=self._fmt_solution(agent_name, current.get(agent_name, {})),
                        other_solutions=self._fmt_other_solutions(current, agent_name)
                    ),
                    response_schema=REVIEW_SCHEMA,
                    schema_name="code_review",
                    retrieval_query=task
                )

                reviews[agent_name] = result
                for r in result.get("reviews", []):
                    logger.info(f"    {agent_name:22}→ {r.get('agent_reviewed', '?'):22} [{r.get('verdict')}]")

            time.sleep(self.sleep_between_phases)

            # Early stop check
            if self._all_approved(reviews):
                logger.info(f"  ✓ All solutions approved — stopping early.")
                break

            # B) Revise
            for agent_name in core_agents:
                agent = self.agent_system[agent_name]
                own = current.get(agent_name, {})

                result = agent.call(
                    task=task,
                    system_prompt=REVISE_SYSTEM_PROMPT.substitute(
                        agent_name=agent_name,
                        agent_expertise=agent.expertise,
                        ideas_text=ideas_text,
                        own_solution=self._fmt_solution(agent_name, own),
                        other_solutions=self._collect_feedback_for(reviews, agent_name)
                    ),
                    response_schema=REVISION_SCHEMA,
                    schema_name="code_revision",
                    retrieval_query=task
                )

                current[agent_name] = result
                summary = result.get("changes_made") or "no changes"

                if not isinstance(summary, str):
                    logger.warning(f"  WARNING: {agent_name} returned changes_made in non-valid form")
                    summary = ""

                logger.info(f"    {agent_name:22} revised — {summary[:55]}")

            time.sleep(self.sleep_between_phases)

        return current

    # ── Phase 6 ─────────────────────────────────────────────────────────

    def _voting_phase(
        self,
        task: str,
        core_agents: List[str],
        solutions: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
        """
        Each agent votes on all solutions (including their own).
        Returns votes dict and winner name.
        In case of tie, winner is chosen randomly.
        """
        all_votes: Dict[str, List[Dict[str, Any]]] = {}
        
        for agent_name in core_agents:
            agent = self.agent_system[agent_name]
            
            result = agent.call(
                task=task,
                system_prompt=VOTING_SYSTEM_PROMPT.substitute(
                    agent_name=agent_name,
                    agent_expertise=agent.expertise,
                    all_solutions=self._format_all_solutions(solutions)
                ),
                response_schema=VOTING_SCHEMA,
                schema_name="voting",
                retrieval_query=task
            )
            
            all_votes[agent_name] = result.get("votes", [])
            logger.info(f"  {agent_name:22} cast {len(result.get('votes', []))} votes")
        
        # Aggregate scores
        score_totals: Dict[str, float] = {name: 0.0 for name in core_agents}
        
        for voter, votes in all_votes.items():
            for vote in votes:
                candidate = vote.get("agent_name")
                score = float(vote.get("score", 0.0))
                if candidate in score_totals:
                    score_totals[candidate] += score
        
        # Find winner (with random tiebreaker)
        max_score = max(score_totals.values())
        winners = [name for name, score in score_totals.items() if score == max_score]
        
        if len(winners) > 1:
            winner = random.choice(winners)
            logger.info(f"  Tie between {winners} with score {max_score:.2f}")
            logger.info(f"  Random selection: {winner}")
        else:
            winner = winners[0]
            logger.info(f"  Winner: {winner} with total score {max_score:.2f}")
        
        # log all scores
        logger.info("  Final scores:")
        for name in sorted(score_totals.keys(), key=lambda n: score_totals[n], reverse=True):
            logger.info(f"    {name:22} {score_totals[name]:6.2f}")
        
        return all_votes, winner

    # ── Phase 7 ─────────────────────────────────────────────────────────

    def _create_final_summary(
        self,
        task: str,
        winner: str,
        solutions: Dict[str, Dict[str, Any]],
        votes: Dict[str, List[Dict[str, Any]]],
        ideas: List[Idea]
    ) -> Dict[str, Any]:
        """
        Create final summary using the winning solution.
        """

        # Get API key from one of the agents
        api_key = list(self.agent_system.values())[0].api_key
        
        summarizer = MistralSummarizerAgent(api_key=api_key)
        
        # Build context for summarizer
        summary_context = {
            "task": task,
            "winner": winner,
            "winning_solution": solutions[winner],
            "all_solutions": solutions,
            "votes": votes,
            "ideas": [
                {
                    "idea_id": idea.idea_id,
                    "text": idea.text,
                    "score": idea.score,
                    "author": idea.author,
                    "supporters": idea.supporters
                }
                for idea in ideas
            ]
        }
        
        summary = summarizer.summarize_winner(summary_context)
        
        logger.info(f"  Summarizer created final summary")
        logger.info(f"  Winner solution: {winner}")
        logger.info(f"  Confidence: {summary.get('confidence', 'N/A')}")

        return summary


    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _find_idea(ideas: List[Idea], idea_id: str) -> Optional[Idea]:
        for idea in ideas:
            if idea.idea_id == idea_id:
                return idea
        return None

    @staticmethod
    def _format_ideas(ideas: List[Idea]) -> str:
        if not ideas:
            return "No ideas yet — you are the first contributor."

        lines = ["Current shared ideas (sorted by score):\n"]
        for idea in ideas:
            lines.append(
                f"  [{idea.idea_id}]  score={idea.score:.2f}  "
                f"(author: {idea.author} | supporters: {', '.join(idea.supporters)})"
            )
            lines.append(f"          {idea.text}")
            for c in idea.critics:
                lines.append(f"          ⚠ criticized by {c['agent']}: {c['reason']}")
            lines.append("")
        return "\n".join(lines)
    
    @staticmethod
    def _format_all_solutions(solutions: Dict[str, Dict[str, Any]]) -> str:
        """Format all solutions for voting."""
        lines = []
        for agent_name, sol in solutions.items():
            lines.append(f"{'='*60}")
            lines.append(f"Solution by: {agent_name}")
            lines.append(f"{'='*60}")
            lines.append(f"Ideas used: {str(sol.get('ideas_used', []))}")
            lines.append(f"Time complexity: {str(sol.get('time_complexity', 'N/A'))}")
            lines.append(f"Space complexity: {str(sol.get('space_complexity', 'N/A'))}")
            lines.append(f"Edge cases: {str(sol.get('edge_cases', []))}")
            lines.append(f"\nCode:")
            lines.append(sol.get('solution', 'N/A'))
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _fmt_solution(agent_name: str, sol: Dict) -> str:
        return (
            f"[{agent_name}]\n"
            f"  ideas_used:       {sol.get('ideas_used', [])}\n"
            f"  time_complexity:  {sol.get('time_complexity', 'N/A')}\n"
            f"  space_complexity: {sol.get('space_complexity', 'N/A')}\n"
            f"  edge_cases:       {sol.get('edge_cases', [])}\n"
            f"  code:\n"
            f"{sol.get('solution', 'N/A')}\n"
        )

    def _fmt_other_solutions(self, solutions: Dict[str, Dict], exclude: str) -> str:
        return "\n".join(
            self._fmt_solution(name, sol)
            for name, sol in solutions.items()
            if name != exclude
        )

    @staticmethod
    def _collect_feedback_for(reviews: Dict[str, Dict], target: str) -> str:
        """Gather every review entry that points at *target*."""
        parts: List[str] = []
        for reviewer, result in reviews.items():
            if reviewer == target:
                continue
            for r in result.get("reviews", []):
                if r.get("agent_reviewed") == target:
                    parts.append(
                        f"From {reviewer}:\n"
                        f"  verdict:        {r.get('verdict')}\n"
                        f"  feedback:       {r.get('feedback')}\n"
                        f"  counterexample: {r.get('counterexample') or 'none'}\n"
                    )
        return "\n".join(parts) if parts else "No feedback received."

    @staticmethod
    def _all_approved(reviews: Dict[str, Dict]) -> bool:
        return all(
            r.get("verdict") == "approve"
            for result in reviews.values()
            for r in result.get("reviews", [])
        )