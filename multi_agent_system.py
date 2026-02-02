from typing import Dict, Any, List

from rag_agent import MistralRAGAgent


class MultiAgentSolver:
    """Coordinates multiple agents to solve tasks through routing."""

    def __init__(self, agent_system: Dict[str, MistralRAGAgent]):
        """
        Initialize the multi-agent solver.

        Args:
            agent_system: Dictionary mapping agent names to MistralRAGAgent instances
        """
        self.agent_system = agent_system
        self.execution_history = []

    def solve_task(self, task: str, starting_agent: str = "Math Agent", max_iterations: int = 10) -> Dict[str, Any]:
        """
        Solve a task by routing through agents until execution.

        Args:
            task: The task description
            starting_agent: Name of the first agent to handle the task
            max_iterations: Maximum number of agent hops to prevent infinite loops

        Returns:
            Dictionary containing all agent responses and final solution
        """
        current_task = task
        current_agent_name = starting_agent
        iteration = 0
        responses = []

        while iteration < max_iterations:
            iteration += 1

            # Get current agent
            if current_agent_name not in self.agent_system:
                return {
                    "success": False,
                    "error": f"Agent '{current_agent_name}' not found in system",
                    "responses": responses
                }

            current_agent = self.agent_system[current_agent_name]

            # Get agent's response
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}: {current_agent_name} processing task...")
            print(f"{'='*70}")

            response = current_agent.solve(current_task)
            response['agent'] = current_agent_name
            response['iteration'] = iteration
            responses.append(response)

            action = response.get('action')
            print(f"Action: {action}")

            # Handle different actions
            if action == "Execute":
                # Task is solved
                print(f"✓ {current_agent_name} executed the solution")
                print(f"Confidence: {response.get('confidence', 'N/A')}")

                return {
                    "success": True,
                    "final_agent": current_agent_name,
                    "solution": response.get('solution'),
                    "confidence": response.get('confidence'),
                    "responses": responses,
                    "iterations": iteration
                }

            elif action == "Pass":
                # Pass to another agent
                next_agent = response.get('pass_to')
                print(f"→ Passing to: {next_agent}")

                if not next_agent or next_agent not in self.agent_system:
                    return {
                        "success": False,
                        "error": f"Invalid agent to pass to: {next_agent}",
                        "responses": responses
                    }

                current_agent_name = next_agent
                # Task remains the same

            elif action == "Split":
                # Split into subtasks
                task1 = response.get('task1')
                task2 = response.get('task2')
                task1_to = response.get('task1_to')
                task2_to = response.get('task2_to')

                print(f"↓ Splitting task:")
                print(f"  Task 1 → {task1_to}: {task1[:50]}...")
                print(f"  Task 2 → {task2_to}: {task2[:50]}...")

                # Solve subtasks recursively
                result1 = self.solve_task(task1, task1_to, max_iterations - iteration)
                result2 = self.solve_task(task2, task2_to, max_iterations - iteration)

                # Combine results
                return {
                    "success": True,
                    "action": "Split",
                    "subtask_results": {
                        "task1": result1,
                        "task2": result2
                    },
                    "responses": responses,
                    "iterations": iteration
                }

            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}",
                    "responses": responses
                }

        # Max iterations reached
        return {
            "success": False,
            "error": f"Max iterations ({max_iterations}) reached without execution",
            "responses": responses
        }

    def get_all_solutions(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract all solutions from a result (including from split tasks).

        Args:
            result: Result dictionary from solve_task

        Returns:
            List of all solution dictionaries
        """
        solutions = []

        if result.get('success') and result.get('action') != 'Split':
            # Single solution
            solutions.append({
                'agent': result.get('final_agent'),
                'solution': result.get('solution'),
                'confidence': result.get('confidence'),
                'iterations': result.get('iterations')
            })
        elif result.get('action') == 'Split':
            # Recursively get solutions from subtasks
            subtasks = result.get('subtask_results', {})
            if 'task1' in subtasks:
                solutions.extend(self.get_all_solutions(subtasks['task1']))
            if 'task2' in subtasks:
                solutions.extend(self.get_all_solutions(subtasks['task2']))

        return solutions

    def print_result(self, result: Dict[str, Any]):
        """Pretty print the result."""
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)

        if not result.get('success'):
            print(f"❌ Failed: {result.get('error')}")
            return

        if result.get('action') == 'Split':
            print("✓ Task was split into subtasks")
            solutions = self.get_all_solutions(result)
            for i, sol in enumerate(solutions, 1):
                print(f"\nSolution {i} (from {sol['agent']}):")
                print(f"Confidence: {sol.get('confidence', 'N/A')}")
                print(f"Iterations: {sol.get('iterations')}")
                print(f"Solution: {sol['solution'][:200]}...")
        else:
            print(f"✓ Solved by: {result['final_agent']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Iterations: {result['iterations']}")
            print(f"\nSolution:\n{result['solution']}")

        print("\n" + "="*70)
        print(f"Total agent calls: {len(result['responses'])}")
        print("Agent chain:")
        for resp in result['responses']:
            print(f"  {resp['iteration']}. {resp['agent']} → {resp['action']}")
        print("="*70)


def solve_with_agents(agent_system: Dict[str, MistralRAGAgent], task: str, starting_agent: str = "Math Agent") -> Dict[str, Any]:
    """
    Convenience function to solve a task with the multi-agent system.

    Args:
        agent_system: Dictionary of agent name to MistralRAGAgent
        task: Task description
        starting_agent: First agent to handle the task

    Returns:
        Result dictionary with solutions
    """
    solver = MultiAgentSolver(agent_system)
    result = solver.solve_task(task, starting_agent)
    solver.print_result(result)
    return result