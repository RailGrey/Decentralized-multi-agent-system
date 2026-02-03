# ─── Response Schemas ──────────────────────
# Every field that may be semantically empty is still *present* in the output
# (set to "" or []).  This satisfies strict: True.

CONFIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": (
                "Your confidence as a NUMERIC VALUE between 0.0 and 1.0. "
                "Examples: 0.9 for high confidence, 0.5 for medium, 0.2 for low. "
                "DO NOT use words like 'high', 'medium', or 'low'. "
                "MUST be a decimal number."
            )
        },
        "reason": {
            "type": "string",
            "description": "One-sentence explanation for your confidence level."
        }
    },
    "required": ["confidence", "reason"],
    "additionalProperties": False
}

BRAINSTORM_SCHEMA = {
    "type": "object",
    "properties": {
        "contributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "support", "criticize"],
                        "description": (
                            "'add' — propose a new idea.  "
                            "'support' — agree with an existing idea (your confidence is added to its score).  "
                            "'criticize' — flag a specific flaw in an existing idea."
                        )
                    },
                    "idea_id": {
                        "type": "string",
                        "description": (
                            "The idea_id to support or criticize.  "
                            "Must be an empty string when action is 'add'."
                        )
                    },
                    "text": {
                        "type": "string",
                        "description": (
                            "The new idea text when action is 'add'.  "
                            "Must be an empty string when action is 'support' or 'criticize'."
                        )
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Reason for criticism when action is 'criticize'.  "
                            "Must be an empty string when action is 'add' or 'support'."
                        )
                    }
                },
                "required": ["action", "idea_id", "text", "reason"],
                "additionalProperties": False
            },
            "description": (
                "Your contributions.  Do NOT add a new idea when an existing one "
                "already covers the same point — support it instead."
            )
        }
    },
    "required": ["contributions"],
    "additionalProperties": False
}

CODE_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "idea_ids from the brainstorm that directly informed your solution."
        },
        "solution": {
            "type": "string",
            "description": "Complete, runnable solution code."
        },
        "time_complexity": {
            "type": "string",
            "description": "Time complexity (e.g. 'O(n log n)')."
        },
        "space_complexity": {
            "type": "string",
            "description": "Space complexity (e.g. 'O(n)')."
        },
        "edge_cases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Edge cases your solution handles and how."
        }
    },
    "required": ["ideas_used", "solution", "time_complexity", "space_complexity", "edge_cases"],
    "additionalProperties": False
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_reviewed": {
                        "type": "string",
                        "description": "Name of the agent whose solution you are reviewing."
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["approve", "fix"],
                        "description": "'approve' if the solution is correct, 'fix' if it has issues."
                    },
                    "feedback": {
                        "type": "string",
                        "description": (
                            "Detailed feedback.  If approving, explain why it is correct.  "
                            "If fixing, describe the exact issue."
                        )
                    },
                    "counterexample": {
                        "type": "string",
                        "description": "A concrete input that breaks the solution.  Empty string if none."
                    }
                },
                "required": ["agent_reviewed", "verdict", "feedback", "counterexample"],
                "additionalProperties": False
            },
            "description": "One review entry per other agent.  You must review every other agent."
        }
    },
    "required": ["reviews"],
    "additionalProperties": False
}

REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "ideas_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "idea_ids from brainstorm that inform your solution."
        },
        "solution": {
            "type": "string",
            "description": "Complete, revised solution code."
        },
        "time_complexity": {
            "type": "string",
            "description": "Time complexity."
        },
        "space_complexity": {
            "type": "string",
            "description": "Space complexity."
        },
        "edge_cases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Edge cases your solution handles."
        },
        "changes_made": {
            "type": "string",
            "description": (
                "What you changed and why, based on peer feedback.  "
                "If no changes were needed, explain why the feedback was incorrect."
            )
        }
    },
    "required": ["ideas_used", "solution", "time_complexity", "space_complexity", "edge_cases", "changes_made"],
    "additionalProperties": False
}

VOTING_SCHEMA = {
    "type": "object",
    "properties": {
        "votes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "description": "Name of the agent whose solution you are voting for"
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 10.0,
                        "description": "Your score for this solution (0.0 = worst, 10.0 = best)"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why you gave this score - consider correctness, efficiency, clarity, edge cases"
                    }
                },
                "required": ["agent_name", "score", "reasoning"],
                "additionalProperties": False
            },
            "description": "One vote entry for each agent's solution (including your own)"
        }
    },
    "required": ["votes"],
    "additionalProperties": False
}
