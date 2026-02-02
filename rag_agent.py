"""
Single RAG-based AI Agent using Mistral API with structured action-based responses.
"""

from mistralai import Mistral
from typing import List, Dict, Any, Optional
import numpy as np
import json
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class MistralRAGAgent:
    """RAG-based AI Agent using Mistral API for embeddings and chat completion."""
    
    def __init__(
        self,
        name: str,
        expertise: str,
        api_key: str,
        available_agents: List[str] = None,
        chat_model: str = "mistral-small-latest",
        embedding_model: str = "mistral-embed",
        cache_dir: str = None
    ):
        """
        Initialize the Mistral RAG Agent.
        
        Args:
            name: Agent name
            expertise: Area of expertise
            api_key: Mistral API key
            available_agents: List of agent names that tasks can be passed to
            chat_model: Model for chat completion
            embedding_model: Model for embeddings
            cache_dir: Directory to cache embeddings. If None, no caching is used.
        """
        self.name = name
        self.expertise = expertise
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.available_agents = available_agents or []
        self.documents: List[Document] = []
        self.client = Mistral(api_key=api_key)
        self.response_schema = self._build_response_schema()
        
        # Cache setup
        self.cache_dir = cache_dir
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache filename."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding if it exists."""
        if not self.cache_dir:
            return None
        
        text_hash = self._compute_hash(text)
        cache_path = os.path.join(self.cache_dir, f"{text_hash}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    return cached_data.get('embedding')
            except Exception as e:
                # If cache read fails, continue to API call
                print(f"Warning: Failed to read cache for {text_hash[:8]}: {e}")
                return None
        
        return None
    
    def _save_cached_embedding(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Save embedding to cache."""
        if not self.cache_dir:
            return
        
        text_hash = self._compute_hash(text)
        cache_path = os.path.join(self.cache_dir, f"{text_hash}.json")
        
        try:
            cache_data = {
                'text': text,
                'embedding': embedding,
                'metadata': metadata or {},
                'model': self.embedding_model
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache for {text_hash[:8]}: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from cache or API."""
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached
        
        # Call API
        embedding_response = self.client.embeddings.create(
            model=self.embedding_model,
            inputs=[text]
        )
        embedding = embedding_response.data[0].embedding
        
        # Save to cache
        self._save_cached_embedding(text, embedding)
        
        return embedding
    
    def _build_response_schema(self) -> Dict[str, Any]:
        """Build JSON schema for structured responses based on action type."""
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["Pass", "Split", "Execute"],
                    "description": "Action to take: Pass to another agent, Split into subtasks, or Execute the solution"
                },
                "pass_to": {
                    "type": "string",
                    "description": "Agent name to pass the task to (only for Pass action)"
                },
                "task1": {
                    "type": "string",
                    "description": "First subtask (only for Split action)"
                },
                "task2": {
                    "type": "string",
                    "description": "Second subtask (only for Split action)"
                },
                "task1_to": {
                    "type": "string",
                    "description": "Agent to handle task1 (only for Split action)"
                },
                "task2_to": {
                    "type": "string",
                    "description": "Agent to handle task2 (only for Split action)"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence level in the solution (only for Execute action)"
                },
                "solution": {
                    "type": "string",
                    "description": "Detailed solution with code and explanation (only for Execute action)"
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to the agent's knowledge base."""
        # Get embedding (from cache or API)
        embedding = self._get_embedding(content)
        
        # Save metadata to cache if provided
        if self.cache_dir and metadata:
            self._save_cached_embedding(content, embedding, metadata)
        
        doc = Document(
            content=content,
            metadata=metadata or {},
            embedding=embedding
        )
        self.documents.append(doc)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from knowledge base using semantic search."""
        if not self.documents:
            return ""
        
        # Get query embedding (from cache or API)
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc in self.documents:
            sim = self._cosine_similarity(query_embedding, doc.embedding)
            similarities.append((sim, doc))
        
        # Sort by similarity and get top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_docs = [doc for _, doc in similarities[:top_k]]
        
        # Format context
        context = "Relevant knowledge:\n\n"
        for i, doc in enumerate(top_docs, 1):
            context += f"[{i}] {doc.content}\n\n"
        
        return context
    
    def call(
        self,
        task: str,
        system_prompt: str,
        response_schema: Dict[str, Any],
        schema_name: str,
        retrieval_query: str = None
    ) -> Dict[str, Any]:
        """
        Generic call with caller-controlled system prompt and response schema.
        RAG context is always prepended to the user message.
        
        Args:
            task: User-facing prompt (becomes the user message body)
            system_prompt: Fully custom system prompt
            response_schema: JSON schema for the structured response
            schema_name: Identifier for the schema
            retrieval_query: What to search the knowledge base with.
                             Defaults to `task` when None.
        """
        context = self.retrieve_context(retrieval_query or task)

        user_prompt = f"{context}\n{task}" if context else task

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
                    "name": schema_name,
                    "schema": response_schema,
                    "strict": True
                }
            }
        )

        return json.loads(response.choices[0].message.content)

    def solve(self, task: str) -> Dict[str, Any]:
        """
        Solve a coding task using RAG with structured action-based response.
        
        Args:
            task: The coding task description
            
        Returns:
            Structured response dictionary with action and relevant fields
        """
        # Retrieve relevant context
        context = self.retrieve_context(task)
        
        # Build system prompt with action instructions
        agents_list = ", ".join(self.available_agents) if self.available_agents else "none"
        system_prompt = f"""You are {self.name}, an expert in {self.expertise}.

You must respond with a structured JSON object containing one of three actions:

1. **Pass**: If the task is outside your expertise, pass it to another agent
   - Required fields: "action": "Pass", "pass_to": "<agent_name>"
   - Available agents: {agents_list}

2. **Split**: If the task requires multiple areas of expertise, split it into subtasks
   - Required fields: "action": "Split", "task1": "<description>", "task2": "<description>", "task1_to": "<agent_name>", "task2_to": "<agent_name>"
   - Available agents: {agents_list}

3. **Execute**: If you can solve the task yourself
   - Required fields: "action": "Execute", "confidence": <0.0-1.0>, "solution": "<detailed solution with code>"

Use the provided context to inform your decision.

RESPOND IN JSON FORMAT. Examples:

Pass example:
{{
  "action": "Pass",
  "pass_to": "Graph Agent"
}}

Split example:
{{
  "action": "Split",
  "task1": "Parse the input",
  "task2": "Compute shortest path",
  "task1_to": "String Agent",
  "task2_to": "Graph Agent"
}}

Execute example:
{{
  "action": "Execute",
  "confidence": 0.85,
  "solution": "def solve(...):\\n    ..."
}}"""
        
        user_prompt = f"{context}\nTask: {task}\n\nAnalyze this task and provide your response in the required JSON format."
        
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
                    "name": "agent_response",
                    "schema": self.response_schema,
                    "strict": True
                }
            }
        )
        
        # Parse and return the structured response
        result = json.loads(response.choices[0].message.content)
        return result