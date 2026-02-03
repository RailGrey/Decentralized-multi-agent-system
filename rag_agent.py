"""
Single RAG-based AI Agent using Mistral API with structured action-based responses.
Uses Qdrant for vector storage and retrieval.
"""

from mistralai import Mistral
from typing import List, Dict, Any, Optional
from settings.prompts import SOLO_AGENT_SYSTEM_PROMPT
import numpy as np
import json
import hashlib
import os
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


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
        cache_dir: str = None,
        qdrant_path: str = "./qdrant_storage",
        qdrant_url: str = None,
        embedding_dim: int = 1024
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
            qdrant_path: Path for local Qdrant storage (used if qdrant_url is None)
            qdrant_url: URL for Qdrant server (e.g., "http://localhost:6333")
            embedding_dim: Dimension of embeddings (1024 for mistral-embed)
        """
        self.name = name
        self.expertise = expertise
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.available_agents = available_agents or []
        self.client = Mistral(api_key=api_key)
        self.response_schema = self._build_response_schema()
        
        # Cache setup
        self.cache_dir = cache_dir
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Qdrant setup
        self.collection_name = f"{name.replace(' ', '_').lower()}_knowledge"
        self.embedding_dim = embedding_dim
        
        # Initialize Qdrant client
        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url)
        else:
            self.qdrant_client = QdrantClient(path=qdrant_path)
        
        # Create collection if it doesn't exist
        self._initialize_collection()
        
        # Counter for point IDs
        self._point_counter = self._get_max_point_id() + 1
    
    def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
    
    def _get_max_point_id(self) -> int:
        """Get the maximum point ID in the collection."""
        try:
            # Scroll through all points to find max ID
            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=False,
                with_vectors=False
            )
            if points:
                return max(point.id for point in points)
            return 0
        except Exception:
            return 0
    
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
        """Build JSON schema for single-agent solve (when not part of multi-agent system)."""
        return {
            "type": "object",
            "properties": {
                "solution": {
                    "type": "string",
                    "description": "Complete, runnable solution code"
                },
                "description": {
                    "type": "string",
                    "description": "Clear explanation of the approach and how it works"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Your confidence in this solution (0.0-1.0)"
                }
            },
            "required": ["solution", "description", "confidence"],
            "additionalProperties": False
        }
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to the agent's knowledge base (Qdrant)."""
        # Get embedding (from cache or API)
        embedding = self._get_embedding(content)
        
        # Save metadata to cache if provided
        if self.cache_dir and metadata:
            self._save_cached_embedding(content, embedding, metadata)
        
        # Prepare metadata for Qdrant
        payload = {
            "content": content,
            "agent_name": self.name,
            **(metadata or {})
        }
        
        # Add to Qdrant
        point_id = self._point_counter
        self._point_counter += 1
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from knowledge base using Qdrant semantic search."""
        # Check if collection has any points
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            if collection_info.points_count == 0:
                return ""
        except Exception:
            return ""
        
        # Get query embedding (from cache or API)
        query_embedding = self._get_embedding(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k
        )
        
        if not search_results:
            return ""
        
        # Format context with both description and solution
        context = "Relevant knowledge:\n\n"
        for i, hit in enumerate(search_results.points, 1):
            content = hit.payload.get("content", "")
            solution = hit.payload.get("solution", "")
            
            context += f"[{i}] Problem:\n{content}\n"
            if solution:
                context += f"\nSolution:\n{solution}\n"
            context += "\n"
        
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
        Solve a coding task directly (single-agent mode).
        
        This method is for standalone use when the agent is NOT part of a multi-agent system.
        For multi-agent systems, use the BrainstormSolver or call() method instead.
        
        Args:
            task: The coding task description
            
        Returns:
            {
                "solution": "<complete code>",
                "description": "<explanation>",
                "confidence": 0.85
            }
        """
        # Retrieve relevant context
        context = self.retrieve_context(task)
        
        # Build system prompt for direct solving
        system_prompt = SOLO_AGENT_SYSTEM_PROMPT.substitute(
            agent_name=self.name,
            agent_expertise=self.expertise
        )
        
        user_prompt = f"{context}\n\nTask: {task}\n\nProvide your solution."
        
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
                    "name": "direct_solution",
                    "schema": self.response_schema,
                    "strict": True
                }
            }
        )
        
        # Parse and return the structured response
        result = json.loads(response.choices[0].message.content)
        return result
    
    # ── Qdrant-specific methods ─────────────────────────────────────────
    
    def search_by_metadata(
        self,
        query: str,
        metadata_filter: Dict[str, Any],
        top_k: int = 3
    ) -> str:
        """
        Search with metadata filtering.
        
        Args:
            query: Search query text
            metadata_filter: Dict of metadata field -> value to filter by
            top_k: Number of results to return
            
        Returns:
            Formatted context string
        """
        query_embedding = self._get_embedding(query)
        
        # Build Qdrant filter
        conditions = []
        for key, value in metadata_filter.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=conditions) if conditions else None,
            limit=top_k
        )
        
        if not search_results:
            return ""
        
        context = "Relevant knowledge:\n\n"
        for i, hit in enumerate(search_results, 1):
            content = hit.payload.get("content", "")
            context += f"[{i}] {content}\n\n"
        
        return context
    
    def get_all_knowledge(self) -> List[Dict[str, Any]]:
        """Retrieve all knowledge from Qdrant collection."""
        points, _ = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        return [point.payload for point in points]
    
    def clear_knowledge(self):
        """Clear all knowledge from the collection."""
        self.qdrant_client.delete_collection(self.collection_name)
        self._initialize_collection()
        self._point_counter = 1
    
    def get_knowledge_count(self) -> int:
        """Get the number of knowledge items stored."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0