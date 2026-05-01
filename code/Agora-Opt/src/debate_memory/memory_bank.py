"""
Memory Bank for storing and retrieving successful problem-solving cases
Uses LlamaIndex for RAG-based case retrieval
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PKG_DIR.parent.parent
DEFAULT_MEMORY_DIR = str(_PROJECT_ROOT / "memory_storage")


class MemoryBank:
    """
    Memory Bank for storing successful problem-solving experiences
    
    Design inspired by Memento (https://arxiv.org/pdf/2508.16153):
    - Episodic memory: Store past successful trajectories
    - Case-based reasoning: Retrieve similar cases to guide current problem
    - Non-parametric: No gradient updates, just memory read/write
    """
    
    def __init__(self, memory_dir: str = DEFAULT_MEMORY_DIR, embedding_model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize Memory Bank
        
        Args:
            memory_dir: Directory to store memory index and cases
            embedding_model: HuggingFace embedding model name or local path
        """
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        self.cases_file = os.path.join(memory_dir, "cases.jsonl")
        self.index_dir = os.path.join(memory_dir, "index")
        
        # Configure embedding model with local caching
        # Set cache_folder to use llama_index's cache directory
        # Set trust_remote_code to False for security
        # If embedding_model is a local path, use it directly
        # Otherwise, try to use cached model to avoid network requests
        os.environ.setdefault("HF_HUB_OFFLINE", "0")  # Allow online access by default
        
        # Check if embedding_model is a local file path
        is_local_path = os.path.isabs(embedding_model) or (os.path.sep in embedding_model and os.path.exists(embedding_model))
        
        try:
            # If it's a local path, use it directly
            if is_local_path:
                print(f"📁 Using local embedding model from: {embedding_model}")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=embedding_model,
                    cache_folder=os.path.expanduser("~/.cache/llama_index"),
                    trust_remote_code=False
                )
            else:
                # Try to load from cache first to avoid network requests
                # Set HF_HUB_OFFLINE=1 to force local-only mode
                print(f"🔍 Loading embedding model: {embedding_model}")
                print("   (If you want to avoid Hugging Face downloads, set HF_HUB_OFFLINE=1 or use a local model path)")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=embedding_model,
                    cache_folder=os.path.expanduser("~/.cache/llama_index"),
                    trust_remote_code=False
                )
        except Exception as e:
            # If model loading fails, try to use cached model only
            print(f"⚠️  Warning: Failed to load embedding model '{embedding_model}': {e}")
            print("   Attempting to use cached model only (setting HF_HUB_OFFLINE=1)...")
            os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=embedding_model,
                    cache_folder=os.path.expanduser("~/.cache/llama_index"),
                    trust_remote_code=False
                )
                print("   ✅ Using cached model")
            except Exception as e2:
                print(f"❌ Error: Could not load embedding model: {e2}")
                print("   Please either:")
                print("   1. Download the model first: python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')\"")
                print("   2. Set HF_HUB_OFFLINE=1 and ensure the model is cached")
                print("   3. Use a local model path: --embedding_model /path/to/local/model")
                raise
        # Disable chunking to ensure one document = one node (no duplicates)
        Settings.chunk_size = 8192  # Large enough to never split
        Settings.chunk_overlap = 0
        
        # Load or create index
        self.index = self._load_or_create_index()
        self.case_count = self._count_cases()
        
        print(f"Memory Bank initialized with {self.case_count} cases")
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
                index = load_index_from_storage(storage_context)
                print(f"Loaded existing memory index from {self.index_dir}")
                return index
            except:
                print("Failed to load index, creating new one")
        
        # Create new empty index
        documents = []
        index = VectorStoreIndex.from_documents(documents)
        os.makedirs(self.index_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=self.index_dir)
        print(f"Created new memory index at {self.index_dir}")
        return index
    
    def _count_cases(self) -> int:
        """Count number of cases in memory"""
        if not os.path.exists(self.cases_file):
            return 0
        with open(self.cases_file, 'r') as f:
            return sum(1 for _ in f)
    
    def add_case(self, problem_id: int, problem_desc: str, solution_code: str, 
                 objective_value: float, is_correct: bool, metadata: Optional[Dict] = None):
        """
        Add a successful case to memory
        
        Args:
            problem_id: Problem ID
            problem_desc: Problem description
            solution_code: Solution code
            objective_value: Computed objective value
            is_correct: Whether the solution is correct
            metadata: Additional metadata (model, debate_rounds, etc.)
        """
        if not is_correct:
            # Only store successful cases
            return
        
        case = {
            'problem_id': problem_id,
            'description': problem_desc,
            'solution_code': solution_code,
            'objective_value': objective_value,
            'is_correct': is_correct,
            'metadata': metadata or {}
        }
        
        # Write to cases file
        with open(self.cases_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        # Create document for indexing
        # Combine description and key solution insights for better retrieval
        doc_text = f"""Problem: {problem_desc}

Solution approach:
{solution_code[:500]}...

Key features:
- Problem ID: {problem_id}
- Objective value: {objective_value}
- Status: Correct
"""
        
        doc = Document(
            text=doc_text,
            metadata={
                'problem_id': problem_id,
                'objective_value': objective_value,
                **case['metadata']
            }
        )
        
        # Add to index
        self.index.insert(doc)
        self.index.storage_context.persist(persist_dir=self.index_dir)
        
        self.case_count += 1
        print(f"✅ Added case {problem_id} to memory (Total: {self.case_count})")
    
    def retrieve_similar_cases(self, query: str, top_k: int = 3, preferred_dataset: Optional[str] = None) -> List[Dict]:
        """
        Retrieve similar cases from memory using RAG based on semantic similarity
        
        Args:
            query: Query text (usually the problem description)
            top_k: Number of similar cases to retrieve (0 = no retrieval)
            preferred_dataset: Preferred dataset name to prioritize (optional)
        
        Returns:
            List of similar cases with scores, sorted by semantic similarity
        """
        if self.case_count == 0 or top_k <= 0:
            return []
        
        # Query the index - purely based on semantic similarity
        retriever = self.index.as_retriever(similarity_top_k=top_k * 2 if preferred_dataset else top_k)
        nodes = retriever.retrieve(query)
        
        # Load corresponding cases from cases.jsonl based on semantic similarity
        similar_cases = []
        seen_keys = set()  # Track which (problem_id, dataset) combinations we've added
        
        # If preferred_dataset is specified, prioritize those cases
        preferred_cases = []
        other_cases = []
        
        for node in nodes:
            problem_id = node.metadata.get('problem_id')
            score = node.score
            node_dataset = node.metadata.get('dataset', '')
            
            # Build key for deduplication
            case_key = (problem_id, node_dataset)
            if case_key in seen_keys:
                continue
            
            # Load the case - use dataset from node metadata to get the exact match
            case_data = None
            if node_dataset:
                # Try to load by problem_id and dataset (more precise)
                case_data = self._load_case_by_id_and_dataset(problem_id, node_dataset)
            
            if not case_data:
                # Fallback: try to load by problem_id only
                case_data = self._load_case_by_id(problem_id)
            
            if case_data:
                seen_keys.add(case_key)
                case_item = {
                    'case': case_data,
                    'score': score,
                    'text_preview': node.text[:200]
                }
                
                # Separate preferred dataset cases from others
                if preferred_dataset and node_dataset == preferred_dataset:
                    preferred_cases.append(case_item)
                else:
                    other_cases.append(case_item)
        
        # Combine: preferred cases first, then others, all sorted by similarity score
        similar_cases = preferred_cases + other_cases
        
        # Return top_k results
        return similar_cases[:top_k]
    
    def _load_case_by_id(self, problem_id: int) -> Optional[Dict]:
        """Load a specific case by problem ID (returns first match)"""
        if not os.path.exists(self.cases_file):
            return None
        
        with open(self.cases_file, 'r', encoding='utf-8') as f:
            for line in f:
                case = json.loads(line)
                if case['problem_id'] == problem_id:
                    return case
        return None
    
    def _load_case_by_id_and_dataset(self, problem_id: int, dataset: str) -> Optional[Dict]:
        """Load a specific case by problem ID and dataset"""
        if not os.path.exists(self.cases_file):
            return None
        
        with open(self.cases_file, 'r', encoding='utf-8') as f:
            for line in f:
                case = json.loads(line)
                if case['problem_id'] == problem_id:
                    case_dataset = case.get('metadata', {}).get('dataset', '')
                    if case_dataset == dataset:
                        return case
        return None
    
    def get_memory_stats(self) -> Dict:
        """Get memory bank statistics"""
        return {
            'total_cases': self.case_count,
            'memory_dir': self.memory_dir,
            'cases_file': self.cases_file,
            'index_dir': self.index_dir
        }
    
    def format_retrieved_cases_for_prompt(self, cases: List[Dict]) -> str:
        """
        Format retrieved cases for inclusion in LLM prompt
        
        Args:
            cases: List of retrieved cases
        
        Returns:
            Formatted string for prompt
        """
        if not cases:
            return ""
        
        prompt = "# Retrieved Similar Cases from Memory\n\n"
        prompt += "The following successful cases from previous problems might be relevant:\n\n"
        
        for i, item in enumerate(cases, 1):
            case = item['case']
            score = item['score']
            
            prompt += f"## Case {i} (Similarity: {score:.3f})\n"
            prompt += f"**Problem:** {case['description']}\n\n"
            prompt += f"**Solution approach:**\n```python\n{case['solution_code']}\n```\n\n"
            prompt += f"**Result:** Objective value = {case['objective_value']}, Status = Correct\n\n"
            prompt += "---\n\n"
        
        return prompt


