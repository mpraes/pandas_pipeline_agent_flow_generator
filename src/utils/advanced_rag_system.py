# src/utils/advanced_rag_system.py
"""
Advanced RAG System - Combines semantic search with traditional keyword search
Uses embeddings for better document retrieval and relevance scoring
"""

import json
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path

# === CONFIGURATION ===

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, good quality
    vector_db_path: str = "data/vector_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 10
    similarity_threshold: float = 0.3
    use_reranking: bool = True
    max_context_length: int = 4000

class DocumentChunk(NamedTuple):
    """Represents a chunk of a document"""
    content: str
    source: str
    chunk_id: int
    metadata: Dict[str, Any]

class RetrievalResult(NamedTuple):
    """Result from document retrieval"""
    chunks: List[DocumentChunk]
    scores: List[float]
    total_retrieved: int

# === PANDAS DOCUMENTATION KNOWLEDGE BASE ===

PANDAS_DOCS_CORPUS = [
    {
        "content": """
        pandas.DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None)
        
        Fill NA/NaN values using the specified method.
        
        Parameters:
        - value: scalar, dict, Series, or DataFrame - Value to use to fill holes
        - method: {'backfill', 'bfill', 'pad', 'ffill', None} - Method to use for filling holes
        - axis: {0 or 'index', 1 or 'columns'} - Axis along which to fill missing values
        - inplace: bool - If True, fill in-place
        - limit: int - Maximum number of consecutive NaN values to forward/backward fill
        
        Examples:
        >>> df.fillna(0)  # Fill with zero
        >>> df.fillna(method='ffill')  # Forward fill
        >>> df.fillna(df.mean())  # Fill with mean
        """,
        "source": "pandas_docs_fillna",
        "title": "pandas.DataFrame.fillna",
        "category": "missing_values"
    },
    {
        "content": """
        pandas.DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
        
        Return DataFrame with duplicate rows removed.
        
        Parameters:
        - subset: column label or sequence of labels - Only consider certain columns for identifying duplicates
        - keep: {'first', 'last', False} - Determines which duplicates to keep
        - inplace: bool - Whether to drop duplicates in place or return a copy
        - ignore_index: bool - If True, the resulting axis will be labeled 0, 1, …, n - 1
        
        Examples:
        >>> df.drop_duplicates()  # Remove all duplicate rows
        >>> df.drop_duplicates(['col1', 'col2'])  # Based on specific columns
        >>> df.drop_duplicates(keep='last')  # Keep last occurrence
        """,
        "source": "pandas_docs_drop_duplicates", 
        "title": "pandas.DataFrame.drop_duplicates",
        "category": "duplicates"
    },
    {
        "content": """
        pandas.to_datetime(arg, errors='raise', dayfirst=False, yearfirst=False, utc=None, format=None)
        
        Convert argument to datetime.
        
        Parameters:
        - arg: str, numeric, array-like, Series, DataFrame/dict-like - The object to convert to a datetime
        - errors: {'ignore', 'raise', 'coerce'} - How to handle parsing errors
        - dayfirst: bool - Specify a date parse order if arg is str or its list-likes
        - yearfirst: bool - Specify a date parse order if arg is str or its list-likes
        - format: str - The strftime to parse time, eg "%d/%m/%Y"
        
        Examples:
        >>> pd.to_datetime('2023-01-01')
        >>> pd.to_datetime(['2023-01-01', '2023-01-02'])
        >>> pd.to_datetime('01/01/2023', format='%d/%m/%Y')
        """,
        "source": "pandas_docs_to_datetime",
        "title": "pandas.to_datetime", 
        "category": "datetime"
    },
    {
        "content": """
        pandas.DataFrame.astype(dtype, copy=True, errors='raise')
        
        Cast a pandas object to a specified dtype.
        
        Parameters:
        - dtype: data type, or dict of column name -> data type
        - copy: bool - Return a copy when copy=True
        - errors: {'raise', 'ignore'} - Control raising of exceptions on invalid data for provided dtype
        
        Examples:
        >>> df.astype('int32')  # Convert all columns
        >>> df.astype({'col1': 'int32', 'col2': 'float64'})  # Specific columns
        >>> df['col'].astype('category')  # Convert to category
        """,
        "source": "pandas_docs_astype",
        "title": "pandas.DataFrame.astype",
        "category": "data_types"
    },
    {
        "content": """
        pandas.DataFrame.rename(mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False)
        
        Alter axes labels.
        
        Parameters:
        - mapper: dict-like or function - Dict-like or functions transformations to apply
        - index: dict-like or function - Alternative to specifying axis
        - columns: dict-like or function - Alternative to specifying axis
        - axis: {0 or 'index', 1 or 'columns'} - Axis to target
        - inplace: bool - Whether to return a new DataFrame
        
        Examples:
        >>> df.rename(columns={'old_name': 'new_name'})
        >>> df.rename(str.lower, axis='columns')  # Apply function
        >>> df.rename(index={0: 'first', 1: 'second'})
        """,
        "source": "pandas_docs_rename",
        "title": "pandas.DataFrame.rename",
        "category": "column_operations"
    },
    {
        "content": """
        Data cleaning best practices with pandas:
        
        1. Handle Missing Values:
           - Identify: df.isnull().sum()
           - Fill: df.fillna(value)
           - Drop: df.dropna()
        
        2. Remove Duplicates:
           - Check: df.duplicated().sum()
           - Remove: df.drop_duplicates()
        
        3. Data Type Conversion:
           - Check: df.dtypes
           - Convert: df.astype()
           - Numeric: pd.to_numeric()
           - Datetime: pd.to_datetime()
        
        4. String Cleaning:
           - Strip: df['col'].str.strip()
           - Case: df['col'].str.lower()
           - Replace: df['col'].str.replace()
        
        5. Outlier Detection:
           - Statistical: df.describe()
           - IQR method: Q1, Q3 = df.quantile([0.25, 0.75])
        """,
        "source": "pandas_best_practices",
        "title": "Pandas Data Cleaning Best Practices",
        "category": "best_practices"
    },
    {
        "content": r"""
        pandas.DataFrame.str accessor for string operations:
        
        Common string cleaning operations:
        - str.strip(): Remove leading and trailing whitespace
        - str.lower() / str.upper(): Change case
        - str.replace(pat, repl): Replace patterns
        - str.contains(pat): Check if pattern exists
        - str.split(pat): Split strings
        - str.extract(pat): Extract groups from regex
        
        Examples:
        >>> df['name'].str.strip()  # Remove whitespace
        >>> df['name'].str.lower()  # Convert to lowercase
        >>> df['phone'].str.replace(r'[^\d]', '', regex=True)  # Keep only digits
        >>> df['email'].str.contains('@')  # Check valid emails
        """,
        "source": "pandas_string_operations",
        "title": "Pandas String Operations",
        "category": "string_cleaning"
    }
]

# === ADVANCED RAG SYSTEM ===

class AdvancedRAGSystem:
    """
    Advanced RAG system combining semantic search with traditional methods
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embedding_model = None
        self.vector_index = None
        self.documents = []
        self.chunks = []
        
        # Initialize paths
        self.vector_db_path = Path(self.config.vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create the knowledge base
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system with embeddings and vector store"""
        print("🚀 Initializing Advanced RAG System...")
        
        # Load embedding model
        try:
            print(f"📥 Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            print("💡 Falling back to basic search...")
            return
        
        # Check if vector store exists
        index_path = self.vector_db_path / "faiss_index.bin"
        chunks_path = self.vector_db_path / "chunks.pkl"
        
        if index_path.exists() and chunks_path.exists():
            print("📂 Loading existing vector store...")
            self._load_vector_store()
        else:
            print("🔨 Creating new vector store...")
            self._create_vector_store()
    
    def _create_vector_store(self):
        """Create vector store from pandas documentation"""
        print("📚 Processing pandas documentation...")
        
        # Create chunks from documentation
        self.chunks = []
        for doc in PANDAS_DOCS_CORPUS:
            chunk = DocumentChunk(
                content=doc["content"],
                source=doc["source"],
                chunk_id=len(self.chunks),
                metadata={
                    "title": doc["title"],
                    "category": doc["category"]
                }
            )
            self.chunks.append(chunk)
        
        if not self.embedding_model:
            print("⚠️ No embedding model available, skipping vector creation")
            return
        
        # Generate embeddings
        print("🔢 Generating embeddings...")
        texts = [chunk.content for chunk in self.chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        print("🗂️ Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.vector_index.add(embeddings.astype('float32'))
        
        # Save vector store
        self._save_vector_store()
        print(f"✅ Vector store created with {len(self.chunks)} chunks")
    
    def _save_vector_store(self):
        """Save vector store to disk"""
        index_path = self.vector_db_path / "faiss_index.bin"
        chunks_path = self.vector_db_path / "chunks.pkl"
        
        faiss.write_index(self.vector_index, str(index_path))
        
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"💾 Vector store saved to {self.vector_db_path}")
    
    def _load_vector_store(self):
        """Load vector store from disk"""
        index_path = self.vector_db_path / "faiss_index.bin"
        chunks_path = self.vector_db_path / "chunks.pkl"
        
        try:
            self.vector_index = faiss.read_index(str(index_path))
            
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"✅ Vector store loaded with {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            print("🔨 Creating new vector store...")
            self._create_vector_store()
    
    def semantic_search(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Perform semantic search using embeddings
        """
        if not self.embedding_model or not self.vector_index:
            print("⚠️ Semantic search not available, using fallback")
            return self._fallback_search(query, top_k)
        
        top_k = top_k or self.config.top_k_retrieval
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search vector index
        scores, indices = self.vector_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Filter by similarity threshold
        valid_results = []
        valid_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.config.similarity_threshold:
                valid_results.append(self.chunks[idx])
                valid_scores.append(float(score))
        
        return RetrievalResult(
            chunks=valid_results,
            scores=valid_scores,
            total_retrieved=len(valid_results)
        )
    
    def _fallback_search(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Fallback search using keyword matching when embeddings fail
        """
        top_k = top_k or self.config.top_k_retrieval
        query_words = set(query.lower().split())
        
        results = []
        for chunk in self.chunks:
            content_words = set(chunk.content.lower().split())
            
            # Simple keyword overlap score
            overlap = len(query_words.intersection(content_words))
            score = overlap / max(len(query_words), 1)
            
            if score > 0:
                results.append((chunk, score))
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        
        chunks = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        return RetrievalResult(
            chunks=chunks,
            scores=scores,
            total_retrieved=len(chunks)
        )
    
    def hybrid_search(self, query: str, top_k: int = None) -> RetrievalResult:
        """
        Combine semantic search with keyword matching for better results
        """
        top_k = top_k or self.config.top_k_retrieval
        
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k * 2)  # Get more for reranking
        
        # Get keyword results
        keyword_results = self._fallback_search(query, top_k)
        
        # Combine and deduplicate
        combined_chunks = {}
        
        # Add semantic results with weight
        for chunk, score in zip(semantic_results.chunks, semantic_results.scores):
            combined_chunks[chunk.chunk_id] = (chunk, score * 0.7)  # 70% weight for semantic
        
        # Add keyword results with weight
        for chunk, score in zip(keyword_results.chunks, keyword_results.scores):
            if chunk.chunk_id in combined_chunks:
                # Combine scores
                existing_chunk, existing_score = combined_chunks[chunk.chunk_id]
                new_score = existing_score + (score * 0.3)  # 30% weight for keyword
                combined_chunks[chunk.chunk_id] = (existing_chunk, new_score)
            else:
                combined_chunks[chunk.chunk_id] = (chunk, score * 0.3)
        
        # Sort by combined score
        sorted_results = sorted(combined_chunks.values(), key=lambda x: x[1], reverse=True)
        sorted_results = sorted_results[:top_k]
        
        final_chunks = [r[0] for r in sorted_results]
        final_scores = [r[1] for r in sorted_results]
        
        return RetrievalResult(
            chunks=final_chunks,
            scores=final_scores,
            total_retrieved=len(final_chunks)
        )
    
    def get_context_for_query(self, query: str, max_length: int = None) -> str:
        """
        Get formatted context for a query to inject into LLM prompt
        """
        max_length = max_length or self.config.max_context_length
        
        # Get relevant chunks with more permissive search
        results = self.hybrid_search(query)
        
        if not results.chunks:
            # Try fallback search with even lower threshold
            try:
                results = self._fallback_search(query, top_k=10)
            except Exception as e:
                print(f"Fallback search also failed: {e}")
                return "No relevant documentation found."
        
        if not results.chunks:
            return "No relevant documentation found."
        
        # Format context - be more permissive with score filtering
        context_parts = []
        current_length = 0
        
        for chunk, score in zip(results.chunks, results.scores):
            # Accept chunks with lower similarity scores for better recall
            if score < 0.05:  # Very permissive threshold
                continue
                
            chunk_text = f"[{chunk.metadata.get('title', 'Unknown')}]\n{chunk.content}\n"
            
            if current_length + len(chunk_text) > max_length:
                break
                
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        if not context_parts:
            # If still no context, return at least one chunk regardless of score
            chunk = results.chunks[0]
            chunk_text = f"[{chunk.metadata.get('title', 'Unknown')}]\n{chunk.content}\n"
            return chunk_text
        
        return "\n---\n".join(context_parts)

# === INTEGRATION FUNCTIONS ===

def create_rag_system() -> AdvancedRAGSystem:
    """
    Factory function to create and initialize the RAG system
    """
    config = RAGConfig(
        embedding_model="all-MiniLM-L6-v2",  # Fast and good quality
        top_k_retrieval=8,  # Increased from 5
        similarity_threshold=0.1,  # Lowered from 0.2 for more recall
        max_context_length=3000
    )
    
    return AdvancedRAGSystem(config)

def enhanced_crag_with_rag(queries: List[str], user_instructions: str) -> Tuple[List[str], bool]:
    """
    Enhanced CRAG using proper RAG instead of Tavily + LLM evaluation
    """
    print("🚀 Using Advanced RAG for document retrieval")
    
    try:
        # Initialize RAG system
        rag_system = create_rag_system()
        
        relevant_docs = []
        
        for query in queries[:5]:  # Limit queries for efficiency
            print(f"🔍 RAG search: {query}")
            
            # Get context using RAG
            context = rag_system.get_context_for_query(query, max_length=1000)
            
            if context and context != "No relevant documentation found.":
                relevant_docs.append(f"Query: {query}\n{context}")
                print(f"✅ Found relevant context for: {query}")
            else:
                print(f"❌ No context found for: {query}")
        
        has_relevant_docs = len(relevant_docs) > 0
        
        if has_relevant_docs:
            print(f"✅ Advanced RAG found {len(relevant_docs)} relevant contexts")
        else:
            print("❌ Advanced RAG found no relevant contexts")
        
        return relevant_docs, has_relevant_docs
        
    except Exception as e:
        print(f"❌ Advanced RAG failed: {e}")
        print("🔄 Falling back to basic search...")
        return [], False

# === TEST FUNCTION ===

def test_advanced_rag():
    """
    Test function for the advanced RAG system
    """
    print("🧪 Testing Advanced RAG System")
    print("=" * 50)
    
    rag_system = create_rag_system()
    
    test_queries = [
        "how to handle missing values",
        "remove duplicate rows",
        "convert string to datetime", 
        "rename columns pandas",
        "data cleaning best practices"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = rag_system.hybrid_search(query, top_k=3)
        
        print(f"📊 Found {results.total_retrieved} relevant chunks")
        
        for i, (chunk, score) in enumerate(zip(results.chunks, results.scores)):
            print(f"  {i+1}. {chunk.metadata['title']} (score: {score:.3f})")
        
        # Get formatted context
        context = rag_system.get_context_for_query(query, max_length=500)
        print(f"📝 Context preview: {context[:200]}...")

if __name__ == "__main__":
    test_advanced_rag()