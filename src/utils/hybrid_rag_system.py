# src/utils/hybrid_rag_system.py
"""
Hybrid RAG System that combines sparse and dense search for better documentation retrieval.
Ideal for retrieving technical documentation for pandas and other libraries.
"""

from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# === TYPES AND CONFIGURATIONS ===

class SearchMethod(Enum):
    SPARSE = "sparse"           # Keyword-based search (TF-IDF, BM25)
    DENSE = "dense"             # Semantic embedding-based search  
    HYBRID = "hybrid"           # Combination of both methods
    
class RetrievalStrategy(Enum):
    DOCUMENTATION = "documentation"  # For technical docs (pandas, etc)
    CODE_EXAMPLES = "code_examples"  # For code examples
    BEST_PRACTICES = "best_practices" # For best practices
    TROUBLESHOOTING = "troubleshooting" # For problem resolution

@dataclass
class SearchResult:
    content: str
    title: str
    url: str
    sparse_score: float
    dense_score: Optional[float]
    hybrid_score: float
    source_type: str
    metadata: Dict

@dataclass
class HybridSearchConfig:
    sparse_weight: float = 0.6      # Weight for sparse search
    dense_weight: float = 0.4       # Weight for dense search
    min_sparse_score: float = 5.0   # Minimum score for sparse search
    min_dense_score: float = 0.7    # Minimum score for dense search
    max_results: int = 10           # Maximum results per search
    rerank_top_k: int = 15          # Top K for reranking
    use_query_expansion: bool = True # Automatically expand queries

# === INTELLIGENT QUERY EXPANSION ===

def expand_pandas_query(original_query: str, strategy: RetrievalStrategy) -> List[str]:
    """
    Expands an original query to improve pandas documentation retrieval
    """
    base_queries = [original_query]
    
    # Strategy-specific expansions
    if strategy == RetrievalStrategy.DOCUMENTATION:
        expansions = _expand_for_documentation(original_query)
    elif strategy == RetrievalStrategy.CODE_EXAMPLES:
        expansions = _expand_for_code_examples(original_query)
    elif strategy == RetrievalStrategy.BEST_PRACTICES:
        expansions = _expand_for_best_practices(original_query)
    else:
        expansions = _expand_for_troubleshooting(original_query)
    
    return base_queries + expansions

def _expand_for_documentation(query: str) -> List[str]:
    """Expansions focused on official documentation"""
    expansions = []
    
    # Map common operations to documentation terms
    operation_mappings = {
        "convert object to datetime": [
            "pandas.to_datetime",
            "datetime conversion pandas",
            "parse dates pandas",
            "convert string to datetime"
        ],
        "fillna strategies": [
            "pandas.DataFrame.fillna",
            "handle missing data pandas", 
            "missing value imputation",
            "forward fill backward fill"
        ],
        "missing values categorical": [
            "categorical data missing values",
            "pandas.Categorical fillna",
            "impute categorical pandas"
        ],
        "standardize categorical": [
            "pandas.Categorical",
            "category dtype pandas",
            "categorical data preprocessing"
        ],
        "normalize numerical": [
            "pandas normalization",
            "sklearn StandardScaler",
            "min-max scaling pandas",
            "feature scaling"
        ],
        "remove duplicates": [
            "pandas.DataFrame.drop_duplicates",
            "duplicate removal pandas",
            "unique values pandas"
        ],
        "rename columns": [
            "pandas.DataFrame.rename",
            "column names pandas",
            "rename multiple columns"
        ]
    }
    
    # Search for partial matches
    for operation, terms in operation_mappings.items():
        if any(word in query.lower() for word in operation.split()):
            expansions.extend(terms)
    
    return expansions[:3]  # Limit to 3 expansions per query

def _expand_for_code_examples(query: str) -> List[str]:
    """Expansions focused on practical examples"""
    return [
        f"{query} example",
        f"{query} code snippet",
        f"how to {query}",
        f"{query} tutorial"
    ]

def _expand_for_best_practices(query: str) -> List[str]:
    """Expansions focused on best practices"""
    return [
        f"{query} best practices",
        f"{query} performance",
        f"{query} efficient way",
        f"{query} recommended approach"
    ]

def _expand_for_troubleshooting(query: str) -> List[str]:
    """Expansions focused on problem resolution"""
    return [
        f"{query} error",
        f"{query} troubleshooting",
        f"{query} common issues",
        f"{query} debugging"
    ]

# === HYBRID SEARCH SYSTEM ===

class HybridRAGSearcher:
    """
    Hybrid search system that combines sparse and dense methods
    """
    
    def __init__(self, config: HybridSearchConfig = None):
        self.config = config or HybridSearchConfig()
        self.sparse_searcher = None  # Implement with BM25 or similar
        self.dense_searcher = None   # Implement with embeddings
        
    async def search(self, 
                    queries: List[str], 
                    strategy: RetrievalStrategy = RetrievalStrategy.DOCUMENTATION,
                    method: SearchMethod = SearchMethod.HYBRID) -> List[SearchResult]:
        """
        Executes hybrid search for a list of queries
        """
        all_results = []
        
        for query in queries:
            # Expand query if enabled
            if self.config.use_query_expansion:
                expanded_queries = expand_pandas_query(query, strategy)
            else:
                expanded_queries = [query]
            
            # Execute search for each expanded query
            for expanded_query in expanded_queries:
                if method == SearchMethod.SPARSE:
                    results = await self._sparse_search(expanded_query)
                elif method == SearchMethod.DENSE:
                    results = await self._dense_search(expanded_query)
                else:  # HYBRID
                    results = await self._hybrid_search(expanded_query)
                
                all_results.extend(results)
        
        # Remove duplicates and reorder
        deduplicated_results = self._deduplicate_results(all_results)
        final_results = self._rerank_results(deduplicated_results)
        
        return final_results[:self.config.max_results]
    
    async def _sparse_search(self, query: str) -> List[SearchResult]:
        """
        Sparse search using TF-IDF/BM25
        Simulates integration with Tavily or similar system
        """
        # This would be the real integration with the sparse search system
        # For now, simulates results
        
        # Here you would integrate with:
        # - Tavily API (current)
        # - Elasticsearch with BM25
        # - Whoosh
        # - Or other sparse search system
        
        mock_results = [
            SearchResult(
                content=f"Documentation for {query}",
                title=f"Pandas: {query}",
                url=f"https://pandas.pydata.org/docs/{query.replace(' ', '-')}",
                sparse_score=8.5,
                dense_score=None,
                hybrid_score=8.5,
                source_type="documentation",
                metadata={"method": "sparse"}
            )
        ]
        
        return [r for r in mock_results if r.sparse_score >= self.config.min_sparse_score]
    
    async def _dense_search(self, query: str) -> List[SearchResult]:
        """
        Dense search using semantic embeddings
        """
        # This would be the integration with an embedding-based search system
        # Possible implementations:
        # - OpenAI Embeddings + FAISS/Chroma
        # - Sentence Transformers + Vector DB
        # - Cohere/Anthropic embeddings
        
        mock_results = [
            SearchResult(
                content=f"Semantic match for {query}",
                title=f"Related: {query}",
                url=f"https://stackoverflow.com/questions/{query.replace(' ', '-')}",
                sparse_score=0.0,
                dense_score=0.85,
                hybrid_score=0.85,
                source_type="community",
                metadata={"method": "dense"}
            )
        ]
        
        return [r for r in mock_results if r.dense_score >= self.config.min_dense_score]
    
    async def _hybrid_search(self, query: str) -> List[SearchResult]:
        """
        Combines sparse and dense search
        """
        # Execute both searches in parallel
        sparse_task = asyncio.create_task(self._sparse_search(query))
        dense_task = asyncio.create_task(self._dense_search(query))
        
        sparse_results, dense_results = await asyncio.gather(sparse_task, dense_task)
        
        # Combine and calculate hybrid scores
        combined_results = []
        
        # Add sparse results
        for result in sparse_results:
            result.hybrid_score = (
                self.config.sparse_weight * result.sparse_score + 
                self.config.dense_weight * (result.dense_score or 0)
            )
            combined_results.append(result)
        
        # Add dense results (avoid duplicates)
        existing_urls = {r.url for r in sparse_results}
        for result in dense_results:
            if result.url not in existing_urls:
                result.hybrid_score = (
                    self.config.sparse_weight * result.sparse_score + 
                    self.config.dense_weight * result.dense_score
                )
                combined_results.append(result)
        
        return combined_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL
        """
        seen_urls = set()
        deduplicated = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                deduplicated.append(result)
        
        return deduplicated
    
    def _rerank_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Reorder results using hybrid score and other factors
        """
        def rerank_score(result: SearchResult) -> float:
            base_score = result.hybrid_score
            
            # Boost for official documentation
            if "pandas.pydata.org" in result.url:
                base_score *= 1.3
            elif "docs.python.org" in result.url:
                base_score *= 1.2
            
            # Boost for practical examples
            if any(keyword in result.title.lower() for keyword in ["example", "tutorial", "how to"]):
                base_score *= 1.1
            
            # Penalty for very old results (if we had date metadata)
            # if result.metadata.get('age_months', 0) > 24:
            #     base_score *= 0.9
            
            return base_score
        
        return sorted(results, key=rerank_score, reverse=True)

# === CONVENIENCE FUNCTIONS ===

async def search_pandas_documentation(queries: List[str], 
                                    strategy: RetrievalStrategy = RetrievalStrategy.DOCUMENTATION,
                                    config: HybridSearchConfig = None) -> List[SearchResult]:
    """
    Convenience function for searching pandas documentation
    """
    searcher = HybridRAGSearcher(config)
    return await searcher.search(queries, strategy, SearchMethod.HYBRID)

def create_optimized_queries_for_pandas(data_analysis_task: str) -> List[str]:
    """
    Creates optimized queries for specific data analysis tasks
    """
    task_lower = data_analysis_task.lower()
    
    base_queries = []
    
    # Task to query mapping
    if "clean" in task_lower or "standardize" in task_lower:
        base_queries.extend([
            "pandas data cleaning best practices",
            "pandas handle missing values",
            "pandas standardize column names",
            "pandas convert data types"
        ])
    
    if "transform" in task_lower or "preprocess" in task_lower:
        base_queries.extend([
            "pandas data transformation",
            "pandas feature engineering",
            "pandas reshape data",
            "pandas apply functions"
        ])
    
    if "analysis" in task_lower or "reporting" in task_lower:
        base_queries.extend([
            "pandas data analysis",
            "pandas groupby aggregation",
            "pandas statistical operations",
            "pandas visualization"
        ])
    
    # If no specific patterns found, use generic queries
    if not base_queries:
        base_queries = [
            "pandas data processing",
            "pandas dataframe operations",
            "pandas best practices"
        ]
    
    return base_queries

# === INTEGRATION WITH CURRENT SYSTEM ===

def enhance_crag_with_hybrid_search(original_queries: List[str], 
                                   user_instructions: str,
                                   config: HybridSearchConfig = None) -> List[str]:
    """
    Improves current CRAG queries using the hybrid system
    """
    # Analyze user instructions to determine strategy
    instructions_lower = user_instructions.lower()
    
    if "documentation" in instructions_lower or "reference" in instructions_lower:
        strategy = RetrievalStrategy.DOCUMENTATION
    elif "example" in instructions_lower or "tutorial" in instructions_lower:
        strategy = RetrievalStrategy.CODE_EXAMPLES
    elif "best practice" in instructions_lower or "optimal" in instructions_lower:
        strategy = RetrievalStrategy.BEST_PRACTICES
    else:
        strategy = RetrievalStrategy.DOCUMENTATION
    
    # Create optimized queries
    optimized_queries = []
    for query in original_queries:
        expanded = expand_pandas_query(query, strategy)
        optimized_queries.extend(expanded)
    
    # Add queries based on user instructions
    task_queries = create_optimized_queries_for_pandas(user_instructions)
    optimized_queries.extend(task_queries)
    
    # Remove duplicates while maintaining order
    seen = set()
    final_queries = []
    for query in optimized_queries:
        if query not in seen:
            seen.add(query)
            final_queries.append(query)
    
    return final_queries[:15]  # Limit to avoid overloading