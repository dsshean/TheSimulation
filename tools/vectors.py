from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    Document
)
from llama_index.vector_stores import QdrantVectorStore
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from qdrant_client import QdrantClient, models
import logging
import sys
from pathlib import Path
from typing import List, Dict
import json

class DocumentProcessor:
    def __init__(
        self,
        openai_api_key: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents",
        persist_dir: str = "./storage",
        model_name: str = "gpt-3.5-turbo",
    ):
        """
        Initialize the document processor with LlamaIndex and Qdrant.
        
        Args:
            openai_api_key: OpenAI API key
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            persist_dir: Directory to persist graph store
            model_name: OpenAI model to use
        """
        # Set up logging
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize paths
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up OpenAI
        self.llm = OpenAI(temperature=0, model=model_name, api_key=openai_api_key)
        self.embed_model = OpenAIEmbedding(api_key=openai_api_key)
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Create Qdrant collection if it doesn't exist
        self._init_qdrant_collection()
        
        # Initialize vector store with Qdrant
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name
        )
        
        # Initialize graph store
        self.graph_store = SimpleGraphStore()
        
        # Set up storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            graph_store=self.graph_store
        )
        
        # Set up service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
        )
        
        # Initialize indexes
        self.vector_index = None

    def _init_qdrant_collection(self):
        """Initialize Qdrant collection with proper configuration."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            
            if not exists:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                
                # Create payload index for metadata
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="metadata",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                
        except Exception as e:
            self.logger.error(f"Error initializing Qdrant collection: {e}")
            raise

    def load_documents(self, directory: str) -> List[Document]:
        """
        Load documents from a directory.
        """
        self.logger.info(f"Loading documents from {directory}")
        documents = SimpleDirectoryReader(directory).load_data()
        return documents

    def process_documents(self, documents: List[Document]):
        """
        Process documents and create both vector and knowledge graph indexes.
        """
        self.logger.info("Creating vector index...")
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.service_context
        )
        
        # Extract and store triplets in knowledge graph
        self.logger.info("Creating knowledge graph...")
        
        # Custom prompt for triple extraction
        triple_extract_prompt = """
        Extract knowledge triplets from the text below. Format should be (subject, predicate, object).
        Extract only the most important relationships. Focus on factual information.
        
        Text: {text}
        
        Triplets:
        """
        
        knowledge_graph = []
        
        for doc in documents:
            # Extract triplets using LLM
            response = self.llm.complete(
                triple_extract_prompt.format(text=doc.text)
            )
            
            # Process extracted triplets and add to graph store
            triplets = self._parse_triplets(response.text)
            for subj, pred, obj in triplets:
                self.graph_store.add_triplet(subj, pred, obj)
                knowledge_graph.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj
                })
        
        # Save knowledge graph to file
        with open(self.persist_dir / "knowledge_graph.json", "w") as f:
            json.dump(knowledge_graph, f, indent=2)

    def query_vector_store(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Query the vector store for similar documents.
        """
        if not self.vector_index:
            raise ValueError("Vector index not initialized. Process documents first.")
        
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=top_k,
        )
        
        response = query_engine.query(query)
        
        # Format results
        results = []
        for node in response.source_nodes:
            results.append({
                "text": node.node.get_text(),
                "score": node.score,
                "metadata": node.node.metadata
            })
        
        return results

    def query_knowledge_graph(self, query: str) -> List[Dict]:
        """
        Query the knowledge graph using natural language.
        """
        # Convert natural language to graph query
        graph_query_prompt = """
        Convert the following question into a graph query pattern.
        Return subject, predicate, and/or object to search for.
        Use * for unknown values.
        
        Question: {query}
        
        Format:
        subject: <value or *>
        predicate: <value or *>
        object: <value or *>
        """
        
        response = self.llm.complete(
            graph_query_prompt.format(query=query)
        )
        
        # Parse query parameters
        query_params = self._parse_graph_query(response.text)
        
        # Query graph store
        results = self.graph_store.query(
            subject=query_params.get("subject", "*"),
            predicate=query_params.get("predicate", "*"),
            object=query_params.get("object", "*")
        )
        
        return results

    def _parse_triplets(self, text: str) -> List[tuple]:
        """
        Parse extracted triplets from LLM response.
        """
        triplets = []
        for line in text.split('\n'):
            if '(' in line and ')' in line:
                content = line[line.find("(")+1:line.find(")")]
                parts = [p.strip() for p in content.split(',')]
                if len(parts) == 3:
                    triplets.append(tuple(parts))
        return triplets

    def _parse_graph_query(self, text: str) -> Dict:
        """
        Parse graph query parameters from LLM response.
        """
        params = {}
        for line in text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                value = value.strip()
                if value != '*':
                    params[key.strip()] = value
        return params

    def search_by_metadata(self, metadata_filter: Dict) -> List[Dict]:
        """
        Search documents by metadata using Qdrant's filtering capabilities.
        """
        # Convert metadata filter to Qdrant filter
        qdrant_filter = {
            "must": [
                {
                    "key": f"metadata.{key}",
                    "match": {"value": value}
                }
                for key, value in metadata_filter.items()
            ]
        }
        
        # Search in Qdrant
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_filter=models.Filter(**qdrant_filter),
            limit=10
        )
        
        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]

# # Example usage
# def main():
#     # Initialize processor
#     processor = DocumentProcessor(
#         openai_api_key="your_openai_api_key",
#         qdrant_host="localhost",
#         qdrant_port=6333
#     )
    
#     # Load and process documents
#     documents = processor.load_documents("./documents")
#     processor.process_documents(documents)
    
#     # Example vector query
#     vector_results = processor.query_vector_store(
#         "What are the main features discussed?",
#         top_k=3
#     )
#     print("\nVector Search Results:")
#     for result in vector_results:
#         print(f"Score: {result['score']:.4f}")
#         print(f"Text: {result['text'][:200]}...")
#         print(f"Metadata: {result['metadata']}\n")
    
#     # Example knowledge graph query
#     kg_results = processor.query_knowledge_graph(
#         "What are the relationships between documents and features?"
#     )
#     print("\nKnowledge Graph Results:")
#     for result in kg_results:
#         print(f"Subject: {result['subject']}")
#         print(f"Predicate: {result['predicate']}")
#         print(f"Object: {result['object']}\n")
    
#     # Example metadata search
#     metadata_results = processor.search_by_metadata({
#         "category": "technical",
#         "status": "active"
#     })
#     print("\nMetadata Search Results:")
#     for result in metadata_results:
#         print(f"Score: {result['score']:.4f}")
#         print(f"Text: {result['text'][:200]}...")
#         print(f"Metadata: {result['metadata']}\n")

# if __name__ == "__main__":
#     main()