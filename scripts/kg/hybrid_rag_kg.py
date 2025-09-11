import os
import json
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from scripts.market_data import get_stock_data

from pathlib import Path

# # LlamaIndex imports
# from llama_index.core import VectorStoreIndex, StorageContext, Settings
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# import faiss

# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core.storage.storage_context import StorageContext
# from llama_index.core import load_index_from_storage

# from llama_index.core.indices.loading import load_index_from_storage

# Core LlamaIndex imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext

# FAISS-specific imports
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss  # Needed for some low-level FAISS operations

# OpenAI embedding and LLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.core.indices.loading import load_index_from_storage
# from llama_index import load_index_from_storage, StorageContext


# Neo4j imports
from neo4j import GraphDatabase
import spacy

class FinancialKnowledgeGraph:
    """Knowledge Graph handler for financial entities and relationships"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", username: str = None, password: str = None):
        load_dotenv()
        
        self.uri = uri
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j Knowledge Graph")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            raise
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self._create_indexes()
    
    def _create_indexes(self):
        """Create Neo4j indexes for better performance"""
        with self.driver.session() as session:
            # Create constraints and indexes
            constraints = [
                "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
                "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
                "CREATE INDEX company_sector IF NOT EXISTS FOR (c:Company) ON (c.sector)",
                "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON (f.date)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    pass
        
        print("✅ Neo4j indexes created/verified")
    
    def extract_entities_from_text(self, text: str, filing_metadata: Dict = None) -> Dict[str, List[str]]:
        """Extract financial entities from text using NER"""
        entities = {
            "companies": set(),
            "people": set(),
            "locations": set(),
            "financial_terms": set(),
            "dates": set()
        }
        
        if not self.nlp:
            return {k: list(v) for k, v in entities.items()}
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ["ORG"]:
                entities["companies"].add(ent.text.strip())
            elif ent.label_ in ["PERSON"]:
                entities["people"].add(ent.text.strip())
            elif ent.label_ in ["GPE", "LOC"]:
                entities["locations"].add(ent.text.strip())
            elif ent.label_ in ["DATE"]:
                entities["dates"].add(ent.text.strip())
        
        # Extract financial terms using regex
        financial_patterns = [
            r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?',
            r'\d+(?:\.\d+)?%',
            r'(?:revenue|profit|loss|earnings|EBITDA|cash flow|debt|equity)',
        ]
        
        for pattern in financial_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["financial_terms"].update(matches)
        
        # Convert sets to lists
        return {k: list(v) for k, v in entities.items()}
    
    def create_filing_graph(self, filing_text: str, filing_metadata: Dict):
        """Create knowledge graph from SEC filing"""
        print(f"📊 Creating knowledge graph for filing: {filing_metadata.get('filing_id', 'Unknown')}")
        
        # Extract entities
        entities = self.extract_entities_from_text(filing_text, filing_metadata)
        
        with self.driver.session() as session:
            # Create filing node
            filing_query = """
            MERGE (f:Filing {filing_id: $filing_id})
            SET f.company = $company,
                f.form_type = $form_type,
                f.date = $date,
                f.text_snippet = substring($text, 0, 1000)
            RETURN f
            """
            
            session.run(filing_query, {
                "filing_id": filing_metadata.get("filing_id", "unknown"),
                "company": filing_metadata.get("company", "unknown"),
                "form_type": filing_metadata.get("form_type", "unknown"),
                "date": filing_metadata.get("date", datetime.now().isoformat()),
                "text": filing_text[:2000]  # Store snippet
            })
            
            # Create company nodes and relationships
            for company in entities["companies"][:10]:  # Limit to top 10
                company_query = """
                MERGE (c:Company {name: $name})
                WITH c
                MATCH (f:Filing {filing_id: $filing_id})
                MERGE (f)-[:MENTIONS]->(c)
                """
                session.run(company_query, {"name": company, "filing_id": filing_metadata.get("filing_id")})
            
            # Create person nodes and relationships
            for person in entities["people"][:10]:
                person_query = """
                MERGE (p:Person {name: $name})
                WITH p
                MATCH (f:Filing {filing_id: $filing_id})
                MERGE (f)-[:MENTIONS_PERSON]->(p)
                """
                session.run(person_query, {"name": person, "filing_id": filing_metadata.get("filing_id")})
        
        print(f"✅ Knowledge graph updated with {len(entities['companies'])} companies, {len(entities['people'])} people")
    
    def query_graph(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """Execute Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [dict(record) for record in result]
    
    def find_company_relationships(self, company_name: str) -> List[Dict]:
        """Find relationships for a specific company"""
        query = """
        MATCH (c:Company {name: $company})
        OPTIONAL MATCH (c)<-[:MENTIONS]-(f:Filing)-[:MENTIONS]->(other:Company)
        WHERE other.name <> c.name
        RETURN c.name as company, 
               collect(DISTINCT other.name)[0..10] as related_companies,
               collect(DISTINCT f.form_type)[0..5] as filing_types
        """
        return self.query_graph(query, {"company": company_name})
    
    def find_common_entities(self, company1: str, company2: str) -> List[Dict]:
        """Find common entities mentioned by two companies"""
        query = """
        MATCH (c1:Company {name: $company1})<-[:MENTIONS]-(f1:Filing)-[:MENTIONS]->(common:Company)
        MATCH (c2:Company {name: $company2})<-[:MENTIONS]-(f2:Filing)-[:MENTIONS]->(common)
        WHERE c1 <> c2 AND common <> c1 AND common <> c2
        RETURN common.name as common_entity, 
               count(DISTINCT f1) as mentions_by_company1,
               count(DISTINCT f2) as mentions_by_company2
        ORDER BY mentions_by_company1 + mentions_by_company2 DESC
        LIMIT 10
        """
        return self.query_graph(query, {"company1": company1, "company2": company2})
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()

class HybridRAGWithKnowledgeGraph:
    """Hybrid system combining Vector RAG with Knowledge Graph"""
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687"):
        load_dotenv()

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        self.faiss_index_dir = str(PROJECT_ROOT / "storage" / "faiss_index")
        self.stock_data_dir = str(PROJECT_ROOT / "data" / "stock_data")

        # Configure LlamaIndex
        Settings.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            temperature=0.1
        )
        Settings.embed_model = OpenAIEmbedding(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )

        self.vector_index = None
        self.query_engine = None
        # self.stock_data = {}
        self.kg = FinancialKnowledgeGraph(uri=neo4j_uri)

        self._load_vector_index()
        # self._load_stock_data()
        self._setup_query_engine()

    def _load_vector_index(self):
        print("🧠 Loading FAISS vector index...")
        try:
            faiss_store = FaissVectorStore.from_persist_dir(self.faiss_index_dir)
            storage_context = StorageContext.from_defaults(
                persist_dir=self.faiss_index_dir,
                vector_store=faiss_store
            )
            self.vector_index = load_index_from_storage(storage_context)
            print("✅ FAISS index loaded successfully.\n")
        except Exception as e:
            print(f"❌ Error loading vector index: {e}")
            raise e
        
    # def _load_vector_index(self):
    #     """Load FAISS vector index"""
    #     print("🧠 Loading FAISS vector index...")
        
    #     try:
    #         faiss_index = faiss.read_index(os.path.join(self.faiss_index_dir, "vector_store.faiss"))
    #         vector_store = FaissVectorStore(faiss_index=faiss_index)
            
    #         storage_context = StorageContext.from_defaults(
    #             vector_store=vector_store,
    #             persist_dir=self.faiss_index_dir
    #         )
            
    #         self.vector_index = VectorStoreIndex.from_vector_store(
    #             vector_store=vector_store,
    #             storage_context=storage_context
    #         )
            
    #         print("✅ Vector index loaded")
            
    #     except Exception as e:
    #         print(f"❌ Error loading vector index: {e}")
    #         raise

    # def _load_stock_data(self):
    #     data = {}

    #     # Load CSV files from data/stock_prices/
    #     if os.path.exists("data/stock_prices"):
    #         for fname in os.listdir("data/stock_prices"):
    #             if fname.endswith(".csv"):
    #                 symbol = fname.split("_")[0]
    #                 try:
    #                     df = pd.read_csv(os.path.join("data/stock_prices", fname))
    #                     data[symbol] = df
    #                 except Exception as e:
    #                     print(f"Error loading {fname}: {e}")

    #     # Load JSON stock prices from data/sec_filings/ like TSLA_stock_price.json
    #     if os.path.exists("data/sec_filings"):
    #         for fname in os.listdir("data/sec_filings"):
    #             if fname.endswith("_stock_price.json"):
    #                 try:
    #                     with open(os.path.join("data/sec_filings", fname)) as f:
    #                         json_data = json.load(f)
    #                         symbol = json_data.get("ticker")
    #                         if symbol:
    #                             data[symbol] = {
    #                                 "current_price": json_data.get("price"),
    #                                 "source": "json"
    #                             }
    #                 except Exception as e:
    #                     print(f"Error loading {fname}: {e}")

    #     return data
    
    
    def _setup_query_engine(self):
        """Setup query engine"""
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=8
        )
        
        self.query_engine = RetrieverQueryEngine(retriever=retriever)
        print("✅ Query engine ready")
    
    def analyze_hybrid_query(self, query: str) -> Dict:
        """Hybrid analysis using both vector search and knowledge graph"""
        print(f"🔍 Analyzing hybrid query: {query}")
        
        # 1. Vector-based retrieval
        vector_response = self.query_engine.query(query)
        
        # 2. Extract companies from query
        companies = self._extract_companies_from_query(query)
        
        # 3. Knowledge graph analysis
        kg_insights = {}
        if companies:
            for company in companies[:3]:  # Limit to 3 companies
                kg_insights[company] = {
                    "relationships": self.kg.find_company_relationships(company),
                    "common_entities": []
                }
                
                # Find relationships with other companies in query
                for other_company in companies:
                    if other_company != company:
                        common = self.kg.find_common_entities(company, other_company)
                        if common:
                            kg_insights[company]["common_entities"].extend(common)
        
        # 4. Stock data (if available)
        # 4. Stock data (live via Alpha Vantage)
        stock_insights = {}
        for company in companies:
            symbol = self._company_to_symbol(company)
            if symbol:
                stock_json = get_stock_data(symbol)
                if stock_json and "Time Series (Daily)" in stock_json:
                    try:
                        latest_date = max(stock_json["Time Series (Daily)"])
                        daily = stock_json["Time Series (Daily)"][latest_date]
                        stock_insights[company] = {
                            "symbol": symbol,
                            "date": latest_date,
                            "close": float(daily["4. close"]),
                            "high": float(daily["2. high"]),
                            "low": float(daily["3. low"]),
                            "volume": int(daily["5. volume"])
                        }
                    except Exception as e:
                        print(f"⚠️ Error parsing stock data for {symbol}: {e}")
        
        # 5. Generate comprehensive analysis
        analysis_context = {
            "query": query,
            "vector_analysis": str(vector_response),
            "knowledge_graph": kg_insights,
            "stock_data": stock_insights,
            "companies_analyzed": companies
        }
        
        comprehensive_analysis = self._generate_hybrid_analysis(analysis_context)
        
        return {
            **analysis_context,
            "comprehensive_analysis": comprehensive_analysis
        }
    
    def _extract_companies_from_query(self, query: str) -> List[str]:
        """Extract company names from query"""
        # Common company names mapping
        company_mappings = {
            "apple": "Apple Inc.",
            "microsoft": "Microsoft Corporation", 
            "tesla": "Tesla, Inc.",
            "amazon": "Amazon.com, Inc.",
            "google": "Alphabet Inc.",
            "meta": "Meta Platforms, Inc.",
            "facebook": "Meta Platforms, Inc.",
            "nvidia": "NVIDIA Corporation"
        }
        
        companies = []
        query_lower = query.lower()
        
        for key, full_name in company_mappings.items():
            if key in query_lower:
                companies.append(full_name)
        
        return companies
    
    def _company_to_symbol(self, company_name: str) -> Optional[str]:
        """Convert company name to stock symbol"""
        symbol_mappings = {
            "Apple Inc.": "AAPL",
            "Microsoft Corporation": "MSFT",
            "Tesla, Inc.": "TSLA",
            "Amazon.com, Inc.": "AMZN",
            "Alphabet Inc.": "GOOGL",
            "Meta Platforms, Inc.": "META",
            "NVIDIA Corporation": "NVDA"
        }
        
        return symbol_mappings.get(company_name)
    
    def _generate_hybrid_analysis(self, context: Dict) -> str:
        """Generate comprehensive analysis using all data sources"""
        
        kg_summary = ""
        if context["knowledge_graph"]:
            kg_summary = "KNOWLEDGE GRAPH INSIGHTS:\n"
            for company, data in context["knowledge_graph"].items():
                relationships = data.get("relationships", [])
                if relationships:
                    related_companies = relationships[0].get("related_companies", [])
                    kg_summary += f"• {company} is connected to: {', '.join(related_companies[:5])}\n"
        
        stock_summary = ""
        if context["stock_data"]:
            stock_summary = "CURRENT MARKET DATA:\n"
            for company, data in context["stock_data"].items():
                stock_summary += f"• {company}: ${data['current_price']:.2f} ({data['30d_change_pct']:+.1f}% 30d)\n"
        
        prompt = f"""
As a senior financial analyst, provide a comprehensive analysis combining multiple data sources:

QUERY: {context['query']}

DOCUMENT ANALYSIS (Vector Search):
{context['vector_analysis']}

{kg_summary}

{stock_summary}

INSTRUCTIONS:
1. Synthesize insights from SEC filings, relationship data, and market performance
2. Identify unique insights that emerge from combining these data sources
3. Highlight any contradictions or confirmations between sources
4. Provide actionable investment insights
5. Use specific examples from each data source
6. Structure with clear sections and bullet points

HYBRID ANALYSIS:
"""
        
        response = Settings.llm.complete(prompt)
        return str(response)
    
    def query(self, question: str) -> str:
        """Main query interface"""
        try:
            analysis = self.analyze_hybrid_query(question)
            return self._format_hybrid_response(analysis)
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _format_hybrid_response(self, analysis: Dict) -> str:
        """Format the hybrid response"""
        output = f"""
🔍 HYBRID RAG + KNOWLEDGE GRAPH ANALYSIS
{'='*60}

QUERY: {analysis['query']}

🧠 COMPREHENSIVE ANALYSIS:
{'-'*40}
{analysis['comprehensive_analysis']}

📊 KNOWLEDGE GRAPH INSIGHTS:
{'-'*40}"""
        
        if analysis['knowledge_graph']:
            for company, data in analysis['knowledge_graph'].items():
                relationships = data.get('relationships', [])
                if relationships:
                    related = relationships[0].get('related_companies', [])
                    output += f"\n{company}: Connected to {', '.join(related[:3])}"
        else:
            output += "\nNo relationship data found for queried entities."
        
        output += f"""

📈 MARKET DATA:
{'-'*40}"""
        
        if analysis['stock_data']:
            for company, data in analysis['stock_data'].items():
                output += f"""
        {company} ({data['symbol']}):
        • Date: {data['date']}
        • Close: ${data['close']:.2f}
        • High: ${data['high']:.2f}
        • Low: ${data['low']:.2f}
        • Volume: {data['volume']} shares
        """

        else:
            output += "\nNo current market data available."
        
        return output
    
    def populate_knowledge_graph_from_filings(self, sec_filings_dir: str = "data/sec_filings"):
        """Populate knowledge graph from existing SEC filings"""
        print("🔄 Populating knowledge graph from SEC filings...")
        
        if not os.path.exists(sec_filings_dir):
            print(f"❌ SEC filings directory not found: {sec_filings_dir}")
            return
        
        file_count = 0
        for root, dirs, files in os.walk(sec_filings_dir):
            for file in files:
                if file.endswith(('.txt', '.html', '.md')):
                    filepath = os.path.join(root, file)
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        # Extract metadata from filename/path
                        filing_metadata = {
                            "filing_id": file,
                            "company": self._extract_company_from_filename(file),
                            "form_type": self._extract_form_type_from_filename(file),
                            "date": datetime.now().isoformat()
                        }
                        
                        # Create knowledge graph
                        self.kg.create_filing_graph(content, filing_metadata)
                        file_count += 1
                        
                        if file_count % 5 == 0:
                            print(f"✅ Processed {file_count} files...")
                            
                    except Exception as e:
                        print(f"⚠️  Error processing {file}: {e}")
        
        print(f"🎉 Knowledge graph populated with {file_count} SEC filings!")
    
    def _extract_company_from_filename(self, filename: str) -> str:
        """Extract company name from filename"""
        # Simple extraction - you may want to improve this
        base_name = filename.split('.')[0]
        parts = base_name.split('_')
        return parts[0] if parts else "Unknown"
    
    def _extract_form_type_from_filename(self, filename: str) -> str:
        """Extract form type from filename"""
        if '10-K' in filename or '10K' in filename:
            return "10-K"
        elif '10-Q' in filename or '10Q' in filename:
            return "10-Q"
        elif '8-K' in filename or '8K' in filename:
            return "8-K"
        else:
            return "Unknown"

# Usage example
if __name__ == "__main__":
    print("🚀 Initializing Hybrid RAG + Knowledge Graph System...")
    
    # Initialize the hybrid system
    hybrid_rag = HybridRAGWithKnowledgeGraph()
    
    # Populate knowledge graph (run this once)
    print("\n📊 Would you like to populate the knowledge graph from existing filings? (y/n)")
    populate = input().lower().strip()
    
    if populate == 'y':
        hybrid_rag.populate_knowledge_graph_from_filings()
    
    print("\n🎯 Hybrid RAG + Knowledge Graph System Ready!")
    print("="*60)
    
    # Interactive mode
    while True:
        user_query = input("\n💬 Enter your financial research query (or 'quit' to exit): ")
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            hybrid_rag.kg.close()
            print("👋 Goodbye!")
            break
        
        if user_query.strip():
            print("\n🔄 Processing hybrid query...")
            response = hybrid_rag.query(user_query)
            print(response)
        else:
            print("Please enter a valid query.")



