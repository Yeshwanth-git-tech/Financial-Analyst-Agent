import os
import json
import pandas as pd
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from pathlib import Path

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import faiss

from llama_index.core.indices.loading import load_index_from_storage

# Neo4j imports (optional)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  Neo4j not available, running in Vector-only mode")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available, using basic entity extraction")

class QuickHybridRAG:
    """Quick working version of Hybrid RAG system"""
    
    def __init__(self):
        load_dotenv()
        
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
        
        # Paths
        self.faiss_index_dir = "storage/faiss_index"
        
        # Initialize components
        self.vector_index = None
        self.query_engine = None
        self.stock_data = {}
        
        # Load systems
        self._load_vector_index()
        self._load_stock_data()
        self._setup_query_engine()
    
    def _load_vector_index(self):
        """Load FAISS vector index"""
        print("🧠 Loading FAISS vector index...")
        
        try:
            # Use the correct method to load from storage
            storage_context = StorageContext.from_defaults(persist_dir=self.faiss_index_dir)
            self.vector_index = load_index_from_storage(storage_context)
            print("✅ Vector index loaded successfully")
        except Exception as e:
            print(f"❌ Error loading vector index: {e}")
            print("Make sure you've run ingest_filing.py first!")
            raise
    
    def _load_stock_data(self):
        """Load stock data from JSON files"""
        print("📈 Loading stock data...")
        
        # Load from sec_filings directory (your JSON files)
        sec_dir = "data/sec_filings"
        if os.path.exists(sec_dir):
            for filename in os.listdir(sec_dir):
                if filename.endswith("_stock_price.json"):
                    filepath = os.path.join(sec_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Extract symbol from filename or data
                        if 'ticker' in data:
                            symbol = data['ticker']
                        else:
                            symbol = filename.split('_')[0]  # e.g., TSLA from TSLA_stock_price.json
                        
                        self.stock_data[symbol] = data
                        print(f"✅ Loaded stock data for {symbol}: ${data.get('price', 'N/A')}")
                        
                    except Exception as e:
                        print(f"⚠️  Error loading {filename}: {e}")
        
        # Also try to load from data/stock_data if it exists
        stock_dir = "data/stock_data"
        if os.path.exists(stock_dir):
            for filename in os.listdir(stock_dir):
                if filename.endswith('.csv'):
                    symbol = filename.replace('.csv', '').upper()
                    try:
                        df = pd.read_csv(os.path.join(stock_dir, filename))
                        if not df.empty:
                            # Get the latest data
                            latest = df.iloc[-1] if 'Date' not in df.columns else df.sort_values('Date').iloc[-1]
                            self.stock_data[symbol] = {
                                'ticker': symbol,
                                'price': latest.get('Close', latest.get('price', 'N/A')),
                                'volume': latest.get('Volume', 'N/A'),
                                'source': 'csv'
                            }
                            print(f"✅ Loaded CSV data for {symbol}")
                    except Exception as e:
                        print(f"⚠️  Error loading {filename}: {e}")
    
    def _setup_query_engine(self):
        """Setup query engine"""
        print("⚙️  Setting up query engine...")
        
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=5
        )
        
        self.query_engine = RetrieverQueryEngine(retriever=retriever)
        print("✅ Query engine ready")
    
    def _extract_companies_from_query(self, query: str) -> List[str]:
        """Extract company names from query"""
        company_mappings = {
            "tesla": "TSLA",
            "apple": "AAPL", 
            "microsoft": "MSFT",
            "amazon": "AMZN",
            "google": "GOOGL",
            "meta": "META",
            "nvidia": "NVDA"
        }
        
        companies = []
        query_lower = query.lower()
        
        for name, symbol in company_mappings.items():
            if name in query_lower or symbol.lower() in query_lower:
                companies.append(symbol)
        
        return list(set(companies))  # Remove duplicates
    
    def analyze_query(self, query: str) -> Dict:
        """Analyze query using vector search and stock data"""
        print(f"🔍 Analyzing query: {query}")
        
        # 1. Vector-based retrieval from SEC filings
        vector_response = self.query_engine.query(query)
        
        # 2. Extract companies from query
        companies = self._extract_companies_from_query(query)
        
        # 3. Get stock data for identified companies
        stock_insights = {}
        for symbol in companies:
            if symbol in self.stock_data:
                data = self.stock_data[symbol]
                stock_insights[symbol] = {
                    "current_price": data.get('price', 'N/A'),
                    "volume": data.get('volume', 'N/A'),
                    "source": data.get('source', 'json')
                }
        
        # 4. Generate comprehensive analysis
        analysis_context = {
            "query": query,
            "sec_analysis": str(vector_response),
            "stock_data": stock_insights,
            "companies_found": companies
        }
        
        comprehensive_analysis = self._generate_analysis(analysis_context)
        
        return {
            **analysis_context,
            "comprehensive_analysis": comprehensive_analysis
        }
    
    def _generate_analysis(self, context: Dict) -> str:
        """Generate comprehensive analysis"""
        
        stock_summary = ""
        if context["stock_data"]:
            stock_summary = "CURRENT STOCK DATA:\n"
            for symbol, data in context["stock_data"].items():
                price = data.get('current_price', 'N/A')
                stock_summary += f"• {symbol}: ${price}\n"
        
        prompt = f"""
As a senior financial analyst, provide a comprehensive analysis combining SEC filing data and market information:

QUERY: {context['query']}

SEC FILING ANALYSIS:
{context['sec_analysis']}

{stock_summary}

INSTRUCTIONS:
1. Synthesize insights from SEC filings and current stock data
2. Provide specific, actionable investment insights
3. Highlight key risks and opportunities
4. Use bullet points for key findings
5. Be specific and cite relevant information

COMPREHENSIVE ANALYSIS:
"""
        
        try:
            response = Settings.llm.complete(prompt)
            return str(response)
        except Exception as e:
            return f"Analysis generation error: {str(e)}\n\nSEC Filing Response: {context['sec_analysis']}"
    
    def query(self, question: str) -> str:
        """Main query interface"""
        try:
            analysis = self.analyze_query(question)
            return self._format_response(analysis)
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def _format_response(self, analysis: Dict) -> str:
        """Format the response"""
        output = f"""
🔍 FINANCIAL ANALYSIS REPORT
{'='*50}

QUERY: {analysis['query']}

📊 STOCK DATA:
{'-'*30}"""
        
        if analysis['stock_data']:
            for symbol, data in analysis['stock_data'].items():
                price = data.get('current_price', 'N/A')
                output += f"\n{symbol}: ${price}"
        else:
            output += "\nNo stock data available for queried companies."
        
        output += f"""

🧠 COMPREHENSIVE ANALYSIS:
{'-'*30}
{analysis['comprehensive_analysis']}

📄 SEC FILING INSIGHTS:
{'-'*30}
{analysis['sec_analysis']}
"""
        
        return output

def test_system():
    """Quick test of the system"""
    print("🧪 Testing the system...")
    
    try:
        rag = QuickHybridRAG()
        
        # Test queries
        test_queries = [
            "What are Tesla's main business risks?",
            "How is Tesla's financial performance?",
            "What does Tesla say about competition?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"TEST QUERY: {query}")
            print('='*60)
            
            response = rag.query(query)
            print(response)
            break  # Just test one query for now
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

# Main execution
if __name__ == "__main__":
    print("🚀 Quick Hybrid RAG System Starting...")
    
    # Check if we should run tests or interactive mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_system()
    else:
        # Initialize the system
        try:
            rag = QuickHybridRAG()
            
            print("\n🎯 Quick Hybrid RAG System Ready!")
            print("="*50)
            
            # Interactive mode
            while True:
                user_query = input("\n💬 Enter your financial query (or 'quit' to exit): ")
                
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_query.strip():
                    print("\n🔄 Processing query...")
                    response = rag.query(user_query)
                    print(response)
                else:
                    print("Please enter a valid query.")
                    
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            print("\n🔧 Quick fixes to try:")
            print("1. Make sure you've run: python ingest_filing.py")
            print("2. Check that storage/faiss_index exists")
            print("3. Verify your .env file has OPENAI_API_KEY")