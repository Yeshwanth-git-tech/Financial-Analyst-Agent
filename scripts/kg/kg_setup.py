# kg_setup.py - Knowledge Graph Setup and Installation

import os
import subprocess
import sys
from dotenv import load_dotenv

def install_requirements():
    """Install additional requirements for Knowledge Graph"""
    print("📦 Installing Knowledge Graph requirements...")

    requirements = [
        "neo4j==5.12.0",
        "spacy>=3.6.0",
        "pandas>=1.5.0",
        "python-dotenv>=1.0.0"
    ]

    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

    # Install spaCy model
    print("\n📚 Installing spaCy English model...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install spaCy model: {e}")

def setup_env_file():
    """Setup .env file with Neo4j credentials"""
    print("🔐 Setting up Neo4j credentials...")

    env_file = ".env"
    load_dotenv()

    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")

    if not neo4j_user:
        neo4j_user = input("Enter Neo4j username (default: neo4j): ").strip() or "neo4j"

    if not neo4j_pass:
        neo4j_pass = input("Enter Neo4j password: ").strip()
        if not neo4j_pass:
            print("⚠️  Using default password 'password'")
            neo4j_pass = "password"

    env_content = ""
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()

    if "NEO4J_USERNAME" not in env_content:
        env_content += f"\nNEO4J_USERNAME={neo4j_user}"

    if "NEO4J_PASSWORD" not in env_content:
        env_content += f"\nNEO4J_PASSWORD={neo4j_pass}"

    with open(env_file, 'w') as f:
        f.write(env_content.strip() + "\n")

    print("✅ Neo4j credentials saved to .env file")

def test_neo4j_connection():
    """Test connection to Neo4j"""
    print("🔍 Testing Neo4j connection...")

    try:
        from neo4j import GraphDatabase
        load_dotenv()

        uri = "bolt://localhost:7687"
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        driver.close()

        print("✅ Neo4j connection successful!")
        return True

    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return False

def run_sample_queries():
    """Run sample queries to demonstrate Knowledge Graph capabilities"""
    print("🔍 Running sample Knowledge Graph queries...")

    try:
        from neo4j import GraphDatabase
        load_dotenv()

        uri = "bolt://localhost:7687"
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")

        driver = GraphDatabase.driver(uri, auth=(username, password))

        sample_queries = [
            {
                "name": "Companies mentioned by Apple",
                "query": """
                MATCH (apple:Company {name: "Apple Inc."})-[:HAS_FILING]->(f:Filing)-[:MENTIONS]->(mentioned:Company)
                RETURN mentioned.name as company, mentioned.sector as sector
                """
            },
            {
                "name": "Common suppliers between companies",
                "query": """
                MATCH (c1:Company)-[:SUPPLIER_RELATIONSHIP]->(supplier:Company)<-[:SUPPLIER_RELATIONSHIP]-(c2:Company)
                WHERE c1.name <> c2.name
                RETURN c1.name as company1, c2.name as company2, supplier.name as common_supplier
                """
            },
            {
                "name": "CEO information",
                "query": """
                MATCH (company:Company)-[:HAS_CEO]->(ceo:Person)
                RETURN company.name as company, ceo.name as ceo, company.ticker as ticker
                ORDER BY company.name
                """
            },
            {
                "name": "Cross-mentions in filings",
                "query": """
                MATCH (c1:Company)-[:HAS_FILING]->(f:Filing)-[:MENTIONS]->(c2:Company)
                WHERE c1.name <> c2.name
                RETURN c1.name as filing_company, c2.name as mentioned_company, f.form_type as filing_type
                """
            }
        ]

        with driver.session() as session:
            for sample in sample_queries:
                print(f"\n📊 {sample['name']}:")
                print("-" * 40)

                result = session.run(sample['query'])
                records = [dict(record) for record in result]

                if records:
                    for record in records:
                        print(f"   • {' | '.join([f'{k}: {v}' for k, v in record.items()])}")
                else:
                    print("   No results found")

        driver.close()
        print("\n🎉 Sample queries completed!")

    except Exception as e:
        print(f"❌ Error running sample queries: {e}")

def check_docker_status():
    """Check if Docker containers are running"""
    print("🐳 Checking Docker container status...")

    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)

        if 'financial-neo4j' in result.stdout:
            print("✅ Neo4j container is running")
        else:
            print("❌ Neo4j container not found")
            print("Run: docker-compose up -d")

        if 'financial-redis' in result.stdout:
            print("✅ Redis container is running")
        else:
            print("⚠️  Redis container not found (optional)")

    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker first.")
    except Exception as e:
        print(f"❌ Error checking Docker: {e}")

def create_cypher_cheatsheet():
    """Create a Cypher query cheatsheet for financial analysis"""
    cheatsheet = """<your big string here>"""  # same markdown string as before
    with open("cypher_cheatsheet.md", "w") as f:
        f.write(cheatsheet)
    print("📚 Cypher cheatsheet created: cypher_cheatsheet.md")

def main():
    print("🚀 Setting up Advanced RAG + Knowledge Graph System")
    print("=" * 60)

    install_requirements()
    setup_env_file()

    print("\n3️⃣ Testing Neo4j connection...")
    if not test_neo4j_connection():
        print("❌ Aborting setup due to connection failure.")
        return

    print("\n4️⃣ Running Docker check...")
    check_docker_status()

    print("\n5️⃣ Running sample queries...")
    run_sample_queries()

    print("\n6️⃣ Creating cheatsheet...")
    create_cypher_cheatsheet()

    print("\n🎉 All setup steps completed successfully!")

if __name__ == "__main__":
    main()