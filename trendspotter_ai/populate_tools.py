import json
import feedparser
import hashlib
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

class ToolPopulator:
    def __init__(self):
        self.client = OpenAI()
        
    def fix_existing_integrity_tool(self):
        """Fix the Integrity tool that's missing data"""
        print("🔧 Fixing Integrity tool...")
        
        with open('data/tools.json', 'r') as f:
            tools = json.load(f)
        
        # Find Integrity
        for tool_id, tool in tools.items():
            if tool.get('name') == 'Integrity':
                print(f"  Found Integrity, fixing...")
                
                # Properly analyze it
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{
                        "role": "user",
                        "content": """
                        Integrity is a unified project brain where notes, canvases, and AI chats become connected layers of thought.
                        
                        Return JSON:
                        {
                            "why_important": "why product teams should care",
                            "immediate_action": "specific first step",
                            "implementation_effort": "hours",
                            "keywords": ["project", "management", "ai", "collaboration", "notes"],
                            "tool_category": "platform",
                            "maturity_level": "beta",
                            "requirements": {
                                "requires_api_key": false,
                                "requires_payment": true
                            }
                        }
                        """
                    }],
                    temperature=0.3,
                    max_tokens=300
                )
                
                analysis = json.loads(response.choices[0].message.content)
                
                # Update tool
                tool.update(analysis)
                tool['description'] = "Unified project brain where notes, canvases, and AI chats become connected layers"
                tool['signal_strength'] = 'high'
                
                print(f"  ✅ Fixed Integrity tool")
        
        # Save
        with open('data/tools.json', 'w') as f:
            json.dump(tools, f, indent=2)
    
    def get_real_ai_tools(self):
        """Get actual AI tools from various sources"""
        tools_to_add = [
            {
                "name": "Claude 3.5 Sonnet",
                "description": "Anthropic's latest AI model with improved coding and analysis",
                "url": "https://www.anthropic.com/claude",
                "category": "model"
            },
            {
                "name": "GPT-4 Turbo", 
                "description": "OpenAI's latest model with 128k context and lower prices",
                "url": "https://openai.com/gpt-4",
                "category": "model"
            },
            {
                "name": "Cursor",
                "description": "AI-powered code editor built on VSCode",
                "url": "https://cursor.sh",
                "category": "tool"
            },
            {
                "name": "v0.dev",
                "description": "Vercel's AI tool for generating React components",
                "url": "https://v0.dev",
                "category": "tool"
            },
            {
                "name": "Perplexity API",
                "description": "Search-enhanced AI API for real-time information",
                "url": "https://perplexity.ai",
                "category": "api"
            },
            {
                "name": "LangChain",
                "description": "Framework for developing LLM applications",
                "url": "https://langchain.com",
                "category": "library"
            },
            {
                "name": "Ollama",
                "description": "Run LLMs locally on your machine",
                "url": "https://ollama.ai",
                "category": "platform"
            },
            {
                "name": "Hugging Face Inference API",
                "description": "Access thousands of models via API",
                "url": "https://huggingface.co",
                "category": "api"
            }
        ]
        
        analyzed_tools = []
        
        for tool_info in tools_to_add:
            print(f"📍 Analyzing {tool_info['name']}...")
            
            prompt = f"""
            Analyze this AI tool for enterprise use:
            Name: {tool_info['name']}
            Description: {tool_info['description']}
            Category: {tool_info['category']}
            
            Return detailed JSON:
            {{
                "name": "{tool_info['name']}",
                "description": "{tool_info['description']}",
                "tool_category": "{tool_info['category']}",
                "keywords": ["list", "relevant", "keywords"],
                "why_important": "specific reason this matters for product teams",
                "immediate_action": "specific first step to evaluate this",
                "implementation_effort": "hours|days|weeks",
                "implementation_keywords": ["api", "python", "javascript"],
                "time_saved_per_month": 10-100,
                "maturity_level": "production|beta|experimental",
                "signal_strength": "high|moderate",
                "requirements": {{
                    "requires_api_key": true/false,
                    "requires_payment": true/false,
                    "programming_language": "any|python|javascript"
                }},
                "alternatives": ["list", "alternatives"],
                "cost_benefit": {{
                    "setup_cost": "low|medium|high",
                    "monthly_cost": "$0-100|$100-1000|$1000+"
                }}
            }}
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an AI tools expert. Provide practical analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                tool_data = json.loads(response.choices[0].message.content)
                tool_data['id'] = hashlib.md5(tool_info['url'].encode()).hexdigest()
                tool_data['source'] = 'Manual Addition'
                tool_data['source_url'] = tool_info['url']
                tool_data['content_type'] = 'TOOL'
                tool_data['ingested_at'] = datetime.now().isoformat()
                
                analyzed_tools.append(tool_data)
                print(f"  ✅ Added {tool_info['name']}")
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return analyzed_tools
    
    def add_tools_to_database(self, new_tools):
        """Add tools to database"""
        # Load existing
        try:
            with open('data/tools.json', 'r') as f:
                tools = json.load(f)
        except:
            tools = {}
        
        # Add new tools
        for tool in new_tools:
            tools[tool['id']] = tool
        
        # Save
        with open('data/tools.json', 'w') as f:
            json.dump(tools, f, indent=2)
        
        print(f"\n✅ Total tools in database: {len(tools)}")
    
    def check_github_trending(self):
        """Get trending AI repositories"""
        print("\n🔍 Checking GitHub trending...")
        
        feed_url = 'https://rsshub.app/github/trending/daily/python'
        feed = feedparser.parse(feed_url)
        
        tools = []
        for entry in feed.entries[:5]:
            if any(term in entry.title.lower() for term in ['ai', 'llm', 'gpt', 'ml', 'model']):
                tool = {
                    'id': hashlib.md5(entry.link.encode()).hexdigest(),
                    'name': entry.title.split('/')[-1] if '/' in entry.title else entry.title,
                    'description': entry.get('summary', '')[:200],
                    'source': 'GitHub Trending',
                    'source_url': entry.link,
                    'content_type': 'TOOL',
                    'tool_category': 'library',
                    'keywords': ['github', 'opensource', 'python'],
                    'why_important': 'Trending on GitHub indicates growing adoption',
                    'immediate_action': f'Clone and test: git clone {entry.link}',
                    'implementation_effort': 'days',
                    'signal_strength': 'moderate',
                    'maturity_level': 'beta',
                    'ingested_at': datetime.now().isoformat()
                }
                tools.append(tool)
                print(f"  ✓ Found: {tool['name']}")
        
        return tools

def main():
    print("🚀 Populating Tools Database")
    print("="*60)
    
    populator = ToolPopulator()
    
    # Fix existing tool
    populator.fix_existing_integrity_tool()
    
    # Add known good AI tools
    print("\n📦 Adding essential AI tools...")
    essential_tools = populator.get_real_ai_tools()
    
    # Check GitHub
    github_tools = populator.check_github_trending()
    
    # Combine and save
    all_tools = essential_tools + github_tools
    populator.add_tools_to_database(all_tools)
    
    print("\n🎉 Done! Your database now has real AI tools.")

if __name__ == "__main__":
    main()