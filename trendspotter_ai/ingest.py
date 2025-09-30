import os
import json
import feedparser
import hashlib
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
import time
import re

load_dotenv()

class EnhancedIngester:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        if not self.client:
            print("⚠️  No OpenAI API key - descriptions will be basic")
        
        self.tools_db_path = 'data/tools.json'
        self.news_db_path = 'data/news.json'
        self.load_existing_data()
        
    def load_existing_data(self):
        """Load existing databases"""
        if os.path.exists(self.tools_db_path):
            with open(self.tools_db_path, 'r') as f:
                self.tools = json.load(f)
        else:
            self.tools = {}
        
        if os.path.exists(self.news_db_path):
            with open(self.news_db_path, 'r') as f:
                self.news = json.load(f)
        else:
            self.news = {}
    
    def extract_tool_description(self, entry, source):
        """Extract a proper description of what the tool actually does"""
        
        # Gather all available text
        title = entry.get('title', '')
        summary = entry.get('summary', '')
        content = entry.get('content', [{}])[0].get('value', '') if entry.get('content') else ''
        
        # Clean HTML from summary
        clean_summary = re.sub('<[^<]+?>', '', summary) if summary else ''
        
        # Combine available text
        full_text = f"{title}\n{clean_summary}\n{content}"[:1500]
        
        # Try AI extraction if available
        if self.client and len(full_text) > 50:
            try:
                prompt = f"""
                Based on this RSS feed entry, write a clear 1-2 sentence description of what this tool/product actually does.
                Focus on its core functionality and purpose. Be specific and practical.
                
                Text: {full_text[:800]}
                
                Return only the description, no other text.
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Extract clear, practical descriptions of tools. Be concise and specific."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=100
                )
                
                description = response.choices[0].message.content.strip()
                if len(description) > 20:  # Sanity check
                    return description
                    
            except Exception as e:
                print(f"      ⚠ AI description extraction failed: {str(e)[:50]}")
        
        # Fallback: Smart extraction from summary
        if clean_summary:
            # Look for common patterns that describe what it does
            patterns = [
                r'is a (.*?) that',
                r'is an? (.*?) for',
                r'enables (.*?)\.',
                r'allows (.*?)\.',
                r'helps (.*?)\.',
                r'provides (.*?)\.',
                r'tool for (.*?)\.',
                r'platform for (.*?)\.',
                r'service that (.*?)\.',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, clean_summary, re.IGNORECASE)
                if match:
                    return f"Tool that {match.group(1)}"
            
            # Take first 200 chars or up to first period
            if '.' in clean_summary[:200]:
                return clean_summary[:clean_summary.index('.')+1]
            return clean_summary[:200].strip() + "..."
        
        # Last resort: use title intelligently
        if ' - ' in title:
            return title.split(' - ')[1].strip()
        return title
    
    def get_all_feeds(self):
        """YOUR ACTUAL RSS FEEDS WITH CORRECT URLS"""
        return [
            # YOUR CUSTOM FEEDS - FIXED URLs
            ('Reddit ChatGPT', 'https://rss.app/feeds/J5zfMj59iCwbgOJ3.xml'),
            ('Reddit Anthropic', 'https://rss.app/feeds/2EZXNje8iYlQNPTD.xml'),
            ('Google Tech News', 'https://rss.app/feeds/DXde8vatMEnzqB71.xml'),
            ('Microsoft LinkedIn', 'https://rss.app/feeds/m3EX09lP5Pj3BN1D.xml'),
            ('Y Combinator', 'https://rss.app/feeds/Z9ZGkpLYFI7Fhm2n.xml'),
            
            # Tool-focused sources
            ('Product Hunt Today', 'https://www.producthunt.com/feed?category=artificial-intelligence'),
            ('HackerNews Show', 'https://hnrss.org/show'),  # Show HN = tools
            ('GitHub Trending Python', 'https://rsshub.app/github/trending/daily/python'),
            ('GitHub Trending JavaScript', 'https://rsshub.app/github/trending/daily/javascript'),
            ('GitHub Trending AI', 'https://rsshub.app/github/trending/daily/jupyter-notebook'),
            
            # Tech news
            ('TechCrunch AI', 'https://techcrunch.com/category/artificial-intelligence/feed/'),
            ('VentureBeat AI', 'https://feeds.feedburner.com/venturebeat/SZYF'),
            
            # Reddit communities  
            ('Reddit LocalLLaMA', 'https://www.reddit.com/r/LocalLLaMA/hot/.rss'),
            
            # Research
            ('ArXiv CS.CL', 'https://arxiv.org/rss/cs.CL'),
            ('Papers with Code', 'https://rsshub.app/paperswithcode/latest'),
        ]
    
    def search_for_tools(self, query_terms):
        """Search for specific tools mentioned in news"""
        tool_searches = [
            'Nano Banana AI',
            'Claude 3.5',
            'GPT-4 Turbo',
            'Gemini 2.0',
            'Mistral Large',
            'Llama 3',
            'Stable Diffusion 3',
            'DALL-E 3',
            'Midjourney v6',
            'Cursor IDE',
            'v0 dev',
            'Windsurf IDE',
            'GitHub Copilot',
            'Devin AI',
            'OpenDevin'
        ]
        
        found_tools = []
        
        for tool_name in tool_searches:
            print(f"  🔍 Searching for {tool_name}...")
            
            # Search in HackerNews
            hn_url = f'https://hnrss.org/newest?q={tool_name.replace(" ", "+")}'
            feed = feedparser.parse(hn_url)
            
            for entry in feed.entries[:2]:
                if self.looks_like_tool(entry.title, entry.link):
                    found_tools.append({
                        'name': tool_name,
                        'title': entry.title,
                        'link': entry.link,
                        'source': 'HackerNews Search',
                        'entry': entry
                    })
        
        return found_tools
    
    def looks_like_tool(self, title, link):
        """Determine if this is likely a tool announcement"""
        tool_indicators = [
            'launch', 'release', 'available', 'announcing', 'introducing',
            'open source', 'api', 'sdk', 'library', 'framework',
            'github.com', 'npm', 'pypi', 'brew'
        ]
        
        title_lower = title.lower()
        return any(indicator in title_lower for indicator in tool_indicators)
    
    def analyze_entry(self, entry, source):
        """Smarter analysis of entries"""
        title = entry.get('title', '')
        link = entry.get('link', '')
        
        # Special handling for known tools
        if 'nano banana' in title.lower():
            description = self.extract_tool_description(entry, source)
            return self.create_tool_entry('Nano Banana', description, link, source, entry)
        
        # GitHub repos are always tools
        if 'github.com' in link and '/pull/' not in link and '/issues/' not in link:
            return self.analyze_github_tool(entry, source)
        
        # Product Hunt items are tools
        if 'producthunt' in source.lower():
            return self.analyze_product_hunt_tool(entry, source)
        
        # Show HN is usually a tool
        if title.startswith('Show HN:'):
            return self.analyze_show_hn_tool(entry, source)
        
        # Everything else
        return self.quick_classify_news(entry, source)
    
    def analyze_github_tool(self, entry, source):
        """Analyze GitHub repository as tool"""
        link = entry.get('link', '')
        title = entry.get('title', '')
        
        # Extract repo name
        parts = link.strip('/').split('/')
        if len(parts) >= 5:
            repo_name = parts[-1]
            owner = parts[-2]
        else:
            repo_name = title
            owner = 'unknown'
        
        # Get proper description
        description = self.extract_tool_description(entry, source)
        if not description or len(description) < 20:
            description = f"Open-source {repo_name} repository providing developer tools and libraries"
        
        tool_data = {
            'id': hashlib.md5(link.encode()).hexdigest(),
            'name': repo_name,
            'description': description,
            'tool_category': 'library',
            'keywords': ['github', 'opensource', 'code'],
            'implementation_keywords': ['git', 'clone', 'install'],
            'source': source,
            'source_url': link,
            'content_type': 'TOOL',
            'why_important': f'Open source {repo_name} gaining traction on GitHub',
            'immediate_action': f'git clone {link} and review README',
            'implementation_effort': 'days',
            'maturity_level': 'beta',
            'signal_strength': 'moderate',
            'ingested_at': datetime.now().isoformat()
        }
        
        print(f"    🔧 GitHub Tool: {repo_name} - {description[:60]}")
        return ('tool', tool_data)
    
    def analyze_product_hunt_tool(self, entry, source):
        """Analyze Product Hunt launch as tool"""
        name = entry.title.split(' - ')[0] if ' - ' in entry.title else entry.title
        
        # Get proper description
        description = self.extract_tool_description(entry, source)
        
        tool_data = {
            'id': hashlib.md5(entry.link.encode()).hexdigest(),
            'name': name,
            'description': description,
            'tool_category': 'platform',
            'keywords': ['product', 'launch', 'startup'],
            'source': source,
            'source_url': entry.link,
            'content_type': 'TOOL',
            'why_important': f'{name} launched on Product Hunt with community interest',
            'immediate_action': 'Visit product page and try demo if available',
            'implementation_effort': 'hours',
            'maturity_level': 'beta',
            'signal_strength': 'moderate',
            'ingested_at': datetime.now().isoformat()
        }
        
        print(f"    🔧 Product Hunt Tool: {name} - {description[:60]}")
        return ('tool', tool_data)
    
    def analyze_show_hn_tool(self, entry, source):
        """Analyze Show HN as tool"""
        name = entry.title.replace('Show HN:', '').strip().split('–')[0].strip()
        
        # Get proper description
        description = self.extract_tool_description(entry, source)
        
        tool_data = {
            'id': hashlib.md5(entry.link.encode()).hexdigest(),
            'name': name,
            'description': description,
            'tool_category': 'tool',
            'keywords': ['hackernews', 'launch', 'demo'],
            'source': source,
            'source_url': entry.link,
            'content_type': 'TOOL',
            'why_important': f'{name} showcased by developer on HackerNews',
            'immediate_action': 'Check out the demo and provide feedback',
            'implementation_effort': 'days',
            'maturity_level': 'experimental',
            'signal_strength': 'moderate',
            'ingested_at': datetime.now().isoformat()
        }
        
        print(f"    🔧 Show HN Tool: {name} - {description[:60]}")
        return ('tool', tool_data)
    
    def create_tool_entry(self, name, description, link, source, entry=None):
        """Create a tool entry for known tools"""
        
        # If we have the entry, extract better description
        if entry:
            description = self.extract_tool_description(entry, source)
        
        tool_data = {
            'id': hashlib.md5(link.encode()).hexdigest(),
            'name': name,
            'description': description,
            'tool_category': 'tool',
            'keywords': [name.lower()] + name.lower().split(),
            'source': source,
            'source_url': link,
            'content_type': 'TOOL',
            'why_important': f'{name} is gaining attention in the AI community',
            'immediate_action': f'Visit {link} to explore {name} capabilities',
            'implementation_effort': 'unknown',
            'maturity_level': 'beta',
            'signal_strength': 'high',
            'ingested_at': datetime.now().isoformat()
        }
        
        print(f"    🔧 Known Tool: {name} - {description[:60]}")
        return ('tool', tool_data)
    
    def quick_classify_news(self, entry, source):
        """Quick classification as news"""
        news_data = {
            'id': hashlib.md5(entry.link.encode()).hexdigest(),
            'headline': entry.title,
            'source': source,
            'source_url': entry.link,
            'content_type': 'NEWS',
            'ingested_at': datetime.now().isoformat()
        }
        
        return ('news', news_data)
    
    def ingest_from_feeds(self):
        """Main ingestion process"""
        feeds = self.get_all_feeds()
        
        new_tools = []
        new_news = []
        
        print(f"\n🚀 Checking {len(feeds)} RSS feeds")
        print("="*60)
        
        for feed_name, feed_url in feeds:
            print(f"\n📡 {feed_name}...")
            
            try:
                feed = feedparser.parse(feed_url)
                
                if not feed.entries:
                    print(f"  No entries found")
                    continue
                
                print(f"  Found {len(feed.entries)} entries")
                
                # Process entries
                for entry in feed.entries[:10]:  # Check more entries
                    if not entry.get('link'):
                        continue
                    
                    item_id = hashlib.md5(entry.link.encode()).hexdigest()
                    
                    # Skip if already processed
                    if item_id in self.tools or item_id in self.news:
                        continue
                    
                    # Analyze entry
                    result = self.analyze_entry(entry, source=feed_name)
                    
                    if result:
                        content_type, data = result
                        if content_type == 'tool':
                            self.tools[item_id] = data
                            new_tools.append(data)
                        else:
                            self.news[item_id] = data
                            new_news.append(data)
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:60]}")
        
        # Search for specific tools
        print("\n🔍 Searching for specific recent tools...")
        found_tools = self.search_for_tools(['Nano Banana', 'Claude 3.5', 'GPT-4'])
        for tool_info in found_tools:
            # Pass the entry for better description extraction
            result = self.create_tool_entry(
                tool_info['name'],
                '',  # Will be extracted
                tool_info['link'],
                tool_info['source'],
                tool_info.get('entry')
            )
            if result:
                _, data = result
                if data['id'] not in self.tools:
                    self.tools[data['id']] = data
                    new_tools.append(data)
        
        # Save
        self.save_data()
        
        print("\n" + "="*60)
        print(f"📊 Results:")
        print(f"  Tools: {len(self.tools)} total ({len(new_tools)} new)")
        print(f"  News: {len(self.news)} total ({len(new_news)} new)")
        
        if new_tools:
            print(f"\n🔧 New tools with descriptions:")
            for tool in new_tools[:10]:
                desc = tool.get('description', 'No description')[:60]
                print(f"  • {tool['name']}: {desc}...")
        
        return new_tools, new_news
    
    def save_data(self):
        """Save databases"""
        os.makedirs('data', exist_ok=True)
        
        with open(self.tools_db_path, 'w') as f:
            json.dump(self.tools, f, indent=2)
        
        with open(self.news_db_path, 'w') as f:
            json.dump(self.news, f, indent=2)

if __name__ == "__main__":
    ingester = EnhancedIngester()
    ingester.ingest_from_feeds()