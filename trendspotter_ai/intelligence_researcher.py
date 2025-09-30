import os
import json
import hashlib
from datetime import datetime
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
import feedparser
import requests

load_dotenv()

class IntelligenceResearcher:
    """Research tools and determine if hype or real"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.evidence_db = 'data/evidence.json'
        self.load_evidence()
    
    def load_evidence(self):
        if os.path.exists(self.evidence_db):
            with open(self.evidence_db, 'r') as f:
                self.evidence = json.load(f)
        else:
            self.evidence = {}
    
    def research_tool(self, tool_name, initial_source):
        """Deep research on a tool - is it hype or real?"""
        
        research = {
            'name': tool_name,
            'first_seen': initial_source,
            'research_date': datetime.now().isoformat(),
            'mentions': {'rss': 0, 'reddit': 0, 'github': 0, 'producthunt': 0, 'hackernews': 0},
            'evidence': [],
            'red_flags': [],
            'green_flags': [],
            'verdict': 'UNKNOWN',
            'confidence': 0,
            'implementation_examples': [],
            'reddit_buzz': 'UNKNOWN'
        }
        
        # Step 1: Check RSS frequency (hype indicator)
        rss_count = self.check_rss_frequency(tool_name)
        research['mentions']['rss'] = rss_count
        
        if rss_count > 10:
            research['red_flags'].append(f"Over-hyped: {rss_count} RSS mentions")
        elif rss_count > 0:
            research['green_flags'].append(f"Moderate coverage: {rss_count} mentions")
        
        # Step 2: Check Reddit discussions via RSS (reality check)
        reddit_data = self.check_reddit_via_rss(tool_name)
        research['mentions']['reddit'] = reddit_data['posts_found']
        research['reddit_buzz'] = reddit_data['buzz_level']
        
        if reddit_data['posts_found'] > 0:
            if reddit_data['sentiment'] == 'positive':
                research['green_flags'].append(f"Positive Reddit discussions ({reddit_data['posts_found']} posts)")
                research['evidence'].append({
                    'source': 'Reddit',
                    'type': 'community_validation',
                    'detail': reddit_data['summary']
                })
            elif reddit_data['sentiment'] == 'negative':
                research['red_flags'].append("Negative Reddit sentiment")
        
        # Step 3: Check HackerNews (implementation reality)
        hn_data = self.check_hackernews(tool_name)
        research['mentions']['hackernews'] = hn_data['posts']
        if hn_data['posts'] > 0:
            research['green_flags'].append(f"HackerNews discussions: {hn_data['posts']} posts")
            if hn_data['implementation_mentions'] > 0:
                research['evidence'].append({
                    'source': 'HackerNews',
                    'type': 'implementation_proof',
                    'detail': f"{hn_data['implementation_mentions']} users discussing actual usage"
                })
        
        # Step 4: Calculate verdict with Reddit signal
        research['verdict'], research['confidence'] = self.calculate_verdict_with_reddit(research)
        
        # Save evidence
        self.evidence[tool_name] = research
        self.save_evidence()
        
        return research
    
    def check_reddit_via_rss(self, tool_name):
        """Check Reddit discussions via RSS - no API needed"""
        
        reddit_data = {
            'posts_found': 0,
            'sentiment': 'neutral',
            'buzz_level': 'NO_BUZZ',
            'implementation_mentions': 0,
            'criticism_count': 0,
            'praise_count': 0,
            'summary': ''
        }
        
        # Target AI-focused subreddits
        subreddits = ['LocalLLaMA', 'OpenAI', 'singularity', 'MachineLearning', 'artificial']
        
        for sub in subreddits:
            try:
                # Use Reddit RSS search
                search_url = f'https://www.reddit.com/r/{sub}/search.rss?q={tool_name.replace(" ", "+")}&restrict_sr=on&sort=relevance&t=month'
                feed = feedparser.parse(search_url)
                
                for entry in feed.entries[:3]:  # Check top 3 posts per subreddit
                    reddit_data['posts_found'] += 1
                    
                    content_lower = entry.title.lower() + entry.get('summary', '').lower()
                    
                    # Look for implementation signals
                    implementation_keywords = ['using', 'built', 'deployed', 'implemented', 'production', 'tried']
                    if any(keyword in content_lower for keyword in implementation_keywords):
                        reddit_data['implementation_mentions'] += 1
                    
                    # Sentiment analysis
                    positive_keywords = ['amazing', 'great', 'excellent', 'impressive', 'game changer']
                    negative_keywords = ['overhyped', 'disappointing', 'waste', 'not worth', 'overrated']
                    
                    if any(keyword in content_lower for keyword in positive_keywords):
                        reddit_data['praise_count'] += 1
                    if any(keyword in content_lower for keyword in negative_keywords):
                        reddit_data['criticism_count'] += 1
            except:
                continue
        
        # Determine sentiment and buzz level
        if reddit_data['posts_found'] == 0:
            reddit_data['buzz_level'] = 'NO_BUZZ'
            reddit_data['summary'] = 'No Reddit discussions found'
        elif reddit_data['implementation_mentions'] >= 2:
            reddit_data['buzz_level'] = 'REAL_USAGE'
            reddit_data['sentiment'] = 'positive'
            reddit_data['summary'] = f"{reddit_data['implementation_mentions']} users report actual usage"
        elif reddit_data['praise_count'] > reddit_data['criticism_count']:
            reddit_data['buzz_level'] = 'POSITIVE_BUZZ'
            reddit_data['sentiment'] = 'positive'
            reddit_data['summary'] = 'Generally positive community sentiment'
        elif reddit_data['criticism_count'] > reddit_data['praise_count']:
            reddit_data['buzz_level'] = 'SKEPTICAL'
            reddit_data['sentiment'] = 'negative'
            reddit_data['summary'] = 'Community expressing skepticism'
        else:
            reddit_data['buzz_level'] = 'MIXED_SIGNALS'
            reddit_data['sentiment'] = 'neutral'
            reddit_data['summary'] = 'Mixed community reactions'
        
        return reddit_data
    
    def check_hackernews(self, tool_name):
        """Check HackerNews for technical discussions"""
        hn_data = {
            'posts': 0,
            'implementation_mentions': 0,
            'technical_discussions': 0
        }
        
        try:
            # Use HackerNews RSS
            hn_url = f'https://hnrss.org/newest?q={tool_name.replace(" ", "+")}'
            feed = feedparser.parse(hn_url)
            
            for entry in feed.entries[:10]:
                hn_data['posts'] += 1
                
                # Check for implementation discussions
                if any(word in entry.title.lower() for word in ['using', 'built', 'deployed', 'production']):
                    hn_data['implementation_mentions'] += 1
                
                # Check for technical depth
                if 'Show HN:' in entry.title or 'Ask HN:' in entry.title:
                    hn_data['technical_discussions'] += 1
        except:
            pass
        
        return hn_data
    
    def check_rss_frequency(self, tool_name):
        """Count RSS mentions across feeds"""
        feeds = [
            'https://feeds.feedburner.com/venturebeat/SZYF',
            'https://techcrunch.com/category/artificial-intelligence/feed/',
            'https://www.theverge.com/ai-artificial-intelligence/rss/index.xml'
        ]
        
        total_mentions = 0
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Check recent 20 items
                    if tool_name.lower() in entry.title.lower():
                        total_mentions += 1
            except:
                pass
        
        return total_mentions
    
    def calculate_verdict_with_reddit(self, research):
        """Enhanced verdict calculation with Reddit signal"""
        
        # Key indicators
        rss_mentions = research['mentions']['rss']
        reddit_buzz = research['reddit_buzz']
        reddit_posts = research['mentions']['reddit']
        hn_posts = research['mentions']['hackernews']
        evidence_count = len(research['evidence'])
        
        # High RSS + No Reddit/HN = HYPE
        if rss_mentions > 10 and reddit_buzz == 'NO_BUZZ' and hn_posts == 0:
            return "CONFIRMED_HYPE", 0.9
        
        # Real Reddit usage = REAL
        if reddit_buzz == 'REAL_USAGE':
            return "VERIFIED_REAL", 0.85
        
        # Good balance of coverage and discussion
        if rss_mentions > 0 and (reddit_posts > 0 or hn_posts > 0) and evidence_count >= 2:
            return "LIKELY_REAL", 0.7
        
        # Only press coverage = suspicious
        if rss_mentions > 5 and reddit_posts == 0 and hn_posts == 0:
            return "LIKELY_HYPE", 0.6
        
        # Negative community sentiment
        if reddit_buzz == 'SKEPTICAL':
            return "QUESTIONABLE", 0.5
        
        # Mixed or unclear signals
        if reddit_buzz == 'MIXED_SIGNALS':
            return "NEEDS_INVESTIGATION", 0.4
        
        return "INSUFFICIENT_DATA", 0.3
    
    def save_evidence(self):
        with open(self.evidence_db, 'w') as f:
            json.dump(self.evidence, f, indent=2)