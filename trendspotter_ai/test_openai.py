import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

def test_openai_connection():
    """Test if OpenAI is working"""
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ No API key found in .env file")
        return False
    
    print(f"✓ API key found: {api_key[:8]}...")
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Simple test
        print("\nTesting basic completion...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Return the JSON: {\"status\": \"working\"}"}
            ],
            temperature=0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"Response: {result}")
        
        # Test JSON parsing
        print("\nTesting tool analysis...")
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a tech analyst. Always return valid JSON."},
                {"role": "user", "content": """
                Analyze this tool:
                Title: Integrity - AI-powered project management
                
                Return JSON:
                {
                    "name": "tool name",
                    "why_important": "one sentence why this matters",
                    "immediate_action": "what to do next",
                    "signal_strength": "high"
                }
                """}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        result = test_response.choices[0].message.content
        print(f"Tool analysis response: {result}")
        
        # Try to parse it
        parsed = json.loads(result)
        print(f"✅ Successfully parsed: {parsed['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        return False

def check_current_data():
    """Check what's actually in your database"""
    try:
        with open('data/tools.json', 'r') as f:
            tools = json.load(f)
        
        print(f"\n📊 Current database status:")
        print(f"Total items: {len(tools)}")
        
        # Check first few items
        for i, (key, tool) in enumerate(list(tools.items())[:3]):
            print(f"\nItem {i+1}:")
            print(f"  Name: {tool.get('name', 'MISSING')}")
            print(f"  Why Important: {tool.get('why_important', 'MISSING')[:50]}...")
            print(f"  Source: {tool.get('source', 'MISSING')}")
            print(f"  Type: {tool.get('content_type', 'UNKNOWN')}")
            
    except Exception as e:
        print(f"Error reading tools.json: {e}")

if __name__ == "__main__":
    print("🔍 Testing OpenAI Integration\n")
    print("="*50)
    
    if test_openai_connection():
        print("\n✅ OpenAI is working correctly!")
    else:
        print("\n❌ OpenAI integration has issues")
    
    check_current_data()