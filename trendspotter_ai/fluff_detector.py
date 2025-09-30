import re
from typing import Dict, List, Tuple

class FluffDetector:
    def __init__(self):
        # Base marketing patterns - can be extended
        self.marketing_patterns = {
            'superlatives': [
                'revolutionary', 'game-changing', 'breakthrough',
                'cutting-edge', 'transformative', 'disruptive',
                'unprecedented', 'groundbreaking', 'paradigm-shifting'
            ],
            'vague_claims': [
                'next-generation', 'state-of-the-art', 'world-class',
                'industry-leading', 'best-in-class', 'enterprise-grade'
            ],
            'action_hype': [
                'unleash', 'supercharge', 'revolutionize',
                'transform', 'empower', 'accelerate'
            ]
        }
        
    def analyze_text(self, text: str) -> Dict:
        """Analyze text for marketing fluff"""
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        
        # Count marketing terms
        fluff_counts = {}
        total_fluff = 0
        
        for category, patterns in self.marketing_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text_lower)
            fluff_counts[category] = count
            total_fluff += count
        
        # Calculate ratios
        fluff_ratio = total_fluff / max(word_count, 1)
        
        # Detect unverifiable claims
        claim_patterns = [
            r'\d+x\s+(faster|better|more)',  # "10x faster"
            r'\d+%\s+(improvement|increase|better)',  # "50% improvement"
            r'(first|only|unique)\s+in\s+the\s+(world|industry)',
        ]
        
        unverifiable_claims = sum(1 for pattern in claim_patterns 
                                 if re.search(pattern, text_lower))
        
        return {
            'fluff_ratio': min(fluff_ratio, 1.0),
            'fluff_counts': fluff_counts,
            'unverifiable_claims': unverifiable_claims,
            'credibility_penalty': min(fluff_ratio * 0.3, 0.3),  # Max 30% penalty
            'cleaned_text': self.remove_fluff(text)
        }
    
    def remove_fluff(self, text: str) -> str:
        """Remove marketing speak and return factual content"""
        # This would be enhanced with AI
        cleaned = text
        for patterns in self.marketing_patterns.values():
            for pattern in patterns:
                cleaned = re.sub(r'\b' + pattern + r'\b', '', cleaned, flags=re.IGNORECASE)
        return ' '.join(cleaned.split())  # Clean up extra spaces