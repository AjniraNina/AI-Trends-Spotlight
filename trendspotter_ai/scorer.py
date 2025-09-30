import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from dataclasses import dataclass

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Common-Sense Tool Gateway (NLP + rules, optional embeddings)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    is_tool: bool
    confidence: float
    reasons: List[str]

class CommonSenseToolGate:
    """
    Decide if an item is an adoptable tool (SDK/API/CLI/platform/app)
    using simple rules + optional semantic similarity.
    """

    # strong positive indicators (tool-like)
    POSITIVE_WORDS = {
        "api","sdk","library","framework","cli","package","plugin","extension",
        "tool","service","platform","server","self-hosted","open source","open-source",
        "repo","repository","npm","pypi","pip","docker","helm","kubernetes","endpoint",
        "install","setup","configure","deploy","integration","docs","readme","examples",
        "code","source code","git clone","import","build","compile"
    }

    # strong negative indicators (game/news-like)
    NEGATIVE_WORDS = {
        "game","play now","play the game","leaderboard","score","level","3d maze",
        "steam","roblox","itch.io","trailer","soundtrack","walkthrough","speedrun",
        "wikispeedia","levels","skins","xp"
    }

    # URL/domain hints
    TOOL_DOMAINS = (
        "github.com","gitlab.com","bitbucket.org","npmjs.com","pypi.org","crates.io",
        "docker.com","hub.docker.com","readthedocs.io"
    )
    GAME_DOMAINS = ("roblox.com","store.steampowered.com","itch.io","epicgames.com")

    TOOL_PROTOTYPE = "A software tool or developer resource you can adopt, like an SDK, API, CLI, framework, library or deployable app."
    NOT_TOOL_PROTOTYPE = "A game, demo, news post, opinion article, or general content not directly adoptable as a developer tool."

    def __init__(self, enable_embeddings: bool = True):
        self.enable_embeddings = enable_embeddings and bool(os.getenv("OPENAI_API_KEY"))
        self.client = OpenAI() if self.enable_embeddings else None
        self._tool_vec = None
        self._not_tool_vec = None

    def _embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model="text-embedding-3-small", input=text[:3000])
        return resp.data[0].embedding

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na*nb + 1e-9)

    def _semantic_vote(self, text: str) -> Tuple[float, List[str]]:
        if not self.enable_embeddings:
            return 0.0, ["embeddings disabled; semantic vote skipped"]
        if self._tool_vec is None:
            self._tool_vec = self._embed(self.TOOL_PROTOTYPE)
            self._not_tool_vec = self._embed(self.NOT_TOOL_PROTOTYPE)
        vec = self._embed(text)
        s_tool = self._cosine(vec, self._tool_vec)
        s_not = self._cosine(vec, self._not_tool_vec)
        score = s_tool - s_not  # >0 favors tool
        reasons = [f"semantic tool score={s_tool:.2f}", f"semantic not-tool score={s_not:.2f}"]
        return score, reasons

    def decide(self, item: Dict) -> GateResult:
        title = (item.get("title") or item.get("name") or "").lower()
        desc = (item.get("description") or "").lower()
        url  = (item.get("source_url") or item.get("url") or "")
        text = f"{title} {desc}"

        reasons: List[str] = []

        # domain vote
        domain_score = 0
        if any(d in url for d in self.TOOL_DOMAINS):
            domain_score += 2; reasons.append("tool domain detected")
        if any(d in url for d in self.GAME_DOMAINS):
            domain_score -= 2; reasons.append("game domain detected")

        # lexical vote
        pos_hits = sum(1 for w in self.POSITIVE_WORDS if w in text)
        neg_hits = sum(1 for w in self.NEGATIVE_WORDS if w in text)
        lex_score = (pos_hits * 0.6) - (neg_hits * 0.8)
        if pos_hits: reasons.append(f"{pos_hits} tool-like terms")
        if neg_hits: reasons.append(f"{neg_hits} game-like terms")

        # artifact cues (very strong)
        artifact_score = 0
        if re.search(r"\b(git clone|pip install|npm i|npm install|docker run|helm install)\b", text):
            artifact_score += 2; reasons.append("install command detected")
        if "readme" in text or "documentation" in text or "docs" in text:
            artifact_score += 1; reasons.append("docs/readme mentioned")

        # semantic vote (optional)
        sem_score = 0.0
        sem_reasons: List[str] = []
        try:
            sem_score, sem_reasons = self._semantic_vote(text[:3000])
            reasons.extend(sem_reasons)
        except Exception as e:
            reasons.append(f"semantic check error: {str(e)[:60]}")

        # aggregate
        combined = domain_score + lex_score + artifact_score + max(min(sem_score, 1.0), -1.0)

        # decision mapping
        if combined >= 1.5:
            return GateResult(True, min(0.9, 0.55 + combined/6), reasons)
        elif combined <= 0.0:
            return GateResult(False, min(0.9, 0.55 + abs(combined)/6), reasons)
        else:
            return GateResult(False, 0.5, reasons)

# ──────────────────────────────────────────────────────────────────────────────
# Existing components (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class SignalClassifier:
    """Classify tools according to clear signal definitions"""
    
    def __init__(self, strict_mode=True):
        self.strict_mode = strict_mode
        self.definitions = {
            'GAME_CHANGING': {
                'indicators': [
                    'new model release',
                    'breakthrough performance',
                    'open sourced',
                    'replaces entire workflow',
                    'first to market',
                    '10x improvement'
                ],
                'impact_threshold': 80,
                'actionability': 'immediate'
            },
            'HIGH': {
                'indicators': [
                    'production ready',
                    'major version release',
                    'significant cost reduction',
                    'adopted by major company',
                    'solves known pain point'
                ],
                'impact_threshold': 60,
                'actionability': 'within_quarter'
            },
            'PROMISING': {  # NEW: Between MODERATE and NOISE
                'indicators': [
                    'early stage',
                    'prototype',
                    'novel approach',
                    'creative solution',
                    'experimental',
                    'high potential'
                ],
                'impact_threshold': 35,
                'actionability': 'track_progress'
            },
            'MODERATE': {
                'indicators': [
                    'incremental improvement',
                    'minor feature',
                    'beta release',
                    'interesting approach',
                    'worth monitoring'
                ],
                'impact_threshold': 40,
                'actionability': 'evaluate_later'
            },
            'NOISE': {
                'indicators': [
                    'no specific details',
                    'unverifiable claims',
                    'pure opinion',
                    'marketing speak only',
                    'no technical substance'
                ],
                'impact_threshold': 20,
                'actionability': 'ignore'
            }
        }
    
    def classify_signal(self, item: Dict) -> Dict:
        """Classify item according to the rubric"""
        
        print(f"  📊 Signal Classification (strict={self.strict_mode}):")
        
        # Check for game-changing signals
        if self.is_game_changing(item):
            print(f"    → GAME_CHANGING: Major breakthrough detected")
            return self.create_signal_assessment('GAME_CHANGING', item)
        
        # Check for high signal
        if self.is_high_signal(item):
            print(f"    → HIGH: Production-ready value detected")
            return self.create_signal_assessment('HIGH', item)
        
        # Check for noise (now requires more red flags)
        if self.is_noise(item):
            print(f"    → NOISE: Marketing fluff detected")
            return self.create_signal_assessment('NOISE', item)
        
        # Check for promising (NEW)
        if self.is_promising(item):
            print(f"    → PROMISING: Early-stage potential detected")
            return self.create_signal_assessment('PROMISING', item)
        
        # Default to moderate
        print(f"    → MODERATE: Incremental improvement")
        return self.create_signal_assessment('MODERATE', item)
    
    def is_game_changing(self, item: Dict) -> bool:
        """Detect truly game-changing developments"""
        game_changing_patterns = [
            (r'new.*model.*release', r'gpt-5|claude-4|gemini-2'),
            (r'open.*sourc', r'previously.*proprietary'),
            (r'acquisition', r'billion|major.*player'),
            (r'benchmark', r'beats.*human|state.*art'),
            (r'cost.*reduction', r'90%|order.*magnitude')
        ]
        
        content = f"{item.get('title', '')} {item.get('description', '')}".lower()
        
        # Check patterns
        pattern_matches = 0
        for pattern, qualifier in game_changing_patterns:
            if re.search(pattern, content) and re.search(qualifier, content):
                pattern_matches += 1
        
        # Check evidence quality
        has_verification = any([
            item.get('github_stars', 0) > 1000,
            item.get('production_deployments', 0) > 0,
            'benchmark' in item.get('evidence', []),
            item.get('company_adoption', [])
        ])
        
        # Less strict in discovery mode
        threshold = 1 if not self.strict_mode else 2
        return pattern_matches >= threshold or (pattern_matches >= 1 and has_verification)
    
    def is_high_signal(self, item: Dict) -> bool:
        """Detect high-value signals with adaptive thresholds"""
        high_signals = {
            'technical_depth': (item.get('technical_depth', 0) > 7, 2),  # (condition, weight)
            'implementation_ready': (item.get('maturity_level') == 'production', 2),
            'solves_real_problem': (item.get('problem_solution_fit', 0) > 70, 2),
            'has_evidence': (len(item.get('evidence', [])) > 2, 1),
            'actionable': (item.get('immediate_action') is not None, 1)
        }
        
        # Weighted scoring instead of hard thresholds
        weighted_score = sum(weight for (condition, weight) in high_signals.values() if condition)
        
        # Adaptive threshold based on mode
        threshold = 4 if self.strict_mode else 3
        return weighted_score >= threshold
    
    def is_promising(self, item: Dict) -> bool:
        """Detect promising early-stage tools (NEW)"""
        promising_indicators = {
            'novel_approach': any(word in str(item).lower() for word in ['novel', 'new approach', 'innovative']),
            'creative_domain': any(word in str(item).lower() for word in ['creative', 'design', 'art', 'assets']),
            'early_traction': item.get('github_stars', 0) > 10,
            'recent_release': 'alpha' in str(item).lower() or 'beta' in str(item).lower(),
            'potential_identified': item.get('technical_depth', 0) >= 5
        }
        
        matches = sum(1 for indicator in promising_indicators.values() if indicator)
        return matches >= 2
    
    def is_noise(self, item: Dict) -> bool:
        """Detect marketing fluff and noise - now less strict"""
        noise_indicators = {
            'no_technical_details': item.get('technical_depth', 0) < 3,
            'high_fluff_ratio': item.get('fluff_ratio', 0) > 0.4,  # Raised from 0.3
            'no_evidence': len(item.get('evidence', [])) == 0,
            'opinion_piece': item.get('content_type') == 'OPINION',
            'unverifiable': item.get('unverifiable_claims', 0) > 3,  # Raised from 2
            'no_github': not item.get('github_url') and 'github' not in str(item).lower()
        }
        
        noise_count = sum(1 for indicator in noise_indicators.values() if indicator)
        
        # Require more red flags in discovery mode
        threshold = 4 if not self.strict_mode else 3
        return noise_count >= threshold
    
    def create_signal_assessment(self, level: str, item: Dict) -> Dict:
        """Create detailed signal assessment"""
        return {
            'signal_level': level,
            'confidence': self.calculate_confidence(item),
            'actionability': self.definitions[level]['actionability'],
            'why_this_level': self.explain_classification(level, item),
            'immediate_value': self.calculate_immediate_value(level, item),
            'strategic_value': self.calculate_strategic_value(level, item)
        }
    
    def calculate_confidence(self, item: Dict) -> float:
        """Calculate confidence in classification based on actual available data"""
        confidence = 0.0
        
        # AI classification confidence (if AI was used)
        if item.get('tool_confirmed') is not None:
            confidence += 0.3  # AI made a determination
        
        # Technical depth from AI (higher depth = more confidence)
        tech_depth = item.get('technical_depth', 0)
        if tech_depth > 0:
            confidence += min(tech_depth * 0.08, 0.4)  # Max 0.4 from tech depth
        
        # Practicality score from AI
        practicality = item.get('practicality_score', 0)
        if practicality > 0:
            confidence += min(practicality * 0.05, 0.2)  # Max 0.2 from practicality
        
        # Category detection (specific category = higher confidence)
        if item.get('category') and item.get('category') != 'General':
            confidence += 0.1
        
        # Primary capabilities identified
        capabilities = item.get('primary_capabilities', [])
        if len(capabilities) > 0:
            confidence += min(len(capabilities) * 0.05, 0.15)  # Max 0.15 from capabilities
        
        # Feed repetition (trending items = higher confidence)
        feed_rep = item.get('feed_repetition', 'low')
        if feed_rep == 'high':
            confidence += 0.15
        elif feed_rep == 'medium':
            confidence += 0.08
        
        # Content type classification
        content_type = item.get('content_type', 'UNKNOWN')
        if content_type in ['TOOL', 'ANNOUNCEMENT']:
            confidence += 0.1
        elif content_type == 'CASE_STUDY':
            confidence += 0.05
        
        # If we have very little data, be honest about low confidence
        if tech_depth == 0 and practicality == 0 and not capabilities:
            confidence = max(confidence, 0.15)  # Minimum 15% for anything we tried to score
        
        return min(confidence, 0.95)  # Cap at 95% - never fully certain
    
    def explain_classification(self, level: str, item: Dict) -> str:
        """Explain why this classification was chosen"""
        explanations = {
            'GAME_CHANGING': f"Major breakthrough that could reshape workflows",
            'HIGH': f"Production-ready solution to real problems",
            'PROMISING': f"Early-stage tool with high potential",
            'MODERATE': f"Incremental improvement worth monitoring",
            'NOISE': f"Marketing content without substance"
        }
        return explanations.get(level, "Unknown classification")
    
    def calculate_immediate_value(self, level: str, item: Dict) -> int:
        """Calculate immediate value (0-100)"""
        values = {
            'GAME_CHANGING': 90,
            'HIGH': 70,
            'PROMISING': 55,
            'MODERATE': 40,
            'NOISE': 10
        }
        return values.get(level, 30)
    
    def calculate_strategic_value(self, level: str, item: Dict) -> int:
        """Calculate long-term strategic value (0-100)"""
        values = {
            'GAME_CHANGING': 100,
            'HIGH': 60,
            'PROMISING': 45,
            'MODERATE': 30,
            'NOISE': 0
        }
        return values.get(level, 20)


class FluffDetector:
    """Detect and remove marketing fluff"""
    
    def __init__(self):
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
        if not text:
            return {'fluff_ratio': 0, 'fluff_counts': {}, 'unverifiable_claims': 0, 'credibility_penalty': 0}
            
        text_lower = text.lower()
        words = text_lower.split()
        word_count = max(len(words), 1)
        
        # Count marketing terms
        fluff_counts = {}
        total_fluff = 0
        
        for category, patterns in self.marketing_patterns.items():
            count = sum(1 for pattern in patterns if pattern in text_lower)
            fluff_counts[category] = count
            total_fluff += count
        
        # Calculate ratios
        fluff_ratio = min(total_fluff / word_count, 1.0)
        
        # Detect unverifiable claims
        claim_patterns = [
            r'\d+x\s+(faster|better|more)',
            r'\d+%\s+(improvement|increase|better)',
            r'(first|only|unique)\s+in\s+the\s+(world|industry)',
        ]
        
        unverifiable_claims = sum(
            1 for pattern in claim_patterns 
            if re.search(pattern, text_lower)
        )
        
        if fluff_ratio > 0:
            print(f"    🎭 Fluff detected: {fluff_ratio:.2%} marketing speak, {unverifiable_claims} unverifiable claims")
        
        return {
            'fluff_ratio': fluff_ratio,
            'fluff_counts': fluff_counts,
            'unverifiable_claims': unverifiable_claims,
            'credibility_penalty': min(fluff_ratio * 0.3, 0.3),
            'cleaned_text': self.remove_fluff(text)
        }
    
    def remove_fluff(self, text: str) -> str:
        """Remove marketing speak and return factual content"""
        cleaned = text
        for patterns in self.marketing_patterns.values():
            for pattern in patterns:
                cleaned = re.sub(r'\b' + pattern + r'\b', '', cleaned, flags=re.IGNORECASE)
        return ' '.join(cleaned.split())


class ROICalculator:
    """Calculate return on investment for tools and the system itself"""
    
    def __init__(self, hourly_rate: int = 75, team_size: int = 5, strict_mode=True):
        self.hourly_rate = hourly_rate
        self.team_size = team_size
        self.monthly_tool_evaluation_hours = 40
        self.strict_mode = strict_mode
    
    def calculate_system_roi(self) -> Dict:
        """ROI for the scoring system itself"""
        
        # One-time costs
        development_costs = {
            'initial_development': 10 * self.hourly_rate,
            'testing_and_refinement': 5 * self.hourly_rate,
            'documentation': 2 * self.hourly_rate
        }
        
        # Monthly recurring costs
        monthly_costs = {
            'api_calls': 10,  # Claude API for 200 tools
            'maintenance': 1 * self.hourly_rate,
            'hosting': 5,
            'feed_subscriptions': 0
        }
        
        # Monthly benefits
        monthly_benefits = {
            'time_saved': self.monthly_tool_evaluation_hours * self.hourly_rate,
            'early_detection_value': 500,
            'avoided_bad_tools': 1000,
            'better_decisions': 300
        }
        
        # Calculate payback period
        total_setup = sum(development_costs.values())
        monthly_net = sum(monthly_benefits.values()) - sum(monthly_costs.values())
        payback_months = total_setup / monthly_net if monthly_net > 0 else float('inf')
        
        return {
            'setup_cost': total_setup,
            'monthly_cost': sum(monthly_costs.values()),
            'monthly_benefit': sum(monthly_benefits.values()),
            'monthly_net': monthly_net,
            'payback_months': payback_months,
            'annual_roi': (monthly_net * 12 - total_setup) / total_setup * 100 if total_setup > 0 else 0,
            'break_even_date': datetime.now() + timedelta(days=payback_months * 30)
        }
    
    def analyze_tool_worth(self, tool: Dict, workflow: Dict, current_state: Dict) -> Dict:
        """Detailed cost-benefit for individual tools"""
        
        # Detect if this is a creative/design tool
        text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
        is_creative = any(word in text for word in ['creative', 'design', 'art', 'assets', 'visual', 'graphics', 'animation'])
        
        # Extract costs
        monthly_cost = self.extract_monthly_cost(tool)
        integration_hours = self.estimate_integration_hours(tool)
        
        # Implementation costs
        implementation_costs = {
            'licensing': monthly_cost * 12,  # Annual
            'integration': integration_hours * self.hourly_rate,
            'training': self.team_size * 4 * self.hourly_rate,
            'migration': self.estimate_migration_effort(tool, current_state),
            'risk_buffer': 0  # Will be calculated
        }
        
        # Hidden costs (reduced for creative tools in discovery mode)
        penalty = 0.5 if (is_creative and not self.strict_mode) else 1.0
        hidden_costs = {
            'vendor_lock_in': monthly_cost * 6 * penalty,
            'technical_debt': monthly_cost * 0.1 * 12 * penalty,
            'opportunity_cost': self.hourly_rate * 20 * penalty,
            'disruption_cost': self.team_size * 2 * self.hourly_rate * penalty
        }
        
        # Benefits (boosted for creative tools)
        creativity_multiplier = 1.5 if is_creative else 1.0
        time_saved = tool.get('time_saved_per_month', 10)
        direct_benefits = {
            'time_saved_monthly': time_saved * self.hourly_rate * creativity_multiplier,
            'quality_improvement': time_saved * self.hourly_rate * 0.2 * creativity_multiplier,
            'cost_reduction': self.calculate_replaced_tools_savings(tool, current_state),
            'revenue_impact': 100 if is_creative else 0  # Creative tools have indirect revenue impact
        }
        
        # Boosted indirect benefits for creative tools
        indirect_benefits = {
            'team_satisfaction': 200 if is_creative else 100,
            'competitive_advantage': 400 if is_creative else 200,
            'learning_value': self.team_size * (100 if is_creative else 50),
            'option_value': 200 if is_creative else 100
        }
        
        # Risk adjustment (less punitive in discovery mode)
        risk_factors = self.calculate_risk_factors(tool)
        risk_multiplier = 0.3 if not self.strict_mode else 0.5
        implementation_costs['risk_buffer'] = sum(implementation_costs.values()) * risk_factors['failure_probability'] * risk_multiplier
        
        # Calculate metrics
        total_implementation = sum(implementation_costs.values()) + sum(hidden_costs.values())
        monthly_net = sum(direct_benefits.values()) - monthly_cost
        payback_months = total_implementation / monthly_net if monthly_net > 0 else 999
        
        print(f"    💰 ROI: {payback_months:.1f} month payback, ${monthly_net:.0f}/month benefit")
        if is_creative:
            print(f"    🎨 Creative tool detected - using relaxed evaluation")
        
        return {
            'total_implementation_cost': total_implementation,
            'monthly_benefit': sum(direct_benefits.values()),
            'monthly_cost': monthly_cost,
            'payback_months': payback_months,
            'two_year_roi': (monthly_net * 24 - total_implementation) / total_implementation * 100 if total_implementation > 0 else -100,
            'risk_factors': risk_factors,
            'is_creative': is_creative,
            'recommendation': self.get_financial_recommendation(payback_months, risk_factors, is_creative)
        }
    
    def extract_monthly_cost(self, tool: Dict) -> float:
        """Extract numerical cost from tool data"""
        cost_str = tool.get('cost_benefit', {}).get('monthly_cost', '0')
        if isinstance(cost_str, (int, float)):
            return float(cost_str)
        
        # Parse string costs
        numbers = re.findall(r'\d+', str(cost_str))
        if numbers:
            return float(numbers[0])
        return 0
    
    def estimate_integration_hours(self, tool: Dict) -> int:
        """Estimate integration effort in hours"""
        effort_map = {
            'hours': 8,
            'days': 40,
            'weeks': 160,
            'months': 640
        }
        effort = tool.get('implementation_effort', 'weeks')
        return effort_map.get(effort, 160)
    
    def estimate_migration_effort(self, tool: Dict, current_state: Dict) -> float:
        """Estimate migration costs"""
        # Simple estimate based on current tools
        if current_state.get('existing_solution'):
            return self.hourly_rate * 40  # Week of migration
        return 0
    
    def calculate_replaced_tools_savings(self, tool: Dict, current_state: Dict) -> float:
        """Calculate savings from replaced tools"""
        replaced_cost = current_state.get('current_monthly_cost', 0)
        return replaced_cost * 0.8  # Assume 80% cost reduction
    
    def calculate_risk_factors(self, tool: Dict) -> Dict:
        """Quantify implementation risks"""
        risks = {
            'technical_risk': 0.1 if tool.get('maturity_level') == 'production' else 0.3,
            'vendor_risk': 0.1 if tool.get('github_stars', 0) > 1000 else 0.2,
            'adoption_risk': 0.1 if tool.get('implementation_effort') == 'hours' else 0.25,
            'integration_risk': 0.1 if 'api' in tool.get('keywords', []) else 0.3
        }
        
        # Combined failure probability
        failure_probability = 1 - np.prod([1 - r for r in risks.values()])
        
        return {
            'individual_risks': risks,
            'failure_probability': failure_probability,
            'confidence_level': 1 - failure_probability
        }
    
    def get_financial_recommendation(self, payback_months: float, risk_factors: Dict, is_creative=False) -> str:
        """Get recommendation based on financial analysis - less strict for creative tools"""
        
        # Creative/design tools get special treatment
        if is_creative and not self.strict_mode:
            if payback_months < 6:
                return 'IMPLEMENT_NOW'
            elif payback_months < 18:  # Much more lenient for creative tools
                return 'PILOT_FIRST'
            elif payback_months < 24:
                return 'EVALUATE_QUARTERLY'
            else:
                return 'TRACK_INNOVATION'
        
        # Standard evaluation (also relaxed in discovery mode)
        if self.strict_mode:
            # Original strict thresholds
            if payback_months < 3 and risk_factors['confidence_level'] > 0.7:
                return 'IMPLEMENT_NOW'
            elif payback_months < 6:
                return 'PILOT_FIRST'
            elif payback_months < 12:
                return 'EVALUATE_QUARTERLY'
            else:
                return 'WAIT_OR_SKIP'
        else:
            # Relaxed thresholds for discovery mode
            if payback_months < 6 and risk_factors['confidence_level'] > 0.5:
                return 'IMPLEMENT_NOW'
            elif payback_months < 12:
                return 'PILOT_FIRST'
            elif payback_months < 24:
                return 'EVALUATE_QUARTERLY'
            else:
                return 'MONITOR_PROGRESS'


class IntelligentClassifier:
    """AI-powered content classification with robust fallback"""
    
    def __init__(self):
        self.client = None
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.client = OpenAI()
                print("  ✓ AI Classifier initialized")
            except Exception as e:
                print(f"  ⚠ AI Classifier not available: {e}")
    
    def classify_and_extract(self, item: Dict) -> Dict:
        """Use AI to deeply analyze content with smart fallback"""
        
        # Try AI classification if available
        if self.client:
            prompt = f"""
            Analyze this RSS feed item. Return ONLY valid JSON:
            
            Title: {item.get('title', 'Unknown')[:100]}
            Content: {item.get('description', '')[:400]}
            
            {{
                "content_type": "TOOL",
                "category": "Dev Tooling",
                "primary_capabilities": ["capability1", "capability2"],
                "implementation_evidence": [],
                "hype_indicators": [],
                "concrete_benefits": [],
                "technical_depth": 5,
                "practicality_score": 5,
                "tool_confirmed": true,
                "time_to_value": "weeks",
                "feed_repetition": "low"
            }}
            
            content_type: TOOL, ANNOUNCEMENT, CASE_STUDY, OPINION, or RESEARCH
            category: AI/LLM, Creative/Assets, Dev Tooling, Data/Analytics, Security, Web/App, or General
            technical_depth: 0-10 (how technical is it?)
            practicality_score: 0-10 (how practical/usable?)
            feed_repetition: low, medium, or high
            """
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a JSON API. Return only valid JSON, no other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Ensure required fields exist with defaults
                result.setdefault('content_type', 'UNKNOWN')
                result.setdefault('category', 'General')
                result.setdefault('primary_capabilities', [])
                result.setdefault('technical_depth', 5)
                result.setdefault('practicality_score', 5)
                result.setdefault('tool_confirmed', False)
                result.setdefault('time_to_value', 'weeks')
                result.setdefault('feed_repetition', 'low')
                result.setdefault('implementation_evidence', [])
                result.setdefault('hype_indicators', [])
                result.setdefault('concrete_benefits', [])
                
                # Validate and fix values
                if result['technical_depth'] not in range(0, 11):
                    result['technical_depth'] = 5
                if result['practicality_score'] not in range(0, 11):
                    result['practicality_score'] = 5
                if result['feed_repetition'] not in ['low', 'medium', 'high']:
                    result['feed_repetition'] = 'low'
                    
                print(f"    🤖 AI Analysis: {result.get('content_type')} | cat={result.get('category')} | depth={result.get('technical_depth')} | feed_rep={result.get('feed_repetition')}")
                return result
            except Exception as e:
                print(f"    ⚠ AI failed, using heuristics: {str(e)[:50]}")
        
        # Intelligent heuristic fallback
        return self.heuristic_classification(item)
    
    def heuristic_classification(self, item: Dict) -> Dict:
        """Smart heuristic classification when AI is unavailable"""
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        
        # Determine content type from keywords
        if any(word in text for word in ['api', 'library', 'framework', 'tool', 'sdk', 'cli', 'plugin']):
            content_type = 'TOOL'
            category = 'Dev Tooling'
        elif any(word in text for word in ['announce', 'release', 'launch', 'introduce', 'unveil']):
            content_type = 'ANNOUNCEMENT'
            category = 'General'
        elif any(word in text for word in ['case study', 'implementation', 'deployed', 'using', 'built with']):
            content_type = 'CASE_STUDY'
            category = 'General'
        elif any(word in text for word in ['research', 'paper', 'study', 'analysis', 'findings']):
            content_type = 'RESEARCH'
            category = 'General'
        elif any(word in text for word in ['opinion', 'think', 'believe', 'perspective']):
            content_type = 'OPINION'
            category = 'General'
        else:
            content_type = 'UNKNOWN'
            category = 'General'
        
        # Detect category from content
        if any(word in text for word in ['ai', 'llm', 'gpt', 'claude', 'gemini', 'neural', 'ml']):
            category = 'AI/LLM'
        elif any(word in text for word in ['creative', 'design', 'art', 'assets', 'graphics']):
            category = 'Creative/Assets'
        elif any(word in text for word in ['message', 'chat', 'whatsapp', 'slack', 'email']):
            category = 'Messaging/Comms'
        elif any(word in text for word in ['data', 'analytics', 'metrics', 'dashboard']):
            category = 'Data/Analytics'
        elif any(word in text for word in ['security', 'auth', 'encryption', 'vulnerability']):
            category = 'Security'
        elif any(word in text for word in ['cloud', 'aws', 'azure', 'kubernetes', 'docker']):
            category = 'Infra/Cloud'
        elif any(word in text for word in ['web', 'app', 'mobile', 'frontend', 'backend']):
            category = 'Web/App'
        elif any(word in text for word in ['game', 'roblox', 'unity', 'unreal']):
            category = 'Game/Roblox'
        
        # Estimate technical depth from content
        technical_indicators = ['api', 'sdk', 'implementation', 'architecture', 'algorithm', 
                               'framework', 'database', 'infrastructure', 'deployment', 'kubernetes']
        technical_depth = sum(1 for ind in technical_indicators if ind in text)
        technical_depth = min(technical_depth, 10)
        
        # Estimate practicality  
        practical_indicators = ['production', 'deployed', 'customers', 'users', 'saves time',
                               'reduces cost', 'improves', 'automates', 'simplifies']
        practicality_score = sum(1 for ind in practical_indicators if ind in text)
        practicality_score = min(practicality_score, 10)
        
        # Extract capabilities
        primary_capabilities = []
        if 'automate' in text:
            primary_capabilities.append("Automation")
        if 'analyze' in text or 'analytics' in text:
            primary_capabilities.append("Analysis")
        if 'generate' in text or 'create' in text:
            primary_capabilities.append("Generation")
        if 'monitor' in text or 'track' in text:
            primary_capabilities.append("Monitoring")
        if 'integrate' in text or 'connect' in text:
            primary_capabilities.append("Integration")
        
        # Find implementation evidence
        implementation_evidence = []
        if 'production' in text:
            implementation_evidence.append("Production deployment mentioned")
        if any(word in text for word in ['customer', 'client', 'user']):
            implementation_evidence.append("Real users mentioned")
        
        # Detect hype
        hype_indicators = []
        if any(word in text for word in ['revolutionary', 'game-changing', 'breakthrough']):
            hype_indicators.append("Marketing superlatives detected")
        
        print(f"    🔍 Heuristic Analysis: {content_type} | cat={category} | depth={technical_depth} | prac={practicality_score}")
        
        return {
            'content_type': content_type,
            'category': category,
            'primary_capabilities': primary_capabilities[:5] if primary_capabilities else [],
            'implementation_evidence': implementation_evidence,
            'hype_indicators': hype_indicators,
            'concrete_benefits': [],
            'technical_depth': technical_depth,
            'practicality_score': practicality_score,
            'tool_confirmed': (content_type in ["TOOL", "CASE_STUDY"]),
            'time_to_value': 'weeks' if practicality_score >= 6 else 'months',
            'feed_repetition': 'low'  # Default to low when heuristic
        }


class EnhancedWorkflowScorer:
    """Complete scoring system combining all components with detailed logging"""
    
    def __init__(self, strict_mode=True):
        """Initialize scorer with optional discovery mode (strict=False)"""
        self.strict_mode = strict_mode
        self.signal_classifier = SignalClassifier(strict_mode=strict_mode)
        self.fluff_detector = FluffDetector()
        self.roi_calculator = ROICalculator(strict_mode=strict_mode)
        self.intelligent_classifier = IntelligentClassifier()
        # NEW: common-sense gate
        self.common_gate = CommonSenseToolGate(enable_embeddings=True)
        
        mode_str = "STRICT" if strict_mode else "DISCOVERY"
        print(f"  🎯 Scorer initialized in {mode_str} mode")
    
    def score_tool_for_workflow(self, item: Dict, workflow: Dict) -> Dict:
        """Complete scoring with all enhancements and detailed logging"""
        
        print(f"\n{'='*60}")
        print(f"🎯 SCORING: {item.get('name', 'Unknown Tool')}")
        print(f"{'='*60}")
        
        # Step 1: Intelligent classification
        print(f"\n📋 Step 1: Content Classification")
        ai_analysis = self.intelligent_classifier.classify_and_extract(item)
        item.update(ai_analysis)

        # Step 1.5: Common-sense tool gate (NEW) — short-circuit if not a tool
        print(f"\n📋 Step 1.5: Common-Sense Tool Gate")
        gate = self.common_gate.decide(item)
        item['commonsense_is_tool'] = gate.is_tool
        item['commonsense_confidence'] = gate.confidence
        item['commonsense_reasons'] = gate.reasons
        if not gate.is_tool:
            reason = "Common-sense gate: not an adoptable tool (" + "; ".join(gate.reasons[:3]) + ")"
            print(f"  ❌ {reason}")
            return {
                'eligible': False,
                'reason': reason,
                'signal_strength': ai_analysis.get('content_type', 'UNKNOWN'),
                'category': ai_analysis.get('category', 'General'),
                'final_score': 0,
                'commonsense_is_tool': gate.is_tool,
                'commonsense_confidence': gate.confidence,
                'commonsense_reasons': gate.reasons
            }
        print("  ✓ Common-sense gate passed: likely a tool")

        # Store feed repetition for later use
        feed_repetition = ai_analysis.get('feed_repetition', 'low')
        
        # Step 2: Detect and remove fluff
        print(f"\n📋 Step 2: Fluff Detection")
        fluff_analysis = self.fluff_detector.analyze_text(
            f"{item.get('title', '')} {item.get('description', '')}"
        )
        item['fluff_ratio'] = fluff_analysis['fluff_ratio']
        item['unverifiable_claims'] = fluff_analysis['unverifiable_claims']
        
        # Step 3: Signal classification
        print(f"\n📋 Step 3: Signal Strength Analysis")
        signal_assessment = self.signal_classifier.classify_signal(item)
        
        # Step 4: Check if it's actually a tool (second check – your original logic)
        print(f"\n📋 Step 4: Tool Validation")
        is_tool = self.is_implementable(item, ai_analysis, self.strict_mode)
        
        if not is_tool:
            reason = f"This is {ai_analysis.get('content_type', 'content')}, not an implementable tool"
            print(f"  ❌ Not a tool: {reason}")
            return {
                'eligible': False,
                'reason': reason,
                'signal_strength': signal_assessment['signal_level'],
                'category': ai_analysis.get('category', 'General'),
                'final_score': 0,
                'commonsense_is_tool': gate.is_tool,
                'commonsense_confidence': gate.confidence,
                'commonsense_reasons': gate.reasons
            }
        print(f"  ✓ Confirmed as implementable tool")
        
        # Step 5: Check workflow constraints
        print(f"\n📋 Step 5: Constraint Check")
        if not self.passes_constraints(item, workflow):
            print(f"  ❌ Failed workflow constraints")
            return {
                'eligible': False,
                'reason': 'Failed workflow constraints',
                'final_score': 0,
                'commonsense_is_tool': gate.is_tool,
                'commonsense_confidence': gate.confidence,
                'commonsense_reasons': gate.reasons
            }
        print(f"  ✓ Passes all constraints")
        
        # Step 6: Calculate component scores
        print(f"\n📋 Step 6: Component Scoring")
        keyword_score = self.calculate_keyword_match(item, workflow)
        print(f"  📌 Keyword Match: {keyword_score:.1f}%")
        
        implementation_score = self.calculate_implementation_fit(item, workflow)
        print(f"  🔧 Implementation Fit: {implementation_score:.1f}%")
        
        evidence_score = self.calculate_evidence_score(item)
        print(f"  📊 Evidence Quality: {evidence_score:.1f}%")
        
        # Step 7: ROI Analysis
        print(f"\n📋 Step 7: ROI Analysis")
        current_state = workflow.get('current_state', {})
        roi_analysis = self.roi_calculator.analyze_tool_worth(item, workflow, current_state)
        
        # Step 8: Calculate final weighted score with feed bonus
        print(f"\n📋 Step 8: Final Scoring")
        final_score, feed_bonus = self.calculate_final_score(
            keyword_score,
            implementation_score,
            evidence_score,
            signal_assessment,
            roi_analysis,
            fluff_analysis,
            item
        )
        
        # Scoring breakdown (for console debugging)
        signal_component = self.get_signal_score(signal_assessment) * (0.30 if self.strict_mode else 0.35)
        keyword_component = keyword_score * (0.25 if self.strict_mode else 0.20)
        implementation_component = implementation_score * (0.20 if self.strict_mode else 0.15)
        evidence_component = evidence_score * (0.15 if self.strict_mode else 0.10)
        roi_component = self.get_roi_score(roi_analysis) * (0.10 if self.strict_mode else 0.20)
        fluff_penalty = fluff_analysis['credibility_penalty'] * (100 if self.strict_mode else 30)
        
        print(f"\n  🎲 Score Breakdown:")
        print(f"    Signal ({30 if self.strict_mode else 35}%):        {signal_component:.1f}")
        print(f"    Keywords ({25 if self.strict_mode else 20}%):      {keyword_component:.1f}")
        print(f"    Implementation ({20 if self.strict_mode else 15}%): {implementation_component:.1f}")
        print(f"    Evidence ({15 if self.strict_mode else 10}%):      {evidence_component:.1f}")
        print(f"    ROI ({10 if self.strict_mode else 20}%):          {roi_component:.1f}")
        print(f"    Fluff Penalty:      -{fluff_penalty:.1f}")
        if feed_bonus > 0:
            print(f"    Feed Bonus:         +{feed_bonus:.1f}")
        print(f"    ─────────────────────────")
        print(f"    FINAL SCORE:        {final_score:.1f}/100")
        
        # Step 9: Generate recommendation
        recommendation = self.get_recommendation(final_score, signal_assessment, roi_analysis)
        print(f"\n📋 Step 9: Recommendation")
        print(f"  💡 {recommendation}")
        
        # Step 10: Generate comprehensive result
        result = {
            'eligible': True,
            'final_score': final_score,
            'signal_strength': signal_assessment['signal_level'],
            'signal_confidence': signal_assessment['confidence'],
            'actionability': signal_assessment['actionability'],
            'keyword_match_score': keyword_score,
            'implementation_score': implementation_score,
            'evidence_score': evidence_score,
            'roi_analysis': roi_analysis,
            'fluff_ratio': fluff_analysis['fluff_ratio'],
            'strategic_tag': self.assign_strategic_tag(final_score, roi_analysis),
            'recommendation': recommendation,
            'immediate_action': self.get_immediate_action(item, signal_assessment),
            'explanation': self.generate_explanation(
                item, workflow, keyword_score, implementation_score, 
                signal_assessment, roi_analysis
            ),
            'category': ai_analysis.get('category', 'General'),
            'primary_capabilities': ai_analysis.get('primary_capabilities', []),
            'feed_repetition': feed_repetition,
            'feed_bonus': feed_bonus,
            'tool_confirmed': ai_analysis.get('tool_confirmed', False),
            # NEW: expose gate signal in result
            'commonsense_is_tool': gate.is_tool,
            'commonsense_confidence': gate.confidence,
            'commonsense_reasons': gate.reasons,
            'scoring_breakdown': {
                'keywords_matched': self.get_matched_keywords(item, workflow),
                'keywords_missing': self.get_missing_keywords(item, workflow),
                'signal_weight': f"{30 if self.strict_mode else 35}%",
                'keyword_weight': f"{25 if self.strict_mode else 20}%",
                'implementation_weight': f"{20 if self.strict_mode else 15}%",
                'evidence_weight': f"{15 if self.strict_mode else 10}%",
                'roi_weight': f"{10 if self.strict_mode else 20}%"
            }
        }
        
        print(f"\n{'='*60}")
        print(f"✅ SCORING COMPLETE: {item.get('name', 'Unknown')}")
        print(f"   Category: {result['category']}")
        print(f"   Signal: {result['signal_strength']}")
        print(f"   Score: {result['final_score']:.1f}/100")
        print(f"   Action: {result['recommendation']}")
        print(f"{'='*60}\n")
        
        return result
    
    def get_signal_score(self, signal_assessment: Dict) -> float:
        """Convert signal level to numeric score"""
        signal_scores = {
            'GAME_CHANGING': 100,
            'HIGH': 80,
            'PROMISING': 65,  # NEW: Between MODERATE and HIGH
            'MODERATE': 50,
            'NOISE': 10
        }
        return signal_scores.get(signal_assessment['signal_level'], 30)
    
    def get_roi_score(self, roi_analysis: Dict) -> float:
        """Convert ROI to numeric score"""
        payback = roi_analysis['payback_months']
        if payback < 3:
            return 100
        elif payback < 6:
            return 80
        elif payback < 12:
            return 60
        elif payback < 24:
            return 40
        else:
            return 20
    
    def is_implementable(self, item: Dict, ai_analysis: Dict, strict_mode=True) -> bool:
        """Determine if this is an actual implementable tool or announcement about one"""
        content_type = ai_analysis.get('content_type')
        
        # Tool confirmed by AI - trust it
        if ai_analysis.get('tool_confirmed', False):
            print(f"    ℹ️  Tool confirmed by AI analysis")
            return True
        
        # In discovery mode, be much more permissive
        if not strict_mode:
            # Almost everything except pure opinion pieces can be considered
            if content_type == 'OPINION' and ai_analysis.get('technical_depth', 0) < 3:
                return False
            # Accept everything else as potentially implementable
            return True
        
        # Direct tool classification
        if content_type == 'TOOL':
            return True
        
        # Announcements are usually about tools - accept them!
        if content_type == 'ANNOUNCEMENT':
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            tool_indicators = ['tool', 'api', 'library', 'framework', 'sdk', 'platform', 
                             'engine', 'system', 'solution', 'software', 'app', 'plugin',
                             'release', 'launch', 'available', 'open source', 'version',
                             'update', 'feature', 'capability', 'model', 'service']
            if any(indicator in text for indicator in tool_indicators):
                print(f"    ℹ️  Announcement about tool/service detected - will score")
                return True
        
        # Case studies are often about tools
        if content_type == 'CASE_STUDY':
            return True
        
        # Research papers about tools/systems
        if content_type == 'RESEARCH':
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if any(word in text for word in ['implementation', 'system', 'framework', 'tool', 'method']):
                print(f"    ℹ️  Research about implementable system detected")
                return True
        
        # High technical depth makes it likely implementable
        if ai_analysis.get('technical_depth', 0) >= 5:
            print(f"    ℹ️  High technical depth ({ai_analysis.get('technical_depth')}) - treating as implementable")
            return True
        
        # High practicality score
        if ai_analysis.get('practicality_score', 0) >= 5:
            print(f"    ℹ️  High practicality ({ai_analysis.get('practicality_score')}) - treating as implementable")
            return True
        
        # Unknown content - check for ANY tool characteristics
        if content_type == 'UNKNOWN':
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if any(word in text for word in ['github', 'install', 'download', 'api', 'library', 'use', 'build']):
                print(f"    ℹ️  Unknown content with tool indicators - will attempt scoring")
                return True
        
        if content_type == 'OPINION':
            return False
        
        print(f"    ⚠️  Unclear if implementable, but attempting to score anyway")
        return True
    
    def passes_constraints(self, item: Dict, workflow: Dict) -> bool:
        """Check if tool passes workflow constraints"""
        constraints = workflow.get('constraints', {})
        
        # Cost constraint
        max_cost = constraints.get('max_monthly_cost')
        if max_cost:
            tool_cost = self.roi_calculator.extract_monthly_cost(item)
            if tool_cost > max_cost:
                print(f"    Cost ${tool_cost} exceeds limit ${max_cost}")
                return False
        
        # Technical constraints
        required_tech = constraints.get('required_technology', [])
        if required_tech:
            tool_tech = item.get('keywords', []) + item.get('implementation_keywords', [])
            if not any(tech in tool_tech for tech in required_tech):
                print(f"    Missing required tech: {required_tech}")
                return False
        
        return True
    
    def calculate_keyword_match(self, tool: Dict, workflow: Dict) -> float:
        """Calculate keyword overlap with semantic understanding"""
        required = set(workflow.get('required_keywords', []))
        if not required:
            return 50
        
        # Priority keywords get more weight
        priority_keywords = {
            'ai': 2.0, 'ml': 2.0, 'api': 1.8, 'automation': 1.8,
            'framework': 1.5, 'library': 1.5, 'sdk': 1.5,
            'tool': 1.0, 'software': 1.0, 'platform': 1.2
        }
        
        # Check multiple sources for keywords
        tool_keywords = set()
        tool_keywords.update(tool.get('keywords', []))
        tool_keywords.update(tool.get('implementation_keywords', []))
        
        # Check in text for both exact and semantic matches
        text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
        
        # Semantic equivalents for common terms
        semantic_matches = {
            'ai': ['artificial intelligence', 'machine learning', 'ml', 'deep learning', 'neural'],
            'automation': ['automate', 'automated', 'automatic', 'bot'],
            'api': ['endpoint', 'rest', 'graphql', 'interface'],
            'creative': ['design', 'art', 'visual', 'graphics', 'assets'],
            'tool': ['utility', 'application', 'software', 'solution']
        }
        
        weighted_score = 0
        max_possible_score = 0
        
        for keyword in required:
            keyword_lower = keyword.lower()
            weight = priority_keywords.get(keyword_lower, 1.0)
            max_possible_score += weight
            
            # Direct match
            if keyword_lower in text or keyword in tool_keywords:
                weighted_score += weight
                tool_keywords.add(keyword)
            # Semantic match
            elif keyword_lower in semantic_matches:
                for equivalent in semantic_matches[keyword_lower]:
                    if equivalent in text:
                        weighted_score += weight * 0.8  # Slightly less weight for semantic match
                        tool_keywords.add(keyword)
                        break
        
        # In discovery mode, be more generous with partial matches
        if not self.strict_mode and weighted_score > 0:
            weighted_score = max(weighted_score, max_possible_score * 0.4)
        
        return (weighted_score / max_possible_score * 100) if max_possible_score > 0 else 50
    
    def calculate_implementation_fit(self, tool: Dict, workflow: Dict) -> float:
        """Calculate implementation feasibility"""
        score = 30  # Base
        
        # Effort scoring
        effort_scores = {
            'hours': 40,
            'days': 30,
            'weeks': 20,
            'months': 10,
            'unknown': 5
        }
        effort = tool.get('implementation_effort', 'unknown')
        score += effort_scores.get(effort, 5)
        
        # Maturity scoring
        maturity_scores = {
            'production': 30,
            'beta': 20,
            'experimental': 10,
            'unknown': 5
        }
        maturity = tool.get('maturity_level', 'unknown')
        score += maturity_scores.get(maturity, 5)
        
        return min(score, 100)
    
    def calculate_evidence_score(self, item: Dict) -> float:
        """Score based on evidence quality"""
        score = 0
        
        # GitHub presence
        if item.get('github_url'):
            stars = item.get('github_stars', 0)
            if stars > 1000:
                score += 30
            elif stars > 100:
                score += 20
            elif stars > 10:
                score += 10
        
        # Production usage
        if item.get('production_deployments', 0) > 0:
            score += 25
        
        # Implementation examples
        if item.get('implementation_evidence', []):
            score += 20
        
        # Technical documentation
        if item.get('technical_documentation'):
            score += 15
        
        # Community adoption
        if item.get('reddit_posts', 0) > 0:
            score += 10
        
        return min(score, 100)
    
    def calculate_final_score(self, keyword_score: float, implementation_score: float,
                            evidence_score: float, signal_assessment: Dict,
                            roi_analysis: Dict, fluff_analysis: Dict, item: Dict = None) -> Tuple[float, float]:
        """Calculate weighted final score with feed repetition bonus - returns (score, bonus)"""
        
        # Signal strength score
        signal_score = self.get_signal_score(signal_assessment)
        
        # ROI score
        roi_score = self.get_roi_score(roi_analysis)
        
        # Fluff penalty (reduced in discovery mode)
        if self.strict_mode:
            fluff_penalty = fluff_analysis['credibility_penalty'] * 100
        else:
            fluff_penalty = fluff_analysis['credibility_penalty'] * 30
        
        # Adjusted weights for discovery mode
        if self.strict_mode:
            final_score = (
                signal_score * 0.30 +
                keyword_score * 0.25 +
                implementation_score * 0.20 +
                evidence_score * 0.15 +
                roi_score * 0.10
            ) - fluff_penalty
        else:
            final_score = (
                signal_score * 0.35 +
                keyword_score * 0.20 +
                implementation_score * 0.15 +
                evidence_score * 0.10 +
                roi_score * 0.20
            ) - fluff_penalty
            
            if roi_analysis.get('is_creative', False):
                final_score += 10  # Creative bonus
            
            if signal_assessment['signal_level'] == 'PROMISING':
                final_score = max(final_score, 45)  # Minimum score for promising tools
        
        # Feed repetition bonus (tools appearing frequently in feeds)
        feed_bonus = 0
        if item:
            feed_repetition = item.get('feed_repetition', 'low')
            if feed_repetition == 'medium':
                feed_bonus = 5
                print(f"    📰 Feed repetition bonus: +5 (medium frequency)")
            elif feed_repetition == 'high':
                feed_bonus = 10
                print(f"    📰 Feed repetition bonus: +10 (high frequency)")
        
        final_score += feed_bonus
        
        return max(0, min(100, final_score)), feed_bonus
    
    def assign_strategic_tag(self, score: float, roi_analysis: Dict) -> str:
        """Assign strategic category"""
        if score >= 85 and roi_analysis['payback_months'] < 6:
            return "Quick Win"
        elif score >= 70:
            return "Strategic Bet"
        elif score >= 50:
            return "Experiment"
        else:
            return "Monitor Only"
    
    def get_recommendation(self, score: float, signal: Dict, roi: Dict) -> str:
        """Get nuanced recommendation based on mode and signal"""
        signal_level = signal['signal_level']
        
        if not self.strict_mode:
            if signal_level == 'GAME_CHANGING' and score > 60:
                return "IMMEDIATE_PILOT"
            elif signal_level == 'HIGH' and score > 50:
                return "PILOT"
            elif signal_level == 'PROMISING':
                if score >= 50:
                    return "EXPERIMENT"
                else:
                    return "TRACK"
            elif score >= 60:
                return "PILOT"
            elif score >= 40:
                return "EXPERIMENT"
            elif score >= 30:
                return "TRACK"
            else:
                return "MONITOR"
        else:
            if signal_level == 'GAME_CHANGING' and score > 70:
                return "IMMEDIATE_PILOT"
            elif score >= 80 and roi['payback_months'] < 3:
                return "ADOPT_NOW"
            elif score >= 60 and roi['payback_months'] < 12:
                return "PILOT"
            elif score >= 40:
                return "ASSESS"
            else:
                return "SKIP"
    
    def get_immediate_action(self, item: Dict, signal: Dict) -> str:
        """Determine immediate next step"""
        if signal['signal_level'] == 'GAME_CHANGING':
            return "Schedule team review this week"
        elif signal['signal_level'] == 'HIGH':
            return f"Test {item.get('name', 'tool')} with small project"
        elif signal['signal_level'] == 'PROMISING':
            return f"Add to innovation backlog for Q2"
        elif item.get('github_url'):
            return f"Clone repo and review documentation"
        else:
            return "Add to quarterly review list"
    
    def generate_explanation(self, item: Dict, workflow: Dict, keyword_score: float,
                           implementation_score: float, signal: Dict, roi: Dict) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Signal explanation
        explanations.append(f"Signal: {signal['signal_level']} - {signal['why_this_level']}")
        
        # Keyword match
        if keyword_score >= 80:
            explanations.append(f"Strong keyword match ({keyword_score:.0f}%)")
        elif keyword_score >= 50:
            explanations.append(f"Moderate keyword match ({keyword_score:.0f}%)")
        else:
            explanations.append(f"Weak keyword match ({keyword_score:.0f}%)")
        
        # ROI
        if roi['payback_months'] < 6:
            explanations.append(f"Fast ROI: {roi['payback_months']:.1f} months payback")
        else:
            explanations.append(f"Slow ROI: {roi['payback_months']:.1f} months payback")
        
        # Risk
        risk_level = roi['risk_factors']['failure_probability']
        if risk_level < 0.2:
            explanations.append("Low implementation risk")
        elif risk_level < 0.4:
            explanations.append("Moderate implementation risk")
        else:
            explanations.append("High implementation risk")
        
        return " | ".join(explanations)
    
    def get_matched_keywords(self, tool: Dict, workflow: Dict) -> List[str]:
        """Get list of matched keywords"""
        required = set(workflow.get('required_keywords', []))
        tool_keywords = set()
        tool_keywords.update(tool.get('keywords', []))
        tool_keywords.update(tool.get('implementation_keywords', []))
        
        return list(required.intersection(tool_keywords))
    
    def get_missing_keywords(self, tool: Dict, workflow: Dict) -> List[str]:
        """Get list of missing keywords"""
        required = set(workflow.get('required_keywords', []))
        tool_keywords = set()
        tool_keywords.update(tool.get('keywords', []))
        tool_keywords.update(tool.get('implementation_keywords', []))
        
        return list(required - tool_keywords)


# Backward compatibility wrapper
class WorkflowScorer(EnhancedWorkflowScorer):
    """Maintains backward compatibility with existing code"""
    pass


if __name__ == "__main__":
    # Example usage
    scorer = EnhancedWorkflowScorer()
    
    # Example tool
    tool = {
        'name': 'Claude API',
        'description': 'Anthropic AI assistant API for production applications',
        'keywords': ['ai', 'api', 'assistant'],
        'implementation_effort': 'days',
        'maturity_level': 'production',
        'cost_benefit': {'monthly_cost': '100'},
        'github_stars': 5000,
        'source': 'GitHub',
        'source_url': 'https://github.com/anthropics/anthropic-sdk'
    }
    
    # Example workflow
    workflow = {
        'name': 'Customer Support Automation',
        'required_keywords': ['ai', 'api', 'support', 'automation'],
        'constraints': {'max_monthly_cost': 500},
        'current_state': {'current_monthly_cost': 300}
    }
    
    result = scorer.score_tool_for_workflow(tool, workflow)
    print(json.dumps(result, indent=2))
