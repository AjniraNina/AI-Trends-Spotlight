import json
from typing import Dict, List
from openai import OpenAI
from scorer import EnhancedWorkflowScorer
from datetime import datetime

class IntelligentReportGenerator:
    def __init__(self):
        self.client = OpenAI()
        self.scorer = EnhancedWorkflowScorer()
        
    def generate_tool_report(self, tool: Dict) -> Dict:
        """Generate comprehensive analysis for a single tool"""
        
        # First, run the tool through the enhanced scorer
        scoring_result = self.scorer.score_tool_for_workflow(
            tool, 
            {'required_keywords': [], 'constraints': {}}  # Generic workflow
        )
        
        # Extract meaningful insights using AI
        enhanced_analysis = self.analyze_tool_deeply(tool, scoring_result)
        
        # Calculate real costs
        cost_analysis = self.calculate_true_costs(tool)
        
        # Find real-world examples
        use_cases = self.find_real_world_examples(tool)
        
        return {
            **tool,
            **scoring_result,
            'enhanced_analysis': enhanced_analysis,
            'cost_breakdown': cost_analysis,
            'real_world_examples': use_cases,
            'actionable_intelligence': self.generate_actionable_intelligence(
                tool, scoring_result, enhanced_analysis
            )
        }
    
    def analyze_tool_deeply(self, tool: Dict, scoring: Dict) -> Dict:
        """Use AI to extract deep insights about the tool"""
        
        prompt = f"""
        Analyze this tool as a pragmatic product manager who needs real results:
        
        Tool: {tool.get('name')}
        Description: {tool.get('description', '')}
        Signal Level: {scoring.get('signal_strength', 'unknown')}
        Evidence Score: {scoring.get('evidence_score', 0)}
        
        Provide brutally honest analysis:
        {{
            "what_it_actually_does": "specific capability in plain language",
            "who_really_uses_this": "actual companies/developers using it",
            "real_problem_solved": "the actual pain point this addresses",
            "hidden_gotchas": ["things they don't tell you"],
            "competitive_reality": "how it compares to existing solutions",
            "implementation_reality": "what it really takes to implement",
            "roi_timeline": "when you'll see actual value",
            "team_reaction": "how your team will actually respond",
            "better_alternatives": ["other tools that might work better"],
            "verdict": "GAME_CHANGER|USEFUL|NICE_TO_HAVE|SKIP"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a skeptical tech evaluator. Cut through the hype."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {
                'what_it_actually_does': tool.get('description', 'Unknown'),
                'verdict': 'NEEDS_RESEARCH'
            }
    
    def calculate_true_costs(self, tool: Dict) -> Dict:
        """Calculate the real cost of implementation vs alternatives"""
        
        prompt = f"""
        Calculate the TRUE cost of implementing {tool.get('name')}:
        
        Current description: {tool.get('description', '')}
        Stated effort: {tool.get('implementation_effort', 'unknown')}
        
        Compare to doing this WITHOUT the tool. Return realistic numbers:
        {{
            "without_tool": {{
                "developer_hours": number,
                "monthly_maintenance_hours": number,
                "total_monthly_cost": dollar_amount,
                "limitations": ["what you can't do"]
            }},
            "with_tool": {{
                "setup_hours": number,
                "learning_curve_hours": number,
                "monthly_cost": dollar_amount,
                "hidden_costs": ["training", "migration", "vendor lock-in"]
            }},
            "breakeven_months": number,
            "savings_per_month_after_breakeven": dollar_amount,
            "confidence_in_estimate": "high|medium|low"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return {
                'without_tool': {'total_monthly_cost': 5000},
                'with_tool': {'monthly_cost': 500},
                'breakeven_months': 3,
                'confidence_in_estimate': 'low'
            }
    
    def find_real_world_examples(self, tool: Dict) -> List[Dict]:
        """Find actual use cases and implementations"""
        
        prompt = f"""
        For the tool "{tool.get('name')}" which {tool.get('description', '')},
        provide SPECIFIC real-world examples:
        
        Return JSON array of 3 examples:
        [
            {{
                "use_case": "specific problem being solved",
                "implementation": "how it's actually used",
                "code_snippet": "if applicable, simple example",
                "time_saved": "hours per week",
                "team_size_appropriate": "solo|small|medium|enterprise"
            }}
        ]
        
        If this is similar to existing tools, explain how those are used instead.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=400
            )
            
            return json.loads(response.choices[0].message.content)
        except:
            return [{
                'use_case': 'General automation',
                'implementation': 'Standard integration',
                'time_saved': 'Unknown'
            }]
    
    def generate_actionable_intelligence(self, tool: Dict, scoring: Dict, analysis: Dict) -> Dict:
        """Generate specific, actionable next steps"""
        
        # Based on signal strength and analysis
        signal = scoring.get('signal_strength', 'MODERATE')
        verdict = analysis.get('verdict', 'UNKNOWN')
        
        if signal == 'GAME_CHANGING' and verdict in ['GAME_CHANGER', 'USEFUL']:
            return {
                'priority': 'IMMEDIATE',
                'next_steps': [
                    f"Schedule 30-min team review TODAY",
                    f"Clone repo: git clone {tool.get('source_url', '')}",
                    f"Run proof-of-concept by end of week",
                    f"Get budget approval if POC succeeds"
                ],
                'success_metrics': [
                    "POC completes in <4 hours",
                    "Solves at least one current pain point",
                    "Team enthusiasm score >7/10"
                ],
                'decision_deadline': "This Friday"
            }
        
        elif signal == 'HIGH' and verdict == 'USEFUL':
            return {
                'priority': 'THIS_QUARTER',
                'next_steps': [
                    f"Add to Q2 evaluation backlog",
                    f"Assign junior dev to research",
                    f"Compare with top 3 alternatives"
                ],
                'success_metrics': [
                    "Clear advantage over current solution",
                    "Implementation under 1 week",
                    "Positive community feedback"
                ],
                'decision_deadline': "End of quarter"
            }
        
        elif signal == 'NOISE' or verdict == 'SKIP':
            return {
                'priority': 'IGNORE',
                'next_steps': [
                    "No action needed",
                    "Mark as evaluated"
                ],
                'reason': analysis.get('hidden_gotchas', ['Too much hype, not enough substance'])[0]
            }
        
        else:
            return {
                'priority': 'MONITOR',
                'next_steps': [
                    f"Add to watch list",
                    f"Check GitHub stars in 3 months",
                    f"Wait for v2.0 or more adoption"
                ],
                'success_metrics': [
                    "1000+ GitHub stars",
                    "3+ production case studies",
                    "Major version release"
                ],
                'decision_deadline': "Next quarter review"
            }


class EnhancedReportFormatter:
    """Format the analyzed data for display"""
    
    def format_tool_card(self, tool_data: Dict) -> Dict:
        """Format tool data for the report template"""
        
        analysis = tool_data.get('enhanced_analysis', {})
        costs = tool_data.get('cost_breakdown', {})
        intelligence = tool_data.get('actionable_intelligence', {})
        examples = tool_data.get('real_world_examples', [])
        
        return {
            'name': tool_data.get('name', 'Unknown Tool'),
            'source': tool_data.get('source', 'Unknown'),
            
            # Signal and confidence
            'signal_strength': tool_data.get('signal_strength', 'MODERATE'),
            'confidence': tool_data.get('signal_confidence', 0.5),
            'verdict': analysis.get('verdict', 'UNKNOWN'),
            
            # Why it matters - with REAL insight
            'why_important': analysis.get('real_problem_solved', 
                                        'Potentially useful but needs investigation'),
            
            # Immediate action - SPECIFIC steps
            'immediate_action': ' → '.join(intelligence.get('next_steps', ['Evaluate'])[:2]),
            
            # Implementation reality
            'implementation_effort': analysis.get('implementation_reality', 
                                                tool_data.get('implementation_effort', 'unknown')),
            
            # Cost analysis
            'time_saved_per_month': self._calculate_time_saved(costs),
            'cost_benefit': {
                'setup_cost': self._format_setup_cost(costs),
                'monthly_cost': self._format_monthly_cost(costs),
                'breakeven': f"{costs.get('breakeven_months', '?')} months"
            },
            
            # Reality checks
            'reality_check': {
                'works_with_existing_stack': self._check_compatibility(tool_data),
                'requires_gpu': 'gpu' in str(tool_data).lower(),
                'requires_data_science_team': analysis.get('team_reaction', '') == 'needs expertise',
                'requires_infrastructure_change': 'migration' in str(costs.get('hidden_costs', [])).lower()
            },
            
            # Use cases
            'use_cases': examples[:2] if examples else [],
            
            # Competitive reality
            'alternatives': analysis.get('better_alternatives', []),
            
            # Hidden gotchas
            'warnings': analysis.get('hidden_gotchas', []),
            
            # Actionability score (0-1)
            'actionability_score': self._calculate_actionability(tool_data),
            
            # ROI metrics
            'roi_timeline': analysis.get('roi_timeline', 'Unknown'),
            
            # Decision support
            'recommendation': tool_data.get('recommendation', 'ASSESS'),
            'priority': intelligence.get('priority', 'MONITOR'),
            
            # Source URL for deep dive
            'source_url': tool_data.get('source_url', '#'),
            
            # Fluff detection
            'fluff_ratio': tool_data.get('fluff_ratio', 0)
        }
    
    def _calculate_time_saved(self, costs: Dict) -> int:
        """Calculate realistic time saved per month"""
        without = costs.get('without_tool', {})
        with_tool = costs.get('with_tool', {})
        
        dev_hours = without.get('developer_hours', 0)
        maintenance_hours = without.get('monthly_maintenance_hours', 0)
        
        # Assume tool reduces time by 70%
        return int((dev_hours + maintenance_hours) * 0.7)
    
    def _format_setup_cost(self, costs: Dict) -> str:
        """Format setup cost in human terms"""
        setup_hours = costs.get('with_tool', {}).get('setup_hours', 0)
        
        if setup_hours < 8:
            return "LOW"
        elif setup_hours < 40:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _format_monthly_cost(self, costs: Dict) -> str:
        """Format monthly cost"""
        monthly = costs.get('with_tool', {}).get('monthly_cost', 0)
        
        if monthly == 0:
            return "Free"
        elif monthly < 100:
            return f"${monthly}"
        elif monthly < 1000:
            return f"${monthly}/mo"
        else:
            return f"${monthly/1000:.1f}k/mo"
    
    def _check_compatibility(self, tool_data: Dict) -> bool:
        """Check if it works with common stacks"""
        keywords = tool_data.get('keywords', [])
        common_stack = ['python', 'javascript', 'api', 'docker', 'cloud']
        return any(tech in keywords for tech in common_stack)
    
    def _calculate_actionability(self, tool_data: Dict) -> float:
        """Calculate how actionable this tool is"""
        score = 0.0
        
        # Clear next steps
        if tool_data.get('actionable_intelligence', {}).get('next_steps'):
            score += 0.3
        
        # Low implementation effort
        if tool_data.get('implementation_effort') in ['hours', 'days']:
            score += 0.3
        
        # Positive ROI
        if tool_data.get('cost_breakdown', {}).get('breakeven_months', 999) < 6:
            score += 0.2
        
        # High confidence
        if tool_data.get('signal_confidence', 0) > 0.7:
            score += 0.2
        
        return min(score, 1.0)