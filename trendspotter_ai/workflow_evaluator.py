import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


class AIUsefulnessOverlay:
    """
    Contextual 'is this actually useful?' analysis.
    Produces: usefulness_score (0-100), decision, why, best_fit_tasks,
    integration_ideas, quick_prompts, blockers, alternatives.
    """

    def __init__(self, client: Optional["OpenAI"] = None):
        self.client = client

    def analyze(self, tool: Dict, workflow: Dict) -> Dict:
        if not self.client:
            return self._heuristic(tool, workflow)

        name = tool.get("name", "Unknown")
        caps = tool.get("primary_capabilities", []) or tool.get("keywords", [])
        category = tool.get("category", "General")
        desc = tool.get("description", "")
        wf_name = workflow.get("name", "Workflow")
        wf_kws = workflow.get("required_keywords", [])
        current = workflow.get("current_state", {})

        prompt = f"""
        You are an expert solution architect. Given a TOOL and a WORKFLOW,
        judge if the tool is contextually useful (not just keyword match),
        be practical and creative. Return ONLY valid JSON.

        TOOL:
        - name: {name}
        - category: {category}
        - description: {desc}
        - capabilities: {', '.join(caps[:8])}

        WORKFLOW:
        - name: {wf_name}
        - required_keywords: {', '.join(wf_kws)}
        - current_state: {json.dumps(current) if current else "unknown"}

        Return exactly this JSON shape:
        {{
          "usefulness_score": 0,
          "decision": "USEFUL" | "MAYBE" | "NOT_USEFUL",
          "why": "one concise paragraph",
          "best_fit_tasks": ["task1","task2","task3"],
          "integration_ideas": ["idea1","idea2","idea3"],
          "quick_prompts": ["prompt the team could try today", "…"],
          "blockers": ["risk or missing piece"],
          "alternatives": ["adjacent approach or tool class"]
        }}
        """

        try:
            resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Return only valid JSON. Be concrete, pragmatic, and concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            return self._fill_defaults(data)
        except Exception as e:
            print(f"[overlay] AI error: {e}")
            return self._heuristic(tool, workflow)

    def _fill_defaults(self, d: Dict) -> Dict:
        d.setdefault("usefulness_score", 0)
        d.setdefault("decision", "MAYBE")
        d.setdefault("why", "Context analysis unavailable.")
        d.setdefault("best_fit_tasks", [])
        d.setdefault("integration_ideas", [])
        d.setdefault("quick_prompts", [])
        d.setdefault("blockers", [])
        d.setdefault("alternatives", [])
        # Clamp score
        try:
            d["usefulness_score"] = int(max(0, min(100, float(d["usefulness_score"]))))
        except Exception:
            d["usefulness_score"] = 0
        # Sanitize decision
        if d["decision"] not in ["USEFUL", "MAYBE", "NOT_USEFUL"]:
            d["decision"] = "MAYBE"
        return d

    def _heuristic(self, tool: Dict, workflow: Dict) -> Dict:
        text = f"{tool.get('name','')} {tool.get('description','')}".lower()
        caps = ' '.join(tool.get('primary_capabilities', [])).lower()
        wf_kws = [k.lower() for k in workflow.get('required_keywords', [])]
        hits = sum(k in text or k in caps for k in wf_kws)

        score = 25 + min(50, hits * 15)
        decision = "NOT_USEFUL"
        if score >= 65:
            decision = "USEFUL"
        elif score >= 40:
            decision = "MAYBE"

        ideas = []
        if "api" in text or "api" in caps:
            ideas.append("Wrap as a microservice callable from workflow steps")
        if "chat" in text or "assistant" in caps:
            ideas.append("Expose as a chat/action in your support or marketing agent desktop")
        if "classification" in text or "nlp" in caps:
            ideas.append("Use for intent/routing before reply generation")

        return {
            "usefulness_score": score,
            "decision": decision,
            "why": "Heuristic alignment based on overlapping terms and generic capability fit.",
            "best_fit_tasks": ["Prototype on one high-volume intent", "A/B test against current baseline"],
            "integration_ideas": ideas[:3] or ["Manual trial by one agent for a week"],
            "quick_prompts": ["Summarize this ticket and propose a reply", "Classify this ticket into intents"],
            "blockers": ["Unclear production readiness", "Integration effort unknown"],
            "alternatives": ["Prebuilt Zendesk/Intercom apps", "Native HubSpot AI"]
        }


class WorkflowEvaluator:
    """Evaluate if tool should be integrated or replace workflow"""

    def __init__(self):
        self.client = None
        if os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI()
        # NEW: usefulness overlay
        self.overlay = AIUsefulnessOverlay(self.client)

    def evaluate(self, tool: Dict, workflow: Dict) -> Dict:
        """Should we integrate, replace, or skip this tool for the workflow?"""

        # Skip if it's noise or low signal
        if tool.get('signal_strength') == 'NOISE':
            base = {
                'recommendation': 'IGNORE',
                'integration_type': 'SKIP',
                'reason': 'Marketing fluff without substance',
                'confidence': tool.get('signal_confidence', 0.2)
            }
            # Even on skip, attach overlay so UI can explain *why not*
            try:
                base['overlay'] = self.overlay.analyze(tool, workflow)
            except Exception as e:
                print(f"[overlay] attach error: {e}")
            return base

        if tool.get('final_score', 0) < 30:
            base = {
                'recommendation': 'IGNORE',
                'integration_type': 'SKIP',
                'reason': 'Tool does not meet minimum quality threshold',
                'confidence': tool.get('signal_confidence', 0.3)
            }
            try:
                base['overlay'] = self.overlay.analyze(tool, workflow)
            except Exception as e:
                print(f"[overlay] attach error: {e}")
            return base

        # Use AI for detailed evaluation if available
        core_eval = self._ai_evaluation(tool, workflow) if self.client else self._heuristic_evaluation(tool, workflow)

        # Attach AI Usefulness Overlay (creative, context-aware pass)
        try:
            overlay = self.overlay.analyze(tool, workflow)
            core_eval['overlay'] = overlay
            # Optional nudge: if overlay says NOT_USEFUL with low score, soften recommendation
            if overlay.get("decision") == "NOT_USEFUL" and overlay.get("usefulness_score", 0) < 35:
                if core_eval.get("final_recommendation") == "IMPLEMENT_NOW":
                    core_eval["final_recommendation"] = "PILOT_FIRST"
                elif core_eval.get("final_recommendation") == "PILOT_FIRST":
                    core_eval["final_recommendation"] = "EVALUATE_QUARTERLY"
        except Exception as e:
            print(f"[overlay] attach error: {e}")

        return core_eval

    def _ai_evaluation(self, tool: Dict, workflow: Dict) -> Dict:
        """AI-powered evaluation of tool impact with semantic alignment"""

        required = workflow.get('required_keywords', [])
        caps = tool.get('primary_capabilities', [])
        current_state = workflow.get('current_state', {})

        # Build a small alignment helper for the model
        matched = []
        missing = []
        tool_text = (" ".join(caps) + " " + tool.get("name","") + " " + tool.get("category","")).lower()
        for k in required:
            if k.lower() in tool_text:
                matched.append(k)
            else:
                missing.append(k)

        prompt = f"""
        You are evaluating whether a tool should be used in a specific workflow.

        TOOL:
        - Name: {tool.get('name')}
        - Category: {tool.get('category', 'Unknown')}
        - Capabilities: {', '.join(caps[:10]) if caps else 'N/A'}
        - Signal Strength: {tool.get('signal_strength')}
        - Score: {tool.get('final_score', 0)}/100

        WORKFLOW:
        - Name: {workflow.get('name')}
        - Goal: {workflow.get('description','')}
        - Required Keywords: {', '.join(required) if required else 'N/A'}
        - Current State: {json.dumps(current_state) if current_state else 'N/A'}

        ALIGNMENT (pre-computed):
        - Matched keywords: {', '.join(matched) if matched else 'None'}
        - Missing keywords: {', '.join(missing) if missing else 'None'}

        TASK:
        Evaluate this tool's usefulness for THIS workflow.
        - Consider semantic fit, not just literal keyword overlap.
        - If it can be adapted to the workflow goal, score it positively with clear caveats.
        - If it’s clearly irrelevant, explain why.

        IMPORTANT: Return ONLY valid JSON, no prose. EXACT KEYS:
        {{
          "integration_type": "REPLACE_WORKFLOW" | "AUGMENT_WORKFLOW" | "EXPERIMENT" | "SKIP",
          "implementation_days": 5,
          "monthly_time_saved": 10,
          "specific_benefits": ["benefit1","benefit2"],
          "specific_risks": ["risk1","risk2"],
          "replaces_tools": [],
          "works_with": [],

          "why_useful": "one-paragraph summary of why this helps the workflow (or an honest note if it doesn't)",
          "when_not_useful": "one-paragraph summary of scenarios where it's a poor fit",
          "best_for": ["task/use-case bullets tailored to this workflow"],
          "not_for": ["task/use-case bullets this tool is NOT good for"],
          "integration_notes": ["glue code, APIs, connectors, content/asset prerequisites, review loops"],
          "example_prompt": "a SINGLE prompt tailored to this workflow and this tool",
          "example_pipeline": ["step 1", "step 2", "step 3"],
          "data_privacy_flags": ["notes like 'sends data to vendor', 'PII caution', 'supports self-hosting'"],

          "keyword_alignment": {{
            "matched": {json.dumps(matched)},
            "missing": {json.dumps(missing)},
            "overall": "HIGH" | "MEDIUM" | "LOW"
          }},

          "confidence": 0.0,
          "reasoning": "short justification for the recommendation",
          "final_recommendation": "IMPLEMENT_NOW" | "PILOT_FIRST" | "EVALUATE_QUARTERLY" | "WAIT_OR_SKIP" | "NEEDS_RESEARCH",
          "payback_months": 6.0
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a JSON API that returns only valid JSON. Never include text outside JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=700,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content

            # Strict JSON parse
            evaluation = json.loads(response_text)

            # Defaults (safety)
            evaluation.setdefault('integration_type', 'EXPERIMENT')
            evaluation.setdefault('implementation_days', 5)
            evaluation.setdefault('monthly_time_saved', 5)
            evaluation.setdefault('specific_benefits', [])
            evaluation.setdefault('specific_risks', [])
            evaluation.setdefault('replaces_tools', [])
            evaluation.setdefault('works_with', [])
            evaluation.setdefault('why_useful', '')
            evaluation.setdefault('when_not_useful', '')
            evaluation.setdefault('best_for', [])
            evaluation.setdefault('not_for', [])
            evaluation.setdefault('integration_notes', [])
            evaluation.setdefault('example_prompt', '')
            evaluation.setdefault('example_pipeline', [])
            evaluation.setdefault('data_privacy_flags', [])
            evaluation.setdefault('keyword_alignment', {"matched": matched, "missing": missing, "overall": "MEDIUM"})
            evaluation.setdefault('confidence', tool.get('signal_confidence', 0.5))
            evaluation.setdefault('reasoning', 'Analysis based on available data')
            evaluation.setdefault('final_recommendation', 'NEEDS_RESEARCH')

            # Compute payback if missing or invalid
            impl_days = max(1, int(evaluation.get('implementation_days', 5)))
            monthly_time_saved = float(evaluation.get('monthly_time_saved', 5))
            if monthly_time_saved > 0:
                implementation_cost = impl_days * 8 * 75  # $75/h
                monthly_savings = monthly_time_saved * 75
                evaluation['payback_months'] = round(implementation_cost / monthly_savings, 2)
            else:
                evaluation['payback_months'] = 999.0

            # Recommendation based on payback (if model didn't set it meaningfully)
            if evaluation.get('final_recommendation') in ('', None, 'NEEDS_RESEARCH'):
                pm = evaluation['payback_months']
                if pm < 3:
                    evaluation['final_recommendation'] = 'IMPLEMENT_NOW'
                elif pm < 6:
                    evaluation['final_recommendation'] = 'PILOT_FIRST'
                elif pm < 12:
                    evaluation['final_recommendation'] = 'EVALUATE_QUARTERLY'
                else:
                    evaluation['final_recommendation'] = 'WAIT_OR_SKIP'

            return evaluation

        except Exception as e:
            print(f"AI evaluation error: {e}")
            return self._heuristic_evaluation(tool, workflow)

    def _heuristic_evaluation(self, tool: Dict, workflow: Dict) -> Dict:
        """Fallback heuristic evaluation (keyword-driven)"""

        score = tool.get('final_score', 0)
        signal = tool.get('signal_strength', 'MODERATE')
        required = workflow.get('required_keywords', [])
        caps = tool.get('primary_capabilities', [])
        tool_text = (" ".join(caps) + " " + tool.get("name","") + " " + tool.get("category","")).lower()
        matched = [k for k in required if k.lower() in tool_text]
        missing = [k for k in required if k.lower() not in tool_text]

        if score > 80 and signal in ['GAME_CHANGING', 'HIGH']:
            integration_type = 'REPLACE_WORKFLOW'
            recommendation = 'IMPLEMENT_NOW'
            why = "High score and strong signal; likely replaces parts of the workflow with better automation/quality."
            when_no = "If strict compliance or on-prem constraints block usage; or if niche tasks require domain-specific tooling not covered."
        elif score > 60:
            integration_type = 'AUGMENT_WORKFLOW'
            recommendation = 'PILOT_FIRST'
            why = "Good fit to enhance existing workflow without full replacement."
            when_no = "If missing capabilities are critical: " + (", ".join(missing) if missing else "none")
        elif score > 40:
            integration_type = 'EXPERIMENT'
            recommendation = 'EVALUATE_QUARTERLY'
            why = "Some potential, but gaps or low evidence suggest a small trial first."
            when_no = "If your priorities rely on the missing keywords: " + (", ".join(missing) if missing else "none")
        else:
            integration_type = 'SKIP'
            recommendation = 'WAIT_OR_SKIP'
            why = "Insufficient alignment/evidence to justify effort."
            when_no = "Low value relative to integration cost."

        return {
            'integration_type': integration_type,
            'final_recommendation': recommendation,
            'reasoning': why,
            'confidence': tool.get('signal_confidence', 0.5),
            'keyword_alignment': {
                'matched': matched,
                'missing': missing,
                'overall': "HIGH" if matched else "LOW"
            }
        }

    # ---------- The following two methods are unchanged in behavior ----------
    # (Kept so your Flask app can call them as before.)
    def explain_scores(self, tool: Dict, workflow: Dict) -> Dict:
        """Convert numeric scores to human-readable explanations"""
        if not self.client:
            return self._heuristic_explanation(tool)

        prompt = f"""
        Explain these scores in practical terms. Return ONLY valid JSON:

        Tool: {tool.get('name')}
        Keyword Match: {tool.get('keyword_match_score', 0):.0f}%
        Implementation: {tool.get('implementation_score', 0):.0f}%
        Evidence: {tool.get('evidence_score', 0):.0f}%
        Final Score: {tool.get('final_score', 0):.1f}/100

        {{
            "keyword_explanation": "what keyword match means",
            "implementation_explanation": "what implementation means",
            "evidence_explanation": "what evidence tells us",
            "overall_meaning": "what final score means",
            "bottom_line": "should you use this?"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Return only valid JSON. Be practical and specific."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # Ensure all fields exist
            result.setdefault('keyword_explanation', 'Keyword match analysis')
            result.setdefault('implementation_explanation', 'Implementation assessment')
            result.setdefault('evidence_explanation', 'Evidence evaluation')
            result.setdefault('overall_meaning', 'Overall assessment')
            result.setdefault('bottom_line', 'Needs further evaluation')

            return result

        except Exception as e:
            print(f"AI explanation error: {e}")
            return self._heuristic_explanation(tool)

    def _heuristic_explanation(self, tool: Dict) -> Dict:
        """Fallback explanations for scores"""
        keyword_score = tool.get('keyword_match_score', 0)
        implementation_score = tool.get('implementation_score', 0)
        evidence_score = tool.get('evidence_score', 0)
        final_score = tool.get('final_score', 0)

        # Keyword explanation
        if keyword_score >= 80:
            keyword_exp = "Perfect fit - has all the capabilities you're looking for"
        elif keyword_score >= 60:
            keyword_exp = "Good fit - covers most of your requirements"
        elif keyword_score >= 40:
            keyword_exp = "Partial fit - covers some key areas but missing others"
        else:
            keyword_exp = "Poor fit - doesn't match what you're looking for"

        # Implementation explanation
        if implementation_score >= 80:
            impl_exp = "Ready to deploy - minimal setup required, production-ready"
        elif implementation_score >= 60:
            impl_exp = "Moderate effort - a few days to integrate properly"
        elif implementation_score >= 40:
            impl_exp = "Significant effort - weeks of work to implement"
        else:
            impl_exp = "Major undertaking - months of effort or experimental"

        # Evidence explanation
        if evidence_score >= 80:
            evidence_exp = "Battle-tested - widely used in production with proven results"
        elif evidence_score >= 60:
            evidence_exp = "Promising - some production use and positive feedback"
        elif evidence_score >= 40:
            evidence_exp = "Early stage - limited real-world validation"
        else:
            evidence_exp = "Unproven - no evidence of production success"

        # Overall meaning
        if final_score >= 80:
            overall = "Excellent match - this tool could transform your workflow"
            bottom = "Start a pilot immediately - high confidence in success"
        elif final_score >= 60:
            overall = "Good option - worth serious consideration"
            bottom = "Run a small pilot to validate benefits"
        elif final_score >= 40:
            overall = "Possible fit - needs more investigation"
            bottom = "Add to backlog for future evaluation"
        else:
            overall = "Not recommended - too many red flags"
            bottom = "Skip this and look for alternatives"

        return {
            'keyword_explanation': keyword_exp,
            'implementation_explanation': impl_exp,
            'evidence_explanation': evidence_exp,
            'overall_meaning': overall,
            'bottom_line': bottom
        }

    def compare_with_current(self, tool: Dict, workflow: Dict) -> Dict:
        """Compare tool with current workflow setup"""
        current_state = workflow.get('current_state', {})

        if not self.client:
            return {
                'comparison': 'Unable to compare without AI',
                'advantages': [],
                'disadvantages': [],
                'migration_complexity': 'UNKNOWN'
            }

        prompt = f"""
        Compare new tool with current setup. Return ONLY valid JSON:

        New Tool: {tool.get('name')}
        Capabilities: {', '.join(tool.get('primary_capabilities', [])[:3])}

        Current: {json.dumps(current_state) if current_state else 'No current tools'}

        {{
            "comparison": "brief summary",
            "advantages": ["adv1", "adv2"],
            "disadvantages": ["dis1"],
            "migration_complexity": "LOW" | "MEDIUM" | "HIGH",
            "keeps_working": [],
            "replaces": []
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Return only valid JSON. Be concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure required fields
            result.setdefault('comparison', 'Comparison analysis')
            result.setdefault('advantages', [])
            result.setdefault('disadvantages', [])
            result.setdefault('migration_complexity', 'MEDIUM')
            result.setdefault('keeps_working', [])
            result.setdefault('replaces', [])

            return result

        except Exception as e:
            print(f"Comparison error: {e}")
            return {
                'comparison': 'Comparison failed - using fallback analysis',
                'advantages': ['Potential improvement over current setup'],
                'disadvantages': ['Integration complexity unknown'],
                'migration_complexity': 'MEDIUM',
                'keeps_working': [],
                'replaces': []
            }


if __name__ == "__main__":
    evaluator = WorkflowEvaluator()

    example_tool = {
        'name': 'Claude 3.5',
        'category': 'AI/LLM',
        'primary_capabilities': ['Text generation', 'Reasoning', 'Analysis'],
        'signal_strength': 'HIGH',
        'signal_confidence': 0.7,
        'final_score': 75.5
    }

    example_workflow = {
        'name': 'Marketing Content Generation',
        'description': 'Automated content creation for marketing campaigns',
        'required_keywords': ['content', 'writing', 'marketing', 'api', 'generation'],
        'current_state': {
            'existing_tools': ['Notion', 'HubSpot'],
            'monthly_cost': 500,
            'pain_points': ['Long drafting cycles', 'Inconsistent tone']
        }
    }

    evaluation = evaluator.evaluate(example_tool, example_workflow)
    print("Evaluation:", json.dumps(evaluation, indent=2))
