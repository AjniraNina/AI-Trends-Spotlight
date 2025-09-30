from flask import Flask, render_template, jsonify, request
import json
import os
from scorer import EnhancedWorkflowScorer
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

def load_data():
    """Load tools and workflows from JSON files"""
    tools_path = 'data/tools.json'
    workflows_path = 'data/workflows.json'
    
    # Load tools
    if os.path.exists(tools_path):
        with open(tools_path, 'r') as f:
            tools = json.load(f)
    else:
        tools = {}
    
    # Load workflows
    if os.path.exists(workflows_path):
        with open(workflows_path, 'r') as f:
            workflows = json.load(f)
    else:
        workflows = {"workflows": []}
    
    return tools, workflows

@app.route('/')
def index():
    """Main dashboard"""
    tools, workflows = load_data()
    return render_template('index.html', 
                         tool_count=len(tools),
                         workflow_count=len(workflows.get('workflows', [])))

@app.route('/report')
def generic_report():
    """Intelligence report using the actual scorer"""
    tools, _ = load_data()
    
    # Check if we're in discovery mode (from query parameter)
    discovery_mode = request.args.get('mode', '').lower() == 'discovery'
    strict_mode = not discovery_mode
    
    # Initialize the scorer with appropriate mode
    scorer = EnhancedWorkflowScorer(strict_mode=strict_mode)
    
    if discovery_mode:
        print("\n🚀 DISCOVERY MODE ENABLED - Surfacing hidden gems...")
    else:
        print("\n📋 STRICT MODE - Conservative evaluation...")
    
    # Process each tool through the actual scorer
    analyzed_tools = []
    for tool_id, tool in list(tools.items())[:50]:  # Increased from 20 to 50 in discovery mode
        print(f"Scoring {tool.get('name')}...")
        
        try:
            # Use the actual scorer with empty workflow for generic scoring
            score_result = scorer.score_tool_for_workflow(
                tool, 
                {'required_keywords': [], 'constraints': {}, 'current_state': {}}
            )
            
            # Only process eligible tools - skip everything else
            if not score_result.get('eligible', False):
                print(f"  Skipping: {score_result.get('reason', 'Not eligible')}")
                continue
            
            # Merge scoring results with tool data
            enhanced_tool = {**tool, **score_result}
            
            # Convert scorer outputs to display format
            enhanced_tool['actionability_score'] = score_result.get('final_score', 0) / 100
            enhanced_tool['confidence_in_assessment'] = score_result.get('signal_confidence', 0.5)
            
            # Add feed intelligence data
            enhanced_tool['feed_bonus'] = score_result.get('feed_bonus', 0)
            enhanced_tool['feed_repetition'] = score_result.get('feed_repetition', 'low')
            enhanced_tool['category'] = score_result.get('category', 'General')
            enhanced_tool['primary_capabilities'] = score_result.get('primary_capabilities', [])
            
            # Use actual scores for metrics
            enhanced_tool['metrics'] = {
                'technicality': score_result.get('implementation_score', 0) / 100,
                'businessImpact': score_result.get('evidence_score', 0) / 100,
                'maturity': score_result.get('keyword_match_score', 0) / 100
            }
            
            # Extract cost info from ROI analysis
            roi = score_result.get('roi_analysis', {})
            enhanced_tool['time_saved_per_month'] = int(roi.get('monthly_benefit', 0) / 75) if roi else 0
            
            # Format cost benefit
            if 'cost_benefit' not in enhanced_tool:
                enhanced_tool['cost_benefit'] = {}
            
            # Use ROI data for costs
            monthly_cost = roi.get('monthly_cost', 0) if roi else 0
            enhanced_tool['cost_benefit']['monthly_cost'] = 'Free' if monthly_cost == 0 else f'${monthly_cost:.0f}'
            enhanced_tool['cost_benefit']['setup_cost'] = 'LOW' if roi.get('total_implementation_cost', 0) < 5000 else 'MEDIUM'
            
            # Use the scorer's immediate action
            enhanced_tool['immediate_action'] = score_result.get('immediate_action', 
                                                f"Evaluate {tool.get('name')}")
            
            # Use the full explanation from scorer (don't slice it)
            enhanced_tool['why_important'] = score_result.get('explanation', tool.get('description', '')[:100])
            
            analyzed_tools.append(enhanced_tool)
            
        except Exception as e:
            print(f"Error scoring {tool.get('name')}: {e}")
            # Skip tools that fail to score - no fallback values
            continue
    
    # Sort by signal strength and actionability
    signal_order = {'GAME_CHANGING': 5, 'HIGH': 4, 'PROMISING': 3, 'MODERATE': 2, 'NOISE': 1}
    analyzed_tools.sort(
        key=lambda x: (
            signal_order.get(x.get('signal_strength', 'MODERATE'), 2),
            x.get('actionability_score', 0)
        ),
        reverse=True
    )
    
    return render_template('report.html', tools=analyzed_tools)

@app.route('/api/analyze-tool/<tool_id>')
def analyze_single_tool_api(tool_id):
    """Analyze a single tool with enhanced AI analysis"""
    try:
        tools, _ = load_data()
        
        if tool_id not in tools:
            return jsonify({'error': 'Tool not found'}), 404
        
        tool = tools[tool_id]
        
        # Check if we have OpenAI configured
        if os.getenv('OPENAI_API_KEY'):
            try:
                from report_generator import IntelligentReportGenerator, EnhancedReportFormatter
                generator = IntelligentReportGenerator()
                formatter = EnhancedReportFormatter()
                
                # Generate analysis
                tool_report = generator.generate_tool_report(tool)
                formatted = formatter.format_tool_card(tool_report)
                return jsonify(formatted)
            except Exception as e:
                print(f"AI analysis failed: {e}")
        
        # Fallback: Use the scorer for analysis
        scorer = EnhancedWorkflowScorer()
        score_result = scorer.score_tool_for_workflow(
            tool,
            {'required_keywords': [], 'constraints': {}, 'current_state': {}}
        )
        
        return jsonify({
            'name': tool.get('name'),
            'description': tool.get('description'),
            'signal_strength': score_result.get('signal_strength', 'MODERATE'),
            'final_score': score_result.get('final_score', 0),
            'immediate_action': score_result.get('immediate_action', 'Review and evaluate'),
            'explanation': score_result.get('explanation', ''),
            'enhanced': False
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/workflow/<workflow_id>')
def workflow_analysis(workflow_id):
    """Workflow-specific analysis with AI evaluation"""
    tools, workflows = load_data()
    scorer = EnhancedWorkflowScorer()
    
    # Import WorkflowEvaluator
    try:
        from workflow_evaluator import WorkflowEvaluator
        evaluator = WorkflowEvaluator()
        has_evaluator = True
    except Exception as e:
        print(f"WorkflowEvaluator not available: {e}")
        evaluator = None
        has_evaluator = False
    
    # Find workflow
    workflow = next((w for w in workflows.get('workflows', []) 
                    if w['id'] == workflow_id), None)
    
    if not workflow:
        return "Workflow not found", 404
    
    # Score all tools
    results = []
    for tool_id, tool in tools.items():
        try:
            score_result = scorer.score_tool_for_workflow(tool, workflow)
            if score_result.get('eligible', False):
                tool_with_score = {**tool, **score_result}
                
                # Add AI evaluation if available
                if has_evaluator:
                    # Get AI evaluation of impact
                    evaluation = evaluator.evaluate(tool_with_score, workflow)
                    tool_with_score['evaluation'] = evaluation
                    
                    # Get human-readable explanations
                    explanations = evaluator.explain_scores(tool_with_score, workflow)
                    tool_with_score['explanations'] = explanations
                    
                    # Compare with current setup
                    comparison = evaluator.compare_with_current(tool_with_score, workflow)
                    tool_with_score['comparison'] = comparison
                
                results.append(tool_with_score)
        except Exception as e:
            print(f"Error scoring {tool.get('name')}: {e}")
            continue
    
    # Sort by final score
    results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    return render_template('workflow.html', 
                         workflow=workflow,
                         tools=results[:10],
                         has_ai=has_evaluator)

@app.route('/api/run-analysis')
def run_analysis():
    """Trigger new ingestion"""
    try:
        from ingest import EnhancedIngester
        ingester = EnhancedIngester()
        new_tools, new_news = ingester.ingest_from_feeds()
        return jsonify({
            'status': 'success',
            'new_tools': len(new_tools),
            'new_news': len(new_news),
            'message': f'Added {len(new_tools)} tools and {len(new_news)} news items'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/research/<tool_name>')
def research_tool(tool_name):
    """Research a specific tool"""
    try:
        from intelligence_researcher import IntelligenceResearcher
        researcher = IntelligenceResearcher()
        research = researcher.research_tool(tool_name, 'manual')
        
        evaluation = {
            'final_recommendation': 'PILOT' if research['verdict'] == 'VERIFIED_REAL' else 'WAIT',
            'payback_months': 3
        }
        
        return render_template('research_report.html',
                             tool_name=tool_name,
                             research=research,
                             evaluation=evaluation)
    except Exception as e:
        return f"Error researching tool: {str(e)}", 500

@app.route('/api/debug-scorer')
def debug_scorer():
    """Debug endpoint to test scorer"""
    tools, _ = load_data()
    scorer = EnhancedWorkflowScorer()
    
    # Get first tool for testing
    tool_id = list(tools.keys())[0] if tools else None
    if not tool_id:
        return jsonify({'error': 'No tools found'})
    
    tool = tools[tool_id]
    
    # Score it
    score_result = scorer.score_tool_for_workflow(
        tool, 
        {'required_keywords': [], 'constraints': {}, 'current_state': {}}
    )
    
    return jsonify({
        'tool_name': tool.get('name'),
        'scoring_result': score_result,
        'signal_strength': score_result.get('signal_strength'),
        'final_score': score_result.get('final_score'),
        'implementation_score': score_result.get('implementation_score'),
        'evidence_score': score_result.get('evidence_score'),
        'keyword_match_score': score_result.get('keyword_match_score')
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("\n" + "="*50)
    print("🚀 GenAI Intelligence Platform")
    print("="*50)
    
    # Check for API key
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key found - Enhanced analysis available")
    else:
        print("⚠️  No OpenAI API key - Using basic scoring only")
        print("   Set OPENAI_API_KEY env variable for AI-powered analysis")
    
    print("\nStarting server...")
    print("Open your browser to: http://localhost:5000")
    print("\nAvailable routes:")
    print("  / - Dashboard")
    print("  /report - Intelligence report (with real scoring)")
    print("  /workflow/<id> - Workflow analysis")
    print("  /research/<tool> - Deep research on a tool")
    print("  /api/run-analysis - Refresh data from RSS feeds")
    print("  /api/debug-scorer - Test the scoring system")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)