## The Problem

  * **Information Overload**: Drowning in news from RSS feeds, social media, and tech blogs.
  * **Hype vs. Reality**: Difficulty distinguishing between genuine breakthroughs and marketing fluff.
  * **Relevance Gap**: A tool might be powerful, but is it useful for *our specific workflows*?
  * **Wasted Effort**: Teams spend hours evaluating tools that are a poor fit, immature, or have a low ROI.

## The Solution

This platform provides a systematic, automated pipeline to ingest, score, and evaluate GenAI tools in the context of your team's specific needs.

### Key Features

  * **Automated Ingestion**: Continuously scans dozens of RSS feeds (TechCrunch, Product Hunt, GitHub Trending, Reddit, ArXiv) to discover new tools and announcements.
  * **Intelligent Classification**: Uses an AI-powered classifier (with a robust heuristic fallback) to categorize content (Tool, Announcement, Research), extract primary capabilities, and assess technical depth.
  * **Common-Sense Tool Gate**: A powerful pre-filter that uses rules and semantic embeddings to discard irrelevant content like games or articles, ensuring only adoptable tools are scored.
  * **Multi-Stage Scoring Pipeline**: Each tool is passed through a sophisticated scoring engine that evaluates:
      * **Signal Strength**: Classifies tools as `GAME_CHANGING`, `HIGH`, `PROMISING`, `MODERATE`, or `NOISE`.
      * **Marketing Fluff**: Detects and penalizes vague, unverifiable marketing claims.
      * **ROI Analysis**: Calculates an estimated payback period, implementation cost, and monthly benefit.
      * **Evidence & Maturity**: Scores the tool based on real-world evidence like GitHub stars, production deployments, and documentation.
  * **Contextual Workflow Analysis**: Moves beyond generic scores by using an **AI Usefulness Overlay** to evaluate a tool's specific utility for a defined workflow (e.g., "Customer Support Bot"), providing creative integration ideas, best-fit tasks, and potential blockers.
  * **Discovery vs. Strict Mode**: Allows you to switch between a conservative "Strict Mode" (prioritizing mature, proven tools) and a creative "Discovery Mode" (surfacing novel, experimental gems).

-----

## How It Works

The platform operates through a simple but powerful data flow:

1.  **Ingest**: The `EnhancedIngester` (`ingest.py`) runs periodically, fetching new items from configured RSS feeds. It uses AI to write concise descriptions and classifies items as potential tools or news, saving them to `data/tools.json` and `data/news.json`.
2.  **Score**: When you visit the `/report` page, the Flask app (`app.py`) loads the tools and runs each one through the `EnhancedWorkflowScorer` (`scorer.py`). This is where the multi-stage analysis happens, generating a final score and detailed metrics.
3.  **Evaluate**: On a workflow-specific page (e.g., `/workflow/customer_support_bot`), the `WorkflowEvaluator` (`workflow_evaluator.py`) performs a deeper, contextual analysis, generating qualitative insights about how a tool could augment or replace parts of that specific workflow.
4.  **Present**: The Flask application renders the results in a clean, data-rich web interface using HTML templates.

-----

## Setup and Installation

### Prerequisites

  * Python 3.8+
  * `pip` package installer

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/genai-intelligence-platform.git
cd genai-intelligence-platform
```

### 2\. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file should be created with libraries like `Flask`, `openai`, `feedparser`, `python-dotenv`, and `numpy`.)*

### 3\. Configure Environment Variables

The platform heavily relies on the OpenAI API for its most advanced features.

Create a `.env` file in the project root:

```
cp .env.example .env
```

Now, edit the `.env` file and add your API key:

```env
OPENAI_API_KEY="sk-..."
```

If you don't provide an API key, the application will gracefully fall back to a less powerful, heuristics-based mode.

### 4\. Run the Application

```bash
python app.py
```

The server will start, and you can access the platform at **http://localhost:5000**.

-----

## Usage

  * **Dashboard (`/`)**: The main landing page with high-level stats and links to reports.
  * **Intelligence Report (`/report`)**: A sortable list of all ingested and scored tools.
      * **Discovery Mode**: To find more experimental tools, append `?mode=discovery` to the URL: `http://localhost:5000/report?mode=discovery`.
  * **Workflow Analysis (`/workflow/<workflow_id>`)**: A detailed analysis of the top-scoring tools for a specific, predefined workflow.
  * **Run Ingestion (`/api/run-analysis`)**: Visit this URL in your browser (or use `curl`) to trigger a fresh scan of all RSS feeds.
  * **Debug Scorer (`/api/debug-scorer`)**: Shows the detailed JSON output of the scorer for the first tool in your database, which is useful for debugging the scoring logic.

## Configuration

### Adding RSS Feeds

To add, remove, or change the data sources, edit the `get_all_feeds` method in `ingest.py`.

```python
# ingest.py

def get_all_feeds(self):
    """YOUR ACTUAL RSS FEEDS WITH CORRECT URLS"""
    return [
        ('TechCrunch AI', 'https://techcrunch.com/category/artificial-intelligence/feed/'),
        ('Product Hunt Today', 'https://www.producthunt.com/feed?category=artificial-intelligence'),
        # ... add more feeds here
    ]
```

### Defining Workflows

Workflows are defined in `data/workflows.json`. You can edit this file to match your team's internal processes. The `id` is used in the URL, and the `required_keywords` are crucial for the scoring algorithm.

```json
{
  "workflows": [
    {
      "id": "customer_support_bot",
      "name": "Customer Support Bot",
      "description": "AI-powered tier-1 support automation for our SaaS product.",
      "required_keywords": ["ai", "api", "support", "automation", "nlp", "chatbot"],
      "constraints": {
        "max_monthly_cost": 500
      },
      "current_state": {
        "existing_solution": "Zendesk Answer Bot",
        "current_monthly_cost": 300,
        "pain_points": ["Slow to train", "Doesn't handle complex queries well"]
      }
    }
  ]
}
```
