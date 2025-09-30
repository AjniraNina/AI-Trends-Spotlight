import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    from openai import OpenAI
    _HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _HAS_OPENAI = False

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

    # strong positive indicators (tool-y)
    POSITIVE_WORDS = {
        "api","sdk","library","framework","cli","package","plugin","extension",
        "tool","service","platform","server","self-hosted","open source","open-source",
        "repo","repository","npm","pypi","pip","docker","helm","kubernetes","endpoint",
        "install","setup","configure","deploy","integration","docs","readme","examples",
        "code","source code","git clone","import","build","compile"
    }

    # strong negative indicators (not adoptable tooling)
    NEGATIVE_WORDS = {
        "game","play now","play the game","leaderboard","score","level","3d maze",
        "steam","roblox","itch.io","trailer","soundtrack","walkthrough","speedrun",
        "wikispeedia","levels","skins","xp"
    }

    # URL/domain hints
    TOOL_DOMAINS = ("github.com","gitlab.com","bitbucket.org","npmjs.com","pypi.org","crates.io","docker.com","hub.docker.com","readthedocs.io")
    GAME_DOMAINS = ("roblox.com","store.steampowered.com","itch.io","epicgames.com")

    # prompts/categories for optional embedding check
    TOOL_PROTOTYPE = "A software tool or developer resource you can adopt, like an SDK, API, CLI, framework, library or deployable app."
    NOT_TOOL_PROTOTYPE = "A game, demo, news post, opinion article, or general content not directly adoptable as a developer tool."

    def __init__(self, enable_embeddings: bool = True):
        self.enable_embeddings = enable_embeddings and _HAS_OPENAI
        self.client = OpenAI() if self.enable_embeddings else None
        self._tool_vec = None
        self._not_tool_vec = None

    def _embed(self, text: str) -> List[float]:
        # small, cheap model; adjust if you prefer
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
        # lazy-init prototypes
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
        # weights: domain (±2), lexical (continuous), artifacts (±3), semantic (±1 range)
        combined = domain_score + lex_score + artifact_score + max(min(sem_score, 1.0), -1.0)

        # map to decision
        # > 1.5 very likely a tool; < 0 likely not a tool; in-between uncertain
        if combined >= 1.5:
            return GateResult(True, min(0.9, 0.55 + combined/6), reasons)
        elif combined <= 0.0:
            return GateResult(False, min(0.9, 0.55 + abs(combined)/6), reasons)
        else:
            # uncertain: treat as not a tool unless content_type already TOOL/ANNOUNCEMENT with dev cues
            return GateResult(False, 0.5, reasons)
