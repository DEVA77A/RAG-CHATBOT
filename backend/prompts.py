"""
WebIntel AI — Prompt Templates

Three templates:
  1. MEGA_ANALYSIS_PROMPT  — Single call → 8 structured outputs
  2. RAG_CHAT_PROMPT       — Contextual Q&A with citations
  3. SYSTEM_PROMPT         — Base system identity

The mega-prompt is the heart of the product. It transforms a raw website
scrape into a full personalized intelligence report in one Gemini call.
"""

# ──────────────────────────────────────────────
# Persona-Specific Score Dimensions
# ──────────────────────────────────────────────

PERSONA_SCORE_DIMENSIONS: dict[str, list[str]] = {
    "student": [
        "Learning Value",
        "Career Value",
        "Research Value",
        "Project Inspiration",
    ],
    "job_seeker": [
        "Hiring Potential",
        "Skill Development",
        "Career Growth",
    ],
    "developer": [
        "API Value",
        "Documentation Quality",
        "Technical Depth",
    ],
    "researcher": [
        "Innovation Score",
        "Research Value",
    ],
    "investor": [
        "Business Potential",
        "Market Strength",
    ],
}

PERSONA_LABELS: dict[str, str] = {
    "student": "Student",
    "job_seeker": "Job Seeker",
    "developer": "Developer",
    "researcher": "Researcher",
    "investor": "Investor",
}

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are WebIntel AI — a Personalized Website Intelligence Assistant.

Your purpose: Transform any website into actionable, persona-specific intelligence.
You don't just summarize websites. You explain WHY a website matters to a specific type of user, WHAT opportunities exist, WHAT skills they need, and WHAT they should do next.

Core principles:
- Be specific and actionable, never generic
- Ground every insight in the actual website content
- Tailor everything to the user's persona
- Surface hidden value that a casual reader would miss
- When detecting opportunities, only include ones with evidence in the content
- When scoring, be calibrated — not every website is a 0.9"""


# ──────────────────────────────────────────────
# Mega Analysis Prompt
# ──────────────────────────────────────────────

def build_mega_prompt(
    content: str,
    persona: str,
    title: str,
    url: str,
    links: list[str] | None = None,
) -> str:
    """
    Build the mega-prompt that generates all 8 analysis outputs in one call.

    This is the most important function in the entire codebase.
    """
    persona_label = PERSONA_LABELS.get(persona, persona.replace("_", " ").title())
    score_dims = PERSONA_SCORE_DIMENSIONS.get(persona, ["Overall Value"])
    score_dims_str = "\n".join(f'      - "{dim}"' for dim in score_dims)

    # Include top links for context (helps opportunity detection)
    links_section = ""
    if links:
        top_links = links[:30]
        links_str = "\n".join(f"  - {link}" for link in top_links)
        links_section = f"""
Links found on the website:
{links_str}
"""

    return f"""Analyze the following website for a **{persona_label}** user.

Website: {title}
URL: {url}
{links_section}
--- WEBSITE CONTENT START ---
{content}
--- WEBSITE CONTENT END ---

Generate a comprehensive intelligence report as a single JSON object with ALL of the following keys.
Be specific, actionable, and deeply relevant to a {persona_label}.

{{
  "summary": "A compelling 3-5 sentence overview of what this website is about, what it offers, and its significance in its domain. Write this for a {persona_label} — highlight what matters most to them.",

  "why_it_matters": {{
    "relevance_score": <float 0.0-1.0, how relevant this website is to a {persona_label}>,
    "explanation": "<2-3 sentences explaining WHY a {persona_label} should care about this website. Be specific about the value they gain.>",
    "key_takeaways": [
      "<takeaway 1 — the single most important thing a {persona_label} should know>",
      "<takeaway 2>",
      "<takeaway 3>",
      "<takeaway 4>"
    ],
    "recommended_actions": [
      "<specific action 1 they should take after visiting this website>",
      "<specific action 2>",
      "<specific action 3>"
    ]
  }},

  "insights": [
    {{
      "category": "<category name relevant to a {persona_label}, e.g. 'Learning Resources', 'Career Opportunities', 'Technical Architecture'>",
      "items": [
        {{
          "title": "<specific insight title>",
          "description": "<2-3 sentences with actionable detail>",
          "relevance": "high|medium|low"
        }}
      ]
    }}
  ],

  "action_plan": [
    {{
      "step": 1,
      "title": "<clear, concise action title>",
      "description": "<what to do and why, specific to a {persona_label}>",
      "priority": "high|medium|low",
      "time_estimate": "<e.g. '30 minutes', '1-2 days', '1 week'>"
    }}
  ],

  "opportunities": [
    {{
      "category": "<one of: job, internship, course, certification, event, program, scholarship, hackathon, competition, developer_program, partner_program>",
      "title": "<opportunity name>",
      "description": "<1-2 sentences about the opportunity>",
      "link": "<direct URL if found in the content or links, otherwise empty string>",
      "match_score": <float 0.0-1.0, how well this opportunity matches a {persona_label}>,
      "why_relevant": "<1 sentence on why this opportunity matters to a {persona_label}>",
      "required_skills": ["<skill1>", "<skill2>"],
      "difficulty": "beginner|intermediate|advanced",
      "recommended_action": "<specific next step to pursue this opportunity>"
    }}
  ],

  "skill_gap": {{
    "required_skills": [
      {{
        "skill": "<skill name>",
        "level": "beginner|intermediate|advanced",
        "context": "<why this skill matters based on the website content>"
      }}
    ],
    "missing_skills": [
      {{
        "skill": "<skill that a typical {persona_label} might need to develop>",
        "priority": "high|medium|low",
        "reason": "<why this skill gap matters>"
      }}
    ],
    "learning_roadmap": [
      {{
        "step": 1,
        "skill": "<skill to learn>",
        "resource_type": "course|tutorial|project|book|documentation",
        "time_estimate": "<e.g. '1 week', '2-3 days'>",
        "priority": "high|medium|low"
      }}
    ]
  }},

  "website_score": {{
    "overall": <float 0.0-1.0, overall value of this website to a {persona_label}>,
    "dimensions": [
{score_dims_str}
    ]
  }},

  "similar_websites": [
    {{
      "name": "<website name>",
      "url": "<full URL>",
      "description": "<1 sentence about what it offers>",
      "why_relevant": "<why a {persona_label} who liked this website should also visit this one>",
      "category": "<e.g. 'competitor', 'complementary', 'alternative', 'learning_resource'>"
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
1. For "insights": Generate 3-5 categories with 2-3 items each. Categories must be persona-specific.
2. For "action_plan": Generate 4-6 concrete, sequential steps. Each must be actionable TODAY.
3. For "opportunities": ONLY include opportunities with evidence in the website content. If none found, return an empty array. Do NOT fabricate opportunities.
4. For "skill_gap": Identify 4-6 required skills and 3-5 missing skills based on the website's domain.
5. For "website_score": Each dimension in the "dimensions" array must be an object with "name" (string), "score" (float 0.0-1.0), and "reason" (string). Use ONLY these dimensions for a {persona_label}:
{score_dims_str}
   Be calibrated — a documentation site might score high on Learning Value but low on Career Value.
6. For "similar_websites": Suggest 3-5 real, well-known websites. They must be actual websites that exist.
7. For "why_it_matters": Be deeply persona-specific. A student cares about learning, a job seeker cares about hiring, an investor cares about market position.

Return ONLY the JSON object. No markdown fences, no explanations, no text before or after the JSON."""


# ──────────────────────────────────────────────
# RAG Chat Prompt
# ──────────────────────────────────────────────

RAG_CHAT_PROMPT = """You are WebIntel AI, a personalized website intelligence assistant.
You are helping a **{persona_label}** understand the website: {title} ({url}).

Use the following context extracted from the website to answer the user's question.
If the context doesn't contain enough information, say so honestly — do not make things up.

--- RELEVANT CONTEXT ---
{context}
--- END CONTEXT ---

{history_section}

User's Question: {question}

Instructions:
- Answer specifically for a {persona_label} — tailor your language and focus
- Be concise but thorough
- If you reference information from the context, mention which part
- If the question is about opportunities, skills, or relevance, leverage your persona expertise
- Do not repeat the question back to the user"""


def build_chat_prompt(
    question: str,
    context_chunks: list[dict],
    persona: str,
    title: str,
    url: str,
    chat_history: list[dict] | None = None,
) -> str:
    """Build the RAG chat prompt with retrieved context and conversation history."""
    persona_label = PERSONA_LABELS.get(persona, persona.replace("_", " ").title())

    # Format context chunks
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        content = chunk.get("content", "")
        score = chunk.get("score", 0.0)
        context_parts.append(f"[Source {i}] (relevance: {score:.2f})\n{content}")

    context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

    # Format chat history
    history_section = ""
    if chat_history:
        history_lines = []
        # Only include last 6 messages to fit context
        recent = chat_history[-6:]
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_section = "Previous conversation:\n" + "\n".join(history_lines) + "\n"

    return RAG_CHAT_PROMPT.format(
        persona_label=persona_label,
        title=title,
        url=url,
        context=context,
        history_section=history_section,
        question=question,
    )
