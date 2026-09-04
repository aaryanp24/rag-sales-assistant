"""
One prompt template per feature. Each function returns (system_prompt, user_prompt)
ready to hand to llm.generate(). Keeping these separate (rather than one generic
"write marketing content" prompt) is the actual design decision worth calling out
on a resume/in an interview - different features need different context shapes,
tones, and constraints.
"""

def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(no matching internal documents found)"
    return "\n\n".join(f"[Source: {c['source']}]\n{c['text']}" for c in chunks)


def cold_call_script(org_context: list[dict], company_name: str, company_research: str, notes: str = "") -> tuple[str, str]:
    system = (
        "You are a sales enablement assistant. You write natural, confident cold-call "
        "scripts for a B2B sales rep. Ground every claim about the rep's own company "
        "strictly in the provided context - never invent services, results, or client names "
        "that aren't in the context. Keep it conversational, not a wall of text a rep "
        "would never actually say out loud."
    )
    user = f"""Write a cold-call opening script (about 30-45 seconds spoken) for calling {company_name}.

Our company's relevant background (use only what's here for any claims about us):
{_format_context(org_context)}

What we know about {company_name}:
{company_research or "(no external research available - keep this generic but professional)"}

Extra notes from the rep: {notes or "(none)"}

Structure: opening hook referencing something specific to {company_name} if possible,
a one-line value prop grounded in our context, and a soft ask for 15 minutes.
Include 2-3 likely objections and one-line responses to each."""
    return system, user


def cold_email_script(org_context: list[dict], company_name: str, company_research: str, notes: str = "") -> tuple[str, str]:
    system = (
        "You are a sales copywriter. You write short, specific cold outreach emails. "
        "No generic filler like 'I hope this finds you well'. Ground every claim about "
        "the sender's company strictly in the provided context."
    )
    user = f"""Write a cold outreach email to a decision-maker at {company_name}.

Our company's relevant background:
{_format_context(org_context)}

What we know about {company_name}:
{company_research or "(no external research available - keep this generic but professional)"}

Extra notes: {notes or "(none)"}

Keep the email under 120 words. Include a subject line. One clear, low-friction call to action."""
    return system, user


def pricing_estimate(org_context: list[dict], scope_description: str) -> tuple[str, str]:
    system = (
        "You are a proposal assistant that builds cost breakdowns. CRITICAL RULE: only use "
        "prices, rates, and packages that literally appear in the provided context. If the "
        "context doesn't contain enough pricing information to answer, say so explicitly "
        "instead of guessing a number. Never invent a dollar figure."
    )
    user = f"""A prospective client has described this project scope:
"{scope_description}"

Our pricing/rate information on file:
{_format_context(org_context)}

Produce:
1. A line-item breakdown (only using rates found above)
2. A total estimate range (only if the source data supports it)
3. A one-line note on what info is missing, if the rate card doesn't fully cover this scope."""
    return system, user


def deliverable_match(org_context: list[dict], client_need: str) -> tuple[str, str]:
    system = (
        "You help match a client's stated need to the most relevant deliverable/case study "
        "the company already has, based only on the provided context."
    )
    user = f"""Client need: "{client_need}"

Our available deliverables/case studies/templates:
{_format_context(org_context)}

Recommend the single best-matching item, explain why in 2-3 sentences, and suggest one way to "
"customize it for this client. If nothing in the context is a good match, say so."""
    return system, user
