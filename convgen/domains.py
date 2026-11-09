"""Domain table for multilingual conversation generation.

Every domain here is a 2-persona advisory conversation (an asker and a
responder). Each domain defines whether it's region-sensitive (uses the
language's region/place/staple/currency/regulator facts from languages.py),
its roles with system-prompt templates, and the variation axes
(topic templates, issues, modifiers) the generator samples from to produce
many distinct specs without needing a model call per spec.

Placeholders available in `role_descs`/`topic_templates`: {issue},
{modifier}, {name0}, {name1}, and — region-sensitive domains only —
{place}, {staple}, {currency}, {regulator}, {region}.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    slug: str
    name: str
    region_sensitive: bool
    roles: list[str]
    role_descs: list[str]
    topic_templates: list[str]
    issues: list[str]
    modifiers: list[str]
    search_roles: list[bool]

    def __post_init__(self):
        n = 2
        assert len(self.roles) == n, self.slug
        assert len(self.role_descs) == n, self.slug
        assert len(self.search_roles) == n, self.slug


DOMAINS: list[Domain] = [
    Domain(
        "agriculture", "Agriculture / Agri-Advisory", True,
        ["Farmer", "Agri-Advisor"],
        [
            "You are a farmer near {place}. You grow {staple} and have a real, specific "
            "problem: {issue}, which has been going on for {modifier}. Ask natural follow-up "
            "questions and mention your specific local conditions as they come up.",
            "You are an agricultural extension advisor serving farmers near {place}. Give "
            "specific, locally-relevant advice (not generic platitudes) about {staple} "
            "cultivation. Use web_search for current local pricing, weather, or treatment "
            "recommendations when it would genuinely help. Ask clarifying questions when the "
            "farmer's description is ambiguous.",
        ],
        ["My {staple} crop near {place} is showing {issue}. What should I do?"],
        ["stunted growth", "yellowing leaves", "a pest infestation", "poor soil drainage",
         "low yield this season", "unexpected wilting", "root rot", "irregular rainfall damage",
         "nutrient deficiency signs", "a fungal outbreak", "salinity buildup in the soil",
         "birds/wildlife damage", "delayed flowering", "post-harvest storage losses"],
        ["about two weeks", "the whole season", "three days", "since the last rains",
         "a month now", "since planting", "the past few harvests", "just this week"],
        [False, True],
    ),
    Domain(
        "banking", "Banking & Financial Regulation", True,
        ["Bank Customer", "Compliance Officer"],
        [
            "You are a bank customer near {place} dealing with {issue}. You want to understand "
            "your rights and options clearly. Ask natural follow-up questions.",
            "You are a banking compliance officer familiar with {regulator}'s rules. Give "
            "specific, accurate guidance about {issue} in {currency} terms where relevant. Use "
            "web_search for current regulations or rates when it would help. Ask clarifying "
            "questions when needed.",
        ],
        ["I've run into an issue with my bank account: {issue}. Can you help me understand what's going on?"],
        ["an unexplained hold on funds", "a disputed transaction", "a loan application rejection",
         "KYC document issues", "a sudden account freeze", "unclear overdraft charges",
         "a failed international transfer", "credit score confusion", "a suspicious fraud alert",
         "mortgage refinancing questions", "a joint account dispute", "an unclear fee structure"],
        ["yesterday", "over the past week", "for the third time this month", "just this morning",
         "since I opened the account", "after a recent policy change", "unexpectedly", "repeatedly"],
        [False, True],
    ),
    Domain(
        "customer_support", "Customer Support / Contact Center", False,
        ["Customer", "Support Agent"],
        [
            "You are a customer who ran into {issue} with a product/service. You're frustrated "
            "but reasonable — explain your situation and ask real follow-up questions based on "
            "what the agent tells you.",
            "You are a customer support agent. Be genuinely helpful and specific about {issue}, "
            "not scripted. Use web_search to look up current policy/troubleshooting info when "
            "it would help. Ask clarifying questions when needed.",
        ],
        ["I'm having trouble with {issue}. Can you help me sort this out?"],
        ["a delayed order", "a broken product on arrival", "a billing discrepancy",
         "an account login issue", "a subscription I can't cancel", "a missing refund",
         "conflicting information from two agents", "a warranty claim", "a shipping address error",
         "an app that keeps crashing", "a duplicate charge", "a promo code that won't apply"],
        ["for the second time", "since yesterday", "for over a week", "right after the last update",
         "out of nowhere", "after I upgraded my plan", "since the outage", "for a while now"],
        [False, True],
    ),
    Domain(
        "ecommerce_retail", "E-commerce & Retail", False,
        ["Shopper", "Retail Advisor"],
        [
            "You are a shopper trying to decide about {issue}. Ask real, natural questions "
            "based on what the advisor tells you.",
            "You are a retail shopping advisor. Give specific, honest guidance about {issue}. "
            "Use web_search for current prices/reviews/availability when it would help.",
        ],
        ["I'm trying to figure out {issue} before I buy. Can you help?"],
        ["which model/brand is actually worth the extra cost", "whether to buy now or wait for a sale",
         "sizing/fit uncertainty for an online purchase", "conflicting reviews for the same product",
         "whether a deal is actually a good deal", "return policy confusion",
         "choosing between two very similar products", "whether a marketplace seller is trustworthy"],
        ["for a big purchase", "on a tight budget", "as a gift", "for a first-time buyer",
         "after a bad past experience", "with a return deadline coming up"],
        [False, True],
    ),
    Domain(
        "education", "Education / EdTech", False,
        ["Student", "Tutor"],
        [
            "You are a student struggling with {issue}. Ask genuine follow-up questions based "
            "on what the tutor explains, and mention your specific confusion as it comes up.",
            "You are a patient, specific tutor. Explain {issue} clearly, check understanding, "
            "and use web_search for current curriculum standards or examples when helpful.",
        ],
        ["I'm stuck on {issue} and could use some help understanding it."],
        ["a tricky math concept", "structuring an essay argument", "exam anxiety and study planning",
         "understanding a science concept from class", "a coding assignment that keeps failing",
         "choosing a research topic", "time management across multiple subjects",
         "preparing for a language proficiency test", "understanding feedback on a graded assignment"],
        ["before a deadline", "after missing a class", "for the second time this term",
         "with a test coming up", "since switching schools", "self-studying"],
        [False, True],
    ),
    Domain(
        "healthcare_ops", "Healthcare / Clinical Operations", False,
        ["Clinic Coordinator", "Operations Advisor"],
        [
            "You are a clinic coordinator dealing with {issue} in scheduling/operations (NOT "
            "medical diagnosis — this is administrative/operational, not clinical advice). Ask "
            "real follow-up questions.",
            "You are a healthcare operations advisor. Give specific, practical guidance on "
            "{issue} — patient flow, scheduling, records, staffing — never medical diagnosis. "
            "Use web_search for current best-practice operational guidance when helpful.",
        ],
        ["We're dealing with {issue} at the clinic and need to sort out a better process."],
        ["appointment no-show rates", "patient intake bottlenecks", "records handoff between departments",
         "staff scheduling conflicts", "long wait times at the front desk", "insurance pre-authorization delays",
         "discharge paperwork backlogs", "supply inventory tracking", "patient follow-up reminders falling through"],
        ["for months now", "since a recent system change", "during peak hours", "since staff turnover",
         "since we grew our patient base", "since switching software"],
        [False, True],
    ),
    Domain(
        "insurance", "Insurance", True,
        ["Policyholder", "Insurance Agent"],
        [
            "You are a policyholder near {place} dealing with {issue}. Ask genuine follow-up "
            "questions about your policy and options.",
            "You are an insurance agent familiar with regulations from {regulator}. Give "
            "specific guidance in {currency} terms about {issue}. Use web_search for current "
            "regulations or typical rates when helpful.",
        ],
        ["I need help understanding {issue} with my insurance policy."],
        ["a denied claim", "unclear premium increase", "whether I'm actually covered for a specific event",
         "a slow claims process", "confusing policy fine print", "choosing between coverage tiers",
         "a dispute over claim valuation", "renewal terms that changed without notice",
         "whether a pre-existing condition affects coverage"],
        ["after a recent incident", "at renewal time", "for the first time", "after a rate hike",
         "since switching providers", "after a life change"],
        [False, True],
    ),
    Domain(
        "legal", "Legal Practice", True,
        ["Client", "Legal Advisor"],
        [
            "You are a client near {place} with a general business/civil legal question about "
            "{issue} (NOT criminal defense — general practice only). Ask genuine follow-up "
            "questions.",
            "You are a legal advisor familiar with the local legal system near {place}. Give "
            "careful, specific general guidance about {issue}, note where a client should "
            "consult a licensed local lawyer for their exact situation, and use web_search for "
            "current relevant law when helpful.",
        ],
        ["I have a question about {issue} and want to understand my options."],
        ["a contract dispute with a vendor", "a lease disagreement", "setting up a small business "
         "structure", "an employment dispute", "an intellectual property concern",
         "a partnership disagreement", "a consumer rights issue", "a property boundary dispute",
         "a late-payment dispute with a client"],
        ["that's been going on for weeks", "that just came up", "before signing something",
         "after receiving a formal notice", "before a deadline", "escalating"],
        [False, True],
    ),
    Domain(
        "cross_domain", "Other Suggestive Use Cases (Cross-Domain)", False,
        ["Person", "Advisor"],
        [
            "You are someone dealing with a real, specific problem: {issue}. Ask genuine "
            "follow-up questions, mentioning your specific situation as it comes up.",
            "You are a generalist advisor. Give specific, practical guidance about {issue}. Use "
            "web_search when real current information would help.",
        ],
        ["I need some advice about {issue}."],
        ["planning a cross-country move", "choosing between two job offers", "organizing a community event",
         "starting a side project", "improving a daily routine", "resolving a roommate disagreement",
         "planning a budget for a big life change", "learning a new skill efficiently",
         "deciding on a major purchase"],
        ["with limited time", "on a budget", "for the first time", "with competing priorities",
         "under some pressure", "after trying a few things that didn't work"],
        [False, True],
    ),
    Domain(
        "real_estate", "Real Estate", True,
        ["Home Buyer", "Real Estate Agent"],
        [
            "You are looking for property near {place} and dealing with {issue}. Ask genuine "
            "follow-up questions about the local market.",
            "You are a real estate agent working near {place}. Give specific, locally-grounded "
            "guidance in {currency} terms about {issue}. Use web_search for current local "
            "listings/prices when helpful.",
        ],
        ["I'm trying to figure out {issue} while looking for property near {place}."],
        ["whether prices are likely to rise or fall soon", "how to negotiate on an overpriced listing",
         "what's a fair price for the area", "whether to buy now or rent longer",
         "financing/mortgage options", "red flags to check before making an offer",
         "how a specific neighborhood compares to another", "timing the sale of a current home"],
        ["with a moving deadline", "as a first-time buyer", "on a tight budget", "in a competitive market",
         "after losing a bid already", "while relocating for work"],
        [False, True],
    ),
    Domain(
        "finance_ca", "Finance / Chartered Accountancy Practice", True,
        ["Business Owner", "Chartered Accountant"],
        [
            "You are a small business owner near {place} dealing with {issue}. Ask genuine "
            "follow-up questions in {currency} terms.",
            "You are a chartered accountant familiar with the local tax/compliance system near "
            "{place} (overseen by {regulator} where relevant). Give specific guidance about "
            "{issue}. Use web_search for current tax rates/rules when helpful.",
        ],
        ["I need help with {issue} for my business."],
        ["quarterly tax filing confusion", "reconciling a mismatch in the books", "GST/VAT compliance",
         "payroll tax calculations", "audit preparation", "cash flow forecasting",
         "choosing a business structure for tax purposes", "handling a supplier invoice dispute in the books"],
        ["before the filing deadline", "after a bookkeeping error", "for the first time this year",
         "after switching accounting software", "flagged by an auditor", "after rapid growth"],
        [False, True],
    ),
    Domain(
        "ai_ml", "AI-ML Applications", False,
        ["ML Engineer", "Product Advisor"],
        [
            "You are an ML engineer dealing with {issue}. Ask genuine technical follow-up "
            "questions.",
            "You are a product/ML advisor. Give specific, practical guidance about {issue}. Use "
            "web_search for current techniques/tools when helpful.",
        ],
        ["I'm running into {issue} with our ML system."],
        ["model drift in production", "a training pipeline that keeps timing out", "poor performance "
         "on an edge case the model never saw", "an evaluation metric that doesn't reflect real user "
         "satisfaction", "data labeling inconsistencies", "an inference latency problem at scale",
         "bias detected in model outputs", "choosing between fine-tuning and prompting for a use case"],
        ["since the last retrain", "in production but not in testing", "for a specific user segment",
         "since a data source changed", "intermittently", "at scale but not locally"],
        [False, True],
    ),
    Domain(
        "cms_content", "CMS & Content", False,
        ["Content Editor", "Platform Advisor"],
        [
            "You are a content editor dealing with {issue} on your CMS platform. Ask genuine "
            "follow-up questions.",
            "You are a CMS platform advisor. Give specific guidance about {issue}. Use "
            "web_search for current platform features/best-practices when helpful.",
        ],
        ["I'm having trouble with {issue} on our content platform."],
        ["a broken content workflow between editors", "SEO metadata not updating correctly",
         "images not rendering after a migration", "version control confusion on shared drafts",
         "a publishing schedule that keeps slipping", "duplicate content across pages",
         "a slow page-load caused by unoptimized media", "permission issues between content roles"],
        ["since the last platform update", "for a specific content type only", "intermittently",
         "since onboarding a new team member", "after a migration", "under a tight deadline"],
        [False, True],
    ),
    Domain(
        "ecommerce_platform", "E-commerce", False,
        ["Merchant", "Platform Advisor"],
        [
            "You are an online merchant dealing with {issue} on your store platform. Ask "
            "genuine follow-up questions.",
            "You are an e-commerce platform advisor. Give specific guidance about {issue}. Use "
            "web_search for current platform features/policies when helpful.",
        ],
        ["I'm dealing with {issue} on my store and need advice."],
        ["cart abandonment rate creeping up", "payment gateway failures at checkout",
         "inventory sync issues across sales channels", "a spike in fraudulent orders",
         "shipping rate calculations coming out wrong", "product page conversion dropping",
         "a marketplace policy change affecting listings", "return rate higher than usual for one product"],
        ["since last month", "since a platform update", "during a big sale", "for one product category only",
         "since adding a new payment method", "unexpectedly"],
        [False, True],
    ),
    Domain(
        "finance_productivity", "Finance & Productivity", False,
        ["User", "Productivity Advisor"],
        [
            "You are someone trying to get better at {issue}. Ask genuine follow-up questions.",
            "You are a finance/productivity advisor. Give specific, practical guidance about "
            "{issue}. Use web_search for current tools/methods when helpful.",
        ],
        ["I want to get better at {issue} but don't know where to start."],
        ["budgeting consistently", "tracking recurring subscriptions", "prioritizing tasks across projects",
         "reducing time lost to context-switching", "saving toward a specific goal", "automating repetitive "
         "financial admin", "building a sustainable daily routine", "managing a team's shared task backlog"],
        ["after several failed attempts", "with a new job", "on a tight timeline", "after a big life change",
         "starting from scratch", "trying to unlearn a bad habit"],
        [False, True],
    ),
    Domain(
        "games", "Games", False,
        ["Player", "Game Designer"],
        [
            "You are a player dealing with {issue} in a game. Ask genuine follow-up questions.",
            "You are a game designer. Give specific, thoughtful guidance about {issue}. Use "
            "web_search for current design patterns/examples when helpful.",
        ],
        ["I've got a question about {issue} in the game I'm designing/playing."],
        ["balancing difficulty so it's fair but still challenging", "a progression system that feels "
         "grindy", "matchmaking that doesn't feel fair", "onboarding that loses new players too fast",
         "a monetization approach that doesn't feel predatory", "multiplayer sync issues affecting fairness",
         "a boss fight that playtesters found frustrating not fun", "retention dropping after the first week"],
        ["based on playtester feedback", "after launch", "in early design", "after a balance patch",
         "for a specific player segment", "compared to a competitor's game"],
        [False, True],
    ),
    Domain(
        "generic_fullstack", "Generic Full-stack", False,
        ["Developer", "Architect"],
        [
            "You are a full-stack developer dealing with {issue}. Ask genuine technical "
            "follow-up questions.",
            "You are a software architect. Give specific, practical guidance about {issue}. Use "
            "web_search for current best practices/tools when helpful.",
        ],
        ["I'm dealing with {issue} in our full-stack app."],
        ["choosing state management for a growing frontend", "an API design that's getting hard to "
         "extend", "slow database queries under real load", "authentication flow edge cases",
         "deciding between monolith and services for a new feature", "test coverage gaps in critical paths",
         "a deployment pipeline that's become fragile", "inconsistent data validation between layers"],
        ["as the app has grown", "since onboarding new team members", "found in code review",
         "surfaced by a production bug", "before a big feature launch", "during a refactor"],
        [False, True],
    ),
    Domain(
        "iot_automation", "IoT & Automation", False,
        ["Homeowner", "IoT Specialist"],
        [
            "You are a homeowner/small-facility operator dealing with {issue} in your smart "
            "devices/automation setup. Ask genuine follow-up questions.",
            "You are an IoT/automation specialist. Give specific, practical guidance about "
            "{issue}. Use web_search for current devices/protocols when helpful.",
        ],
        ["I'm having trouble with {issue} in my automation setup."],
        ["devices dropping off the network randomly", "automations triggering at the wrong time",
         "a sensor giving unreliable readings", "battery-powered devices dying faster than expected",
         "conflicting automations fighting each other", "a hub that won't talk to a specific device brand",
         "security concerns about an internet-exposed device", "local automation breaking when internet is down"],
        ["since a firmware update", "intermittently", "since adding a new device", "since a router change",
         "for one room only", "since switching hubs"],
        [False, True],
    ),
    Domain(
        "security_privacy", "Security & Privacy", False,
        ["User/DPO", "Security Advisor"],
        [
            "You are a data protection officer / privacy-conscious user dealing with {issue}. "
            "Ask genuine follow-up questions.",
            "You are a security/privacy advisor. Give specific, practical guidance about "
            "{issue}. Use web_search for current regulations/best practices when helpful.",
        ],
        ["I need help thinking through {issue}."],
        ["a data breach notification requirement", "unclear data retention policy",
         "third-party vendor access to sensitive data", "a user data deletion request",
         "weak default privacy settings in a product", "cross-border data transfer compliance",
         "an access control gap found in an audit", "consent flow that may not meet current regulations"],
        ["found in an audit", "flagged by a user complaint", "before a compliance deadline",
         "after a near-miss incident", "raised by legal", "since a regulation changed"],
        [False, True],
    ),
    Domain(
        "storage_files", "Storage & Files", False,
        ["Admin", "Storage Engineer"],
        [
            "You are an IT/systems admin dealing with {issue} in your storage systems. Ask "
            "genuine follow-up questions.",
            "You are a storage engineer. Give specific, practical guidance about {issue}. Use "
            "web_search for current tools/best practices when helpful.",
        ],
        ["I'm dealing with {issue} in our storage setup."],
        ["running out of space faster than expected", "backup jobs silently failing",
         "slow file access over the network", "permission issues after a migration",
         "duplicate files bloating storage", "a corrupted file discovered in backups",
         "choosing between cold and hot storage tiers", "sync conflicts between local and cloud copies"],
        ["since last month", "since a migration", "for one file share only", "intermittently",
         "since adding more users", "discovered during an audit"],
        [False, True],
    ),
]

BY_SLUG: dict[str, Domain] = {d.slug: d for d in DOMAINS}

assert len(DOMAINS) == 20, f"expected 20 domains, got {len(DOMAINS)}"
