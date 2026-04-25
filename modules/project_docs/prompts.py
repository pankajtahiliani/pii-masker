"""
Agile artifact prompts and token budgets.
Module-level constants — no longer duplicated across generate/refine functions.
"""

# ── Per-artifact focused prompts (small = fast) ────────────────────────────────
_ARTIFACT_PROMPTS = {
    'backlog': """\
Role: Senior Scrum Master (15y exp).
Task: Generate JSON Product Backlog for the project in source.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"epics":[{{"id":"EP-01","title":"Epic title","description":"What this epic covers","stories":[{{"id":"US-01-01","title":"As a [role] I want [goal] so that [benefit]","story_points":3,"priority":"Must Have","acceptance_criteria":["Criterion 1","Criterion 2","Criterion 3"]}}]}}]}}
Constraints: 5 epics. 4 stories per epic. Points: 1/2/3/5/8/13. Priority: Must Have/Should Have/Could Have/Won't Have. 2-3 AC per story.
Focus: Derive every epic and story directly from features, integrations, user roles, and performance targets in source. No generic items.
Project info: {src}""",

    'sprint_plan': """\
Role: Senior Scrum Master (15y exp).
Task: Generate JSON Sprint Plan for the project in source. Velocity 28 pts, 2-week sprints.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"duration_weeks":2,"team_velocity":28,"sprints":[{{"number":1,"goal":"Specific sprint goal tied to source deliverables","stories":["US-01-01","US-01-02","US-02-01"],"total_points":27,"deliverable":"What is demo-ready at end of sprint","risks":"Top risk for this sprint"}}]}}
Constraints: 4-5 sprints covering all stories. Each sprint: specific goal, 5-6 story IDs, points near velocity, one demo deliverable, one risk. Reference actual source deliverables.
Project info: {src}""",

    'sprint_review': """\
Role: Senior Scrum Master (15y exp).
Task: Generate JSON Sprint Review templates for the project in source.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"reviews":[{{"sprint":1,"planned_stories":["US-01-01","US-01-02"],"completed_stories":["US-01-01"],"incomplete_stories":["US-01-02"],"demo_notes":"What to demo and how, naming actual features","stakeholder_feedback":"Specific questions for stakeholders named in source","next_sprint_adjustments":"Concrete changes based on this sprint outcome","velocity_actual":24,"velocity_planned":28}}]}}
Constraints: One review per sprint (3-4 reviews). Each: planned vs completed, specific demo notes naming features, stakeholder questions referencing source, concrete next-sprint adjustments.
Project info: {src}""",

    'retrospective': """\
Role: Senior Scrum Master (15y exp).
Task: Generate JSON Start-Stop-Continue Retrospective for the project in source.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"went_well":["Item"],"improve":["Item"],"action_items":[{{"action":"Specific action","owner":"Role","due":"Sprint N"}}],"team_health_check":["Dimension: rating and note"],"process_improvements":["Improvement"]}}
Constraints: went_well: 5 specific items. improve: 5 specific items. action_items: 4-5 with owner and due sprint. team_health_check: 4 dimensions with rating. process_improvements: 3-4 items.
Focus: Reference actual technical challenges, integration risks, stakeholder dynamics, and team constraints from source. No generic Agile platitudes.
Project info: {src}""",

    'risk_register': """\
Role: Senior Scrum Master (15y exp).
Task: Generate JSON Risk Register for the project in source.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"risks":[{{"id":"R-01","category":"Scope Creep","title":"Brief risk title","description":"One specific sentence from source.","probability":"High","impact":"High","severity":"Critical","mitigation":"One concrete sentence.","owner":"Role","sprint_impact":"Sprint 1-2"}}]}}
Constraints: 6 risks. Cover categories: Scope Creep, Resource Availability, Technical Integration, Security/Compliance, Timeline/Delivery, Team Capacity. severity: Critical/High/Medium/Low. probability/impact: High/Medium/Low. Values from source only.
Project info: {src}""",

    'test_cases': """\
Role: Senior QA Lead (15y exp).
Task: Generate JSON Test Suite for the project in source.
Output: ONLY valid JSON. No markdown, no prose.
Structure: {{"test_cases":[{{"id":"TC-01","title":"Test title","type":"Unit","feature":"Feature under test","story_id":"US-01-01","steps":["Step 1","Step 2","Step 3","Step 4"],"expected_result":"Specific expected outcome","priority":"High"}}]}}
Constraints: 10 test cases. type: Unit/Integration/UAT (at least 3 of each). 3-4 steps per test. Cover core features, API integrations, edge cases, and security requirements from source.
Project info: {src}""",
}

# ── Per-artifact max_tokens budget (module-level — no longer duplicated) ───────
# Sized for qwen2.5:7b at ~10 tok/s on M4 Pro.
_ARTIFACT_PREDICT = {
    'backlog':        1500,   # 5E × 4S + AC ≈ 1400 tok
    'sprint_plan':     900,   # 4-5 sprints ≈ 850 tok
    'sprint_review':  1000,   # 4 reviews ≈ 950 tok
    'retrospective':   800,   # 4 sections × 5 items ≈ 750 tok
    'risk_register':  1100,   # 8 risks × 7 fields ≈ 1050 tok
    'test_cases':     1300,   # 10 cases × 4 steps ≈ 1250 tok
}

# ── Array keys for deep-truncation object extraction ─────────────────────────
_ARTIFACT_ARRAY_KEYS = {
    'backlog':        'epics',
    'sprint_plan':    'sprints',
    'sprint_review':  'reviews',
    'retrospective':  None,
    'risk_register':  'risks',
    'test_cases':     'test_cases',
}
