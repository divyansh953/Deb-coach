import argparse
import json
import os

from agents.argue_node import argue_node
from agents.coach_node import coach_node
from agents.english_coach_node import english_coach_node
from agents.english_quiz_node import english_quiz_node
from agents.format_node import format_node
from agents.night_agent import night_agent_node
from agents.summarize_node import summarize_node
from agents.topic_selector import choose_topic_for_today
from agents.weekend_agent import weekend_agent_node
from core.network_utils import clear_broken_local_proxies
from core.topic_utils import topic_name
from delivery.telegram import send_digest
from delivery.website import send_to_website
from evals.rag.fixtures import RETRIEVAL_EVAL_CASES
from evals.rag.real_trace_eval import evaluate_recent_traces
from evals.rag.retrieval_eval import evaluate_query_plan, evaluate_structured_evidence
from graph import build_daily_graph, build_night_graph, build_weekend_graph
from rag.metadata import build_metadata
from rag.query_planner import build_query_plan


_CLEARED_PROXIES = clear_broken_local_proxies()
if _CLEARED_PROXIES:
    print(f"[Network] Cleared broken proxy env vars: {', '.join(_CLEARED_PROXIES)}")


def _load_topics_config() -> tuple[list[str], dict]:
    with open("topics.json", "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        topics = [item if isinstance(item, str) else item.get("topic", "") for item in payload]
        return [t for t in topics if t], {}

    raw_topics = payload.get("priority_topics", [])
    topics: list[str] = []
    for entry in raw_topics:
        if isinstance(entry, str):
            topics.append(entry)
        elif isinstance(entry, dict):
            name = entry.get("topic")
            if name:
                topics.append(name)

    metadata = {
        "study_scope": payload.get("study_scope", ""),
        "selection_lens": payload.get("selection_lens", {}),
        # Preserve the structured priority_topics for nodes that want frame guidance later.
        "priority_topics": raw_topics,
    }
    return topics, metadata


def _initial_state(topic: str | dict) -> dict:
    normalized_topic = topic_name(topic)
    return {
        "topic": normalized_topic,
        "selector_reason": "",
        "topic_info": {},
        "topic_foundation": {},
        "article_context": {},
        "topic_motion_set": {},
        "motion_intelligence": {},
        "drafted_motion": {},
        "debate_teaching": {},
        "vocabulary_output": {},
        "final_sections": {},
        "candidate_articles": [],
        "lead_case": {},
        "lead_case_reason": "",
        "raw_articles": [],
        "reference_background": "",
        "article_background": "",
        "enriched_context": "",
        "preknowledge_notes": [],
        "article_context_notes": [],
        "case_deep_dive": [],
        "vocab_candidates": [],
        "vocab_context_notes": [],
        "topic_motion_set": {},
        "motion_intelligence": {},
        "ranked_articles": [],
        "summaries": [],
        "key_facts": [],
        "concepts": [],
        "arguments": {},
        "debate_angle": "",
        "debate_packet": {},
        "english_lesson": "",
        "vocab_words": [],
        "word_roots": [],
        "final_doc": "",
        "task_type": "fetch",
        "article_length": 0,
        "studied_today": None,
        "quiz_score": None,
        "retrieval_plans": {},
        "retrieval_traces": {},
    }


def _lookup_topic_info(topic_override: str, priority_entries: list[str | dict]) -> dict:
    normalized = topic_name(topic_override)
    normalized_terms = set(normalized.split())
    best_match: dict = {}
    best_score = -1

    for entry in priority_entries:
        if not isinstance(entry, dict):
            continue
        candidate = topic_name(entry.get("topic", ""))
        if not candidate:
            continue
        candidate_terms = set(candidate.split())
        score = len(normalized_terms & candidate_terms)
        if normalized == candidate or normalized in candidate or candidate in normalized:
            score += 5
        if score > best_score:
            best_score = score
            best_match = entry

    return best_match if best_score > 0 else {}


def run_daily(topic_override: str | None = None):
    topics, topics_metadata = _load_topics_config()

    priority_entries = topics_metadata.get("priority_topics", topics)

    if topic_override:
        matched_info = _lookup_topic_info(topic_override, priority_entries)
        selection = {
            "topic": topic_override,
            "reason": "Manual topic override.",
            "topic_info": matched_info,
        }
    else:
        selection = choose_topic_for_today(priority_entries)

    graph = build_daily_graph()

    if topics_metadata:
        print("Study scope loaded.")
        print(topics_metadata.get("study_scope", ""))

    normalized_topic = topic_name(selection["topic"])
    print(f"\n{'=' * 40}")
    print(f"Processing topic: {normalized_topic}")
    print(f"Reason: {selection.get('reason', '')}")
    print(f"{'=' * 40}")
    state = _initial_state(normalized_topic)
    state["selector_reason"] = selection.get("reason", "")
    state["topic_info"] = selection.get("topic_info", {}) or {}
    result = graph.invoke(state)
    print(f"Raw articles: {len(result['raw_articles'])}")
    print(f"Ranked articles: {len(result['ranked_articles'])}")
    print(f"Summaries: {len(result['summaries'])}")
    print(f"Arguments FOR: {len(result['arguments'].get('for', []))}")
    print(f"Enriched context chars: {len(result['enriched_context'])}")
    print(f"Final doc chars: {len(result['final_doc'])}")
    send_digest(result["final_doc"])
    # Best-effort mirror to the debate website; a site outage must never
    # break the Telegram delivery that already succeeded above.
    try:
        send_to_website(result["final_doc"])
    except Exception as exc:
        print(f"[Website] Delivery skipped: {exc}")


def run_summarize_smoke():
    state = _initial_state("feminism")
    state["ranked_articles"] = [
        {
            "title": "Sample article",
            "url": "https://example.com",
            "content": "Feminism debates often focus on equality, representation, and policy impact. "
            "Recent coverage discusses economic participation and legal safeguards. "
            "Debaters can use this to compare principle and outcome based framing.",
            "source": "sample",
            "published": "",
        }
    ]
    state["enriched_context"] = "Sample background context."
    result = summarize_node(state)
    print(f"Summaries: {len(result['summaries'])}")
    print(f"Key facts: {len(result['key_facts'])}")
    print(f"Concepts: {len(result['concepts'])}")


def run_argue_smoke():
    state = _initial_state("feminism")
    state["summaries"] = [
        "- Equality claims often hinge on access to opportunity.\n- Economic participation shapes autonomy."
    ]
    result = argue_node(state)
    print(f"FOR arguments: {len(result['arguments'].get('for', []))}")
    print(f"AGAINST arguments: {len(result['arguments'].get('against', []))}")
    print(f"Middle present: {bool(result['arguments'].get('middle'))}")


def run_coach_smoke():
    state = _initial_state("feminism")
    state["summaries"] = [
        "- Equality debates turn on access to opportunity.\n- Economic autonomy strengthens broader freedom."
    ]
    state["arguments"] = {
        "for": ["Feminism expands access and voice.", "It reduces structural barriers.", "It improves autonomy."],
        "against": ["Implementation can create backlash.", "Bad policy design can misfire.", "Tradeoffs still matter."],
        "middle": "Support the principle but scrutinize mechanisms.",
    }
    result = coach_node(state)
    print(f"Coaching chars: {len(result['debate_angle'])}")
    print(f"Contains UNIQUE ANGLE: {'UNIQUE ANGLE' in result['debate_angle']}")


def run_english_smoke():
    state = _initial_state("geopolitics")
    result = english_coach_node(state)
    print(f"English lesson chars: {len(result['english_lesson'])}")
    print(f"Vocab words: {len(result['vocab_words'])}")


def run_english_quiz():
    state = _initial_state(os.getenv("ENGLISH_QUIZ_TOPIC", "vocabulary roots etymology"))
    result = english_quiz_node(state)
    print(f"English quiz score: {result.get('english_quiz_score')}")
    print(f"Question count: {len(result.get('english_quiz_questions', []))}")


def run_english_quiz_smoke():
    os.environ["DEV_MODE"] = "true"
    state = _initial_state("feminism")
    result = english_quiz_node(state)
    questions = result.get("english_quiz_questions", [])
    print(f"Questions generated: {len(questions)}")
    if questions:
        print(f"First question: {questions[0].get('question', '')[:100]}")
    print(f"Score (dev reply was 'timeout'): {result.get('english_quiz_score')}")


def run_format_smoke():
    state = _initial_state("feminism")
    state["enriched_context"] = "Sample background context for the topic."
    state["ranked_articles"] = [
        {"title": "Sample article one", "url": "", "content": "", "source": "sample", "published": ""},
        {"title": "Sample article two", "url": "", "content": "", "source": "sample", "published": ""},
    ]
    state["summaries"] = ["- Equality and access matter.\n- Framing determines the round."]
    state["arguments"] = {
        "for": ["It expands autonomy.", "It improves access.", "It changes institutions."],
        "against": ["It can trigger backlash.", "Bad design has costs.", "Tradeoffs matter."],
        "middle": "Support the principle, scrutinize the mechanism.",
    }
    state["debate_angle"] = "UNIQUE ANGLE: Win on implementation quality."
    state["english_lesson"] = "ENGLISH POWER\nWord set: lucid, precise, nuance"
    state["vocab_words"] = ["lucid", "precise", "nuance"]
    state["word_roots"] = ["dict", "cred"]
    state["key_facts"] = ["Participation rates shape autonomy."]
    state["concepts"] = ["Structural inequality"]
    state["topic_info"] = {
        "why_this_matters_for_debate": "Gender rounds are won by debaters who distinguish theory, mechanism, and implementation.",
        "key_concepts_own_these_precisely": [
            "Patriarchy means structured power, not just individual prejudice.",
            "Intersectionality tracks how multiple structures of disadvantage overlap.",
        ],
        "essential_theoretical_frameworks": [
            "Liberal feminism focuses on equal rights within institutions.",
        ],
        "argument_angles_most_debaters_miss": [
            "The best feminist case is often about institutional design rather than moral outrage.",
        ],
    }
    result = format_node(state)
    print(f"Final doc chars: {len(result['final_doc'])}")
    print(f"Contains topic banner: {'TOPIC FOR TODAY' in result['final_doc']}")


def run_night_smoke():
    os.environ["DEV_MODE"] = "true"
    state = _initial_state("night")
    result = night_agent_node(state)
    print(f"Studied today: {result['studied_today']}")
    print(f"Quiz score: {result['quiz_score']}")


def run_night():
    graph = build_night_graph()
    result = graph.invoke(_initial_state("night"))
    print(f"Studied today: {result['studied_today']}")
    print(f"Quiz score: {result['quiz_score']}")


def run_weekend():
    graph = build_weekend_graph()
    result = graph.invoke(_initial_state("weekend"))
    print(f"Weekend output chars: {len(result['final_doc'])}")


def run_weekend_smoke():
    state = _initial_state("weekend")
    result = weekend_agent_node(state)
    print(f"Weekend output chars: {len(result['final_doc'])}")
    print(f"Contains weekly header: {'WEEKLY BRAIN UPLOAD' in result['final_doc']}")


def run_retrieval_eval():
    print("Running retrieval evaluation harness...")
    print("")

    total_scores: list[float] = []
    for case in RETRIEVAL_EVAL_CASES:
        plan = build_query_plan(case["node_name"], case["state"])
        score = evaluate_query_plan(plan, case["expected"])
        total_scores.append(score["total_score"])
        print(f"[{case['name']}] total={score['total_score']:.3f}")
        print(
            "  "
            f"stores={score['store_score']:.3f} "
            f"terms={score['term_score']:.3f} "
            f"utility={score['utility_score']:.3f} "
            f"classes={score['class_score']:.3f}"
        )

    definition_meta = build_metadata(
        "wikipedia",
        "https://example.com/wiki",
        {"url": "https://example.com/wiki", "title": "Sovereignty"},
    )
    mechanism_meta = build_metadata(
        "debate_theory",
        "knowledge_base/pdfs/debate.pdf",
        {"source_path": "knowledge_base/pdfs/debate.pdf"},
    )
    example_meta = build_metadata(
        "news",
        "https://example.com/news",
        {"url": "https://example.com/news", "title": "Ukraine case"},
    )
    sample_chunks = {
        "knowledge_db": [
            {"page_content": "Sovereignty refers to supreme authority within a territory.", "metadata": definition_meta},
            {"page_content": "Ukraine creates a live example of sovereignty pressure.", "metadata": example_meta},
        ],
        "reasoning_db": [
            {"page_content": "The security dilemma explains why deterrence can produce escalation.", "metadata": mechanism_meta},
        ],
    }
    evidence_score = evaluate_structured_evidence(
        sample_chunks,
        expected_sections=["definitions", "mechanisms", "examples"],
    )
    print("")
    print(
        "[structured_evidence] "
        f"section_score={evidence_score['section_score']:.3f} "
        f"present={', '.join(evidence_score['present_sections'])}"
    )
    print("")
    average = sum(total_scores) / len(total_scores) if total_scores else 0.0
    print(f"Average planner score: {average:.3f}")


def run_real_retrieval_eval():
    report = evaluate_recent_traces(limit_days=7)
    print("Running real retrieval trace evaluation...")
    print("")
    print(f"Days scanned: {report['days_scanned']}")
    print(f"Lessons scored: {report['lessons_scored']}")
    print(f"Average score: {report['average_score']:.3f}")
    print(f"Node coverage: {report['node_coverage']}")
    print(f"Score bands: {report['score_bands']}")
    if report["low_scoring_topics"]:
        print("Low-scoring topics:")
        for topic in report["low_scoring_topics"][:5]:
            print(f"  - {topic}")
    if report["repeated_sources"]:
        print("Repeated weak source refs:")
        for source, count in report["repeated_sources"]:
            print(f"  - {source} ({count})")
    print("")
    for lesson in report["lesson_scores"][:5]:
        print(
            f"[{lesson['date']}] {lesson['topic']} total={lesson['total_score']:.3f} "
            f"memory={lesson['memory_present']:.3f} "
            f"sources={lesson['source_diversity_score']:.3f} "
            f"terms={lesson['term_coverage_score']:.3f} "
            f"vocab={lesson['vocab_score']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily", "night", "weekend", "summarize-smoke", "argue-smoke", "coach-smoke", "english-smoke", "english-quiz", "english-quiz-smoke", "format-smoke", "night-smoke", "weekend-smoke", "retrieval-eval", "real-retrieval-eval"], required=True)
    parser.add_argument("--topic", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "daily":
        run_daily(args.topic)
    elif args.mode == "night":
        run_night()
    elif args.mode == "weekend":
        run_weekend()
    elif args.mode == "summarize-smoke":
        run_summarize_smoke()
    elif args.mode == "argue-smoke":
        run_argue_smoke()
    elif args.mode == "coach-smoke":
        run_coach_smoke()
    elif args.mode == "english-smoke":
        run_english_smoke()
    elif args.mode == "english-quiz":
        run_english_quiz()
    elif args.mode == "english-quiz-smoke":
        run_english_quiz_smoke()
    elif args.mode == "format-smoke":
        run_format_smoke()
    elif args.mode == "night-smoke":
        run_night_smoke()
    elif args.mode == "weekend-smoke":
        run_weekend_smoke()
    elif args.mode == "retrieval-eval":
        run_retrieval_eval()
    elif args.mode == "real-retrieval-eval":
        run_real_retrieval_eval()
