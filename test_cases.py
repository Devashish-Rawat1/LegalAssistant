"""
test_cases.py

Evaluation harness for the LegalAssistant RAG pipeline.

Runs a fixed set of legal questions with known correct source acts,
and reports for each:
  - which act/section chunks were retrieved
  - whether the EXPECTED act appeared in the retrieved sources
  - the generated answer (for you to eyeball grounding)

This is not an accuracy score — legal answers need human judgement.
It's a regression harness: run it after any change to chunking,
retrieval settings, or prompts, and see whether retrieval got better
or worse in a repeatable way.

Usage:
    python test_cases.py              # run all
    python test_cases.py --quick      # retrieval only, no LLM calls (fast, free)
"""

import sys

from vector_store import get_vector_store
from chains import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage


# Each case: the question, and a substring that SHOULD appear in the
# source metadata of at least one retrieved chunk.
# Keep expected_source lowercase — matching is case-insensitive.
TEST_CASES = [
    {
        "id": "theft-general",
        "question": "What is the punishment for theft?",
        "expected_source": "nyaya",          # BNS 2023
        "note": "Core criminal offence. Should cite BNS, NOT the repealed IPC.",
    },
    {
        "id": "murder",
        "question": "What is the punishment for murder?",
        "expected_source": "nyaya",
        "note": "Should retrieve the BNS murder provision.",
    },
    {
        "id": "fundamental-rights",
        "question": "What are the fundamental rights guaranteed by the Constitution?",
        "expected_source": "constitution",
        "note": "Should retrieve Part III of the Constitution.",
    },
    {
        "id": "consumer-complaint",
        "question": "How do I file a consumer complaint and what is the time limit?",
        "expected_source": "consumer",
        "note": "Should retrieve Consumer Protection Act, 2019.",
    },
    {
        "id": "drunk-driving",
        "question": "What is the penalty for driving under the influence of alcohol?",
        "expected_source": "motor",
        "note": "Should retrieve Motor Vehicles Act, 1988.",
    },
    {
        "id": "identity-theft-online",
        "question": "What is the punishment for identity theft using a computer?",
        "expected_source": "information technology",
        "note": "Should retrieve IT Act 2000 (s.66C), NOT general BNS theft.",
    },
    {
        "id": "workplace-harassment",
        "question": "What is an Internal Complaints Committee and who must set one up?",
        "expected_source": "sexual harassment",
        "note": "Should retrieve the Workplace Harassment Act, 2013.",
    },
    {
        "id": "evidence-admissibility",
        "question": "When is electronic evidence admissible in court?",
        "expected_source": "sakshya",       # Bharatiya Sakshya Adhiniyam
        "note": "Should retrieve BSA 2023 (replaced the Evidence Act).",
    },
    {
        "id": "arrest-procedure",
        "question": "What is the procedure for arrest without a warrant?",
        "expected_source": "suraksha",      # BNSS
        "note": "Should retrieve BNSS 2023 (replaced the CrPC).",
    },
    {
        "id": "out-of-scope",
        "question": "What are the tax slabs for income tax in India?",
        "expected_source": None,            # nothing in corpus covers this
        "note": "NEGATIVE TEST: no income tax act indexed. The assistant "
                "should say it cannot answer, NOT invent tax rates.",
    },
]

# Multi-turn case, tested separately since it needs chat history
FOLLOWUP_CASE = {
    "id": "followup-context",
    "first": "What is the punishment for theft?",
    "followup": "What if it is a repeat offence?",
    "note": "Tests history-aware retrieval: the follow-up has no subject "
            "of its own, so it must be rewritten using the first question.",
}


def check_sources(retrieved_docs, expected_source):
    """Returns (passed, list_of_source_labels)."""
    labels = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        labels.append(f"{source} {section}".strip())

    if expected_source is None:
        # Negative test — we can't judge by sources, only by the answer text
        return None, labels

    joined = " ".join(labels).lower()
    return (expected_source.lower() in joined), labels


def run_retrieval_only(vector_store):
    """Fast, free check: retrieval quality without calling the LLM."""
    print("=" * 70)
    print("RETRIEVAL-ONLY MODE (no LLM calls)")
    print("=" * 70)

    passed = failed = skipped = 0

    for case in TEST_CASES:
        docs = vector_store.similarity_search(case["question"], k=4)
        ok, labels = check_sources(docs, case["expected_source"])

        if ok is None:
            status = "SKIP (negative test — needs LLM)"
            skipped += 1
        elif ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"\n[{status}] {case['id']}")
        print(f"  Q: {case['question']}")
        if case["expected_source"]:
            print(f"  Expected source to contain: {case['expected_source']!r}")
        print(f"  Retrieved:")
        for label in labels:
            print(f"    - {label}")

    print(f"\n{'=' * 70}")
    print(f"Retrieval: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)


def run_full(vector_store):
    """Full run: retrieval + generation, so you can judge answer grounding."""
    rag_chain = get_rag_chain(vector_store)

    print("=" * 70)
    print("FULL MODE (retrieval + LLM generation)")
    print("=" * 70)

    passed = failed = 0

    for case in TEST_CASES:
        response = rag_chain.invoke({
            "input": case["question"],
            "chat_history": [],
        })

        ok, labels = check_sources(response["context"], case["expected_source"])

        if ok is None:
            status = "REVIEW"
        elif ok:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"\n{'-' * 70}")
        print(f"[{status}] {case['id']}")
        print(f"  Q: {case['question']}")
        print(f"  Note: {case['note']}")
        print(f"  Retrieved:")
        for label in labels:
            print(f"    - {label}")
        print(f"\n  Answer:\n{response['answer']}\n")

    # --- Multi-turn follow-up test ---
    print(f"\n{'=' * 70}")
    print(f"[REVIEW] {FOLLOWUP_CASE['id']}")
    print(f"  Note: {FOLLOWUP_CASE['note']}")
    print("=" * 70)

    r1 = rag_chain.invoke({"input": FOLLOWUP_CASE["first"], "chat_history": []})
    history = [
        HumanMessage(content=FOLLOWUP_CASE["first"]),
        AIMessage(content=r1["answer"]),
    ]
    r2 = rag_chain.invoke({"input": FOLLOWUP_CASE["followup"], "chat_history": history})

    print(f"\n  Q1: {FOLLOWUP_CASE['first']}")
    print(f"  Q2: {FOLLOWUP_CASE['followup']}")
    print(f"\n  Retrieved for follow-up:")
    for doc in r2["context"]:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        print(f"    - {source} {section}")
    print(f"\n  Follow-up answer:\n{r2['answer']}\n")

    print("=" * 70)
    print(f"Scored cases: {passed} passed, {failed} failed")
    print("Manually review: does each answer cite the correct ACT NAME,")
    print("and do the cited section numbers match the retrieved metadata?")
    print("=" * 70)


def main():
    quick = "--quick" in sys.argv
    vector_store = get_vector_store()

    if quick:
        run_retrieval_only(vector_store)
    else:
        run_full(vector_store)


if __name__ == "__main__":
    main()
    