import sys
import json
sys.path.append(".")
import warnings
warnings.filterwarnings("ignore")

from src.rag_chain import ask

print("=== Sprint 5: Evaluation ===\n")

with open("eval/test_qa.json") as f:
    test_cases = json.load(f)

results = []
print(f"Running RAG on {len(test_cases)} questions...\n")

for i, tc in enumerate(test_cases):
    print(f"[{i+1}/{len(test_cases)}] {tc['question'][:50]}...")
    try:
        answer, sources = ask(tc["question"])
        results.append({
            "question": tc["question"],
            "ground_truth": tc["ground_truth"],
            "rag_answer": answer,
            "sources": sources,
            "status": "ok"
        })
    except Exception as e:
        print(f"  Error: {e}")
        results.append({
            "question": tc["question"],
            "ground_truth": tc["ground_truth"],
            "rag_answer": "",
            "sources": [],
            "status": "error"
        })

print("\nComputing ROUGE-L scores...")
try:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for r in results:
        if r["status"] == "ok" and r["rag_answer"]:
            s = scorer.score(r["ground_truth"], r["rag_answer"])
            scores.append(s['rougeL'].fmeasure)
        else:
            scores.append(0.0)
    avg_rouge = sum(scores) / len(scores)
    print(f"\n✓ Average ROUGE-L Score: {avg_rouge:.3f}")
    print(f"✓ Questions answered: {sum(1 for r in results if r['status'] == 'ok')}/{len(results)}")
except ImportError:
    print("rouge_score not installed. Installing...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "rouge-score"])
    avg_rouge = 0.0

output = {
    "rouge_l": round(avg_rouge, 3),
    "n_questions": len(results),
    "answered": sum(1 for r in results if r["status"] == "ok"),
    "results": results
}

with open("eval/results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to eval/results.json")
print("\n=== Sample Answers ===")
for r in results[:3]:
    print(f"\nQ: {r['question']}")
    print(f"Ground Truth: {r['ground_truth']}")
    print(f"RAG Answer: {r['rag_answer'][:150]}...")
    print("─" * 50)