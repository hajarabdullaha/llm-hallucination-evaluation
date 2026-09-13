# LLM Hallucination Evaluation

Comparing two open-source instruction-tuned LLMs on SQuAD 2.0 — how well they answer questions when the context actually has the answer, and whether they make things up when it doesn't.

## Objective

Most QA benchmarks only test whether a model can find the right answer. This project also tests the opposite case: what happens when the context genuinely doesn't contain the answer. A good model should recognize that and refuse rather than guess. A model that hallucinates a confident-sounding answer here is arguably worse than one that gives a wrong answer to an answerable question, since there's no signal to the user that anything went wrong.

## Dataset

[SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) validation split. SQuAD 2.0 is built on top of the original SQuAD, with the addition of unanswerable questions written to look plausible but not be supported by the given passage.

From the validation set, 100 answerable and 100 unanswerable questions were sampled (seeded, `random.seed(42)`), for a balanced 200-example evaluation set.

## Models

- [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (gated — requires accepting the license on the Hub and authenticating)

Both were run with the same frozen prompt template and greedy decoding (`do_sample=False`), so any differences come from the models themselves rather than prompting or sampling variance.

## Methodology

1. Load SQuAD 2.0 validation, split into answerable / unanswerable, sample 100 of each.
2. For each example, prompt the model with the context + question and ask it to answer only from the context, or say `INSUFFICIENT_CONTEXT` if the context doesn't support an answer.
3. Run both models on the same 200 examples, recording the generated answer and latency.
4. Score answerable examples with Exact Match, Token F1, and semantic similarity (cosine similarity between sentence embeddings of the prediction and the gold answer).
5. Score unanswerable examples by checking whether the model's response contains `INSUFFICIENT_CONTEXT`. A correct refusal counts toward refusal accuracy; anything else — including an answer that happens to be factually correct — counts as a hallucination, since the model wasn't actually grounded in the given context.
6. Pull out specific cases where one model refused and the other didn't, to look at qualitatively.

## Metrics

- **Exact Match** — normalized string match against the gold answer (answerable questions only)
- **Token F1** — token-overlap F1 against the gold answer (answerable questions only)
- **Semantic Similarity** — cosine similarity between sentence embeddings of prediction and gold answer (answerable questions only)
- **Refusal Accuracy** — % of unanswerable questions where the model correctly said it didn't have enough context
- **Hallucination Rate** — 1 − refusal accuracy, on unanswerable questions
- **Average Latency** — average generation time per example

## Final Results

| Model | Exact Match | Token F1 | Semantic Similarity | Refusal Accuracy | Hallucination Rate | Avg Latency |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 30.0% | 53.7% | 67.9% | 28.0% | 72.0% | 0.59 s |
| Mistral-7B-Instruct-v0.3 | 40.0% | 64.6% | 75.3% | 56.0% | 44.0% | 4.69 s |

**Qualitative comparison** (unanswerable questions where one model refused and the other didn't):
- Qwen hallucinated, Mistral correctly refused: 23 cases
- Mistral hallucinated, Qwen correctly refused: 10 cases

## Key Findings

- Mistral is stronger across every answer-quality metric — Exact Match, Token F1, and semantic similarity are all higher than Qwen's.
- The biggest gap isn't on answerable questions, it's on knowing when *not* to answer. Mistral refused correctly more than twice as often (56% vs 28%), which more than halves its hallucination rate compared to Qwen's.
- Qwen is roughly 8x faster per example (0.59s vs 4.69s), which lines up with it being a much smaller model (1.5B vs 7B parameters). There's a real efficiency-vs-reliability trade-off here, not just a "Mistral is better" story.
- The qualitative cases back up the aggregate numbers — Qwen hallucinated more than twice as often as Mistral did in these head-to-head refusal comparisons.

## Limitations

- Evaluated on a 200-example sample of SQuAD 2.0, not the full dataset.
- Only two models were compared, and they're not close in size (1.5B vs 7B), so this isn't a clean architecture-vs-architecture comparison.
- Latency numbers reflect this specific Colab runtime and model-loading setup — not a controlled hardware benchmark.
- Hallucination here is defined strictly as "answered when it should have refused." A model could still hallucinate details within a correct-in-spirit answer to an answerable question, and this setup doesn't capture that.
- Exact Match penalizes valid paraphrases, which is part of why Token F1 and semantic similarity are reported alongside it.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `notebooks/llm_hallucination_evaluation.ipynb` in Jupyter or Colab.
3. Mistral-7B-Instruct-v0.3 is gated on Hugging Face — accept the license on the [model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and log in with `huggingface-cli login` (or the Colab auth cell) before running that section.
4. Run all cells top to bottom. GPU is strongly recommended, especially for the Mistral run.

## Project Structure

```
llm-hallucination-evaluation/
├── README.md
├── notebooks/
│   └── llm_hallucination_evaluation.ipynb
├── figures/
│   ├── hallucination_rate.png
│   ├── f1_scores.png
│   └── semantic_similarity.png
├── results/
│   └── final_model_comparison.csv
├── requirements.txt
└── .gitignore
```

Note: a latency comparison chart is generated by the notebook but wasn't included in this repo since the file wasn't available locally when this was packaged up.
