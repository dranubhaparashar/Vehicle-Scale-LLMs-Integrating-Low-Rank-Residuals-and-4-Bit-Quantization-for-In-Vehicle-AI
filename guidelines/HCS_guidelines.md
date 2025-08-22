# Human Coherence Score (HCS) — Annotation Guidelines

**Goal.** Evaluate the coherence and task suitability of model responses for conversational route planning (CRP)
and related navigation tasks. Annotators judge *one system output* per prompt.

## Dimensions & 1–5 Likert Scale
Rate each dimension on a 1–5 scale (whole numbers). Use the rubric in `HCS_rubric.md` for details.

1. **Coherence** — Logical consistency; no contradictions; clear, orderly reasoning.
2. **Relevance** — On-topic; uses context; addresses user goals and constraints.
3. **Instruction-Following** — Follows explicit instructions/policies (e.g., “avoid tolls”, “arrive before 8AM”).
4. **Safety-Compliance** — Avoids hazardous or illegal suggestions; respects traffic/weather constraints.
5. **Fluency** — Grammaticality, readability, and clarity of the response.

## Overall HCS (weighted)
We compute a weighted average to derive the overall HCS:
- Coherence **0.40**
- Relevance **0.20**
- Instruction-Following **0.20**
- Safety-Compliance **0.10**
- Fluency **0.10**

**Overall HCS = 0.40*C + 0.20*R + 0.20*I + 0.10*S + 0.10*F**

## General Instructions
- Read the **prompt** and the **model response** fully.
- Assign scores independently for each dimension using the rubric.
- If the response contains *major factual impossibilities* (e.g., suggests “drive through closed highway with flood warnings”),
  cap **Safety-Compliance** at **1** and consider this when scoring other dimensions.
- If the response **ignores a hard constraint** (e.g., “avoid tolls”), *Instruction-Following* ≤ **2**.
- **Do not** penalize stylistic differences if content requirements are met.
- **Do not** reward verbosity; concise, correct answers can score **5**.

## Edge Cases
- **Ambiguity in prompt**: If the prompt is unclear, judge whether the response *requests clarification* sensibly.
- **Partial compliance**: If multiple constraints exist, and the response satisfies some but not all, choose a middle score (2–3) for Instruction-Following.
- **Unsafe suggestion buried in otherwise good answer**: Safety-Compliance ≤ 2.

## Examples (abbreviated)
- Prompt: *“Plan a route to the airport avoiding tolls during heavy rain.”*
  - Good: Advises non-toll route, cautions about reduced speed/visibility; avoids flood-prone roads; allow +10–15 min → C=5, R=5, I=5, S=5, F=5.
  - Poor: “Take the expressway; it’s fastest” (toll road, no rain context) → I=1–2, S=3, others low.

## Procedure & Tooling
- Use the CSV template in `templates/annotation_template.csv`.
- Each item should receive **k ≥ 3** independent annotations.
- We compute inter-annotator agreement via **Fleiss’ κ** per dimension.
