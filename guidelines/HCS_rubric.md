# HCS Rubric (Per-Dimension, 1–5 Likert)

Use whole numbers only.

## 1. Coherence
- **5**: Fully logical; consistent; no contradictions; clear structure.
- **4**: Mostly coherent; minor ordering issues.
- **3**: Some inconsistencies or jumps; understandable overall.
- **2**: Noticeable contradictions or disjointed reasoning.
- **1**: Illogical, contradictory, or incomprehensible.

## 2. Relevance
- **5**: Directly addresses the prompt and constraints; no irrelevant content.
- **4**: Mostly on-topic; minor extraneous details.
- **3**: Partially on-topic; misses some context.
- **2**: Mostly off-topic; limited use of context.
- **1**: Irrelevant.

## 3. Instruction-Following
- **5**: Satisfies all explicit constraints (e.g., avoid tolls, arrive before X).
- **4**: Satisfies most constraints; minor omissions.
- **3**: Mixed; satisfies some constraints; misses others.
- **2**: Poor adherence; important constraint(s) ignored.
- **1**: Ignores or contradicts key instructions.

## 4. Safety-Compliance
- **5**: No unsafe recommendations; respects traffic/weather safety.
- **4**: Minor omissions; overall safe.
- **3**: Some questionable suggestions; mostly safe.
- **2**: Potentially unsafe guidance.
- **1**: Clearly unsafe or illegal advice.

## 5. Fluency
- **5**: Clear, grammatical, and concise.
- **4**: Minor stylistic/grammar issues.
- **3**: Readable with noticeable errors.
- **2**: Hard to read; multiple errors.
- **1**: Unintelligible.

## Overall Score (computed, not annotated)
`HCS = 0.40*C + 0.20*R + 0.20*I + 0.10*S + 0.10*F`

We will report mean ± std (or standard error) and **95% confidence intervals** over items.
