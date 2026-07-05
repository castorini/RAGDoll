from __future__ import annotations

import re

ARENA_JUDGE_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER = """Please act as a careful human Search Arena voter. Read the user's question and the two assistant answers, then choose the answer you would rather receive as the user.

Do not use a generic short-answer rubric. First infer what the user is trying to get done, then judge the answer by the dimensions that matter for that request. In search-result comparisons, a strong answer often wins because it gives more useful answer content, not because it is shorter or more polished.

Consider a broad set of possible dimensions:
- Intent match: answers the exact question, language, location, time frame, and requested scope.
- Answer density: contains many relevant, non-duplicative pieces of useful information.
- Coverage and recall: includes the important options, entities, subquestions, examples, caveats, and perspectives the user likely needs.
- Specificity: gives concrete names, dates, numbers, prices, addresses, source names, mechanisms, or distinctions when useful.
- Explanation quality: for "how", "why", effects, tradeoffs, or research questions, explains causes, mechanisms, and implications rather than only listing facts.
- Recommendation quality: for "best", "must try", "which", or planning questions, gives useful options, criteria, and practical details.
- Evidence and trust: claims are plausible, grounded, and not misleading; citations or source-derived details help when they support the answer.
- Calibration: handles uncertainty, currentness, locality, and assumptions honestly.
- Organization: makes a rich answer easy to use through grouping, ordering, and clear takeaways.
- Noise control: extra text is a problem only when it is irrelevant, repetitive, unsupported, or distracts from the user's need.

Prefer the answer with greater useful substance for this user's question. Do not penalize an answer merely for being long if the added material is relevant and useful. Do not reward brevity, citation count, or fluent wording by itself. If both answers would be similarly useful or similarly flawed, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER_TREC = """Please act as a careful human TREC RAG side-by-side voter. Read the user's question and the two assistant answers, then choose the answer you would rather receive as the user.

Do not use a generic short-answer rubric. First infer what the user is trying to get done, then judge the answer by the dimensions that matter for that request. In RAG answer comparisons, a strong answer often wins because it gives more useful answer content, not because it is shorter or more polished.

Consider a broad set of possible dimensions:
- Intent match: answers the exact question, language, location, time frame, and requested scope.
- Answer density: contains many relevant, non-duplicative pieces of useful information.
- Coverage and recall: includes the important options, entities, subquestions, examples, caveats, and perspectives the user likely needs.
- Specificity: gives concrete names, dates, numbers, prices, addresses, source names, mechanisms, or distinctions when useful.
- Explanation quality: for "how", "why", effects, tradeoffs, or research questions, explains causes, mechanisms, and implications rather than only listing facts.
- Recommendation quality: for "best", "must try", "which", or planning questions, gives useful options, criteria, and practical details.
- Evidence and trust: claims are plausible, grounded, and not misleading; source-derived details or citations help when they support the answer.
- Calibration: handles uncertainty, currentness, locality, and assumptions honestly.
- Organization: makes a rich answer easy to use through grouping, ordering, and clear takeaways.
- Noise control: extra text is a problem only when it is irrelevant, repetitive, unsupported, or distracts from the user's need.

Prefer the answer with greater useful substance for this user's question. Do not penalize an answer merely for being long if the added material is relevant and useful. Do not reward brevity, citation count, or fluent wording by itself. If both answers would be similarly useful or similarly flawed, output a tie.

Do not write explanation or analysis. Output only the final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_RUBRICS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the following rubric when comparing the two answers:
1. Relevance: The answer directly addresses the user's question and does not drift into unrelated content.
2. Correctness: The answer is factually accurate, internally consistent, and does not make unsupported claims.
3. Completeness: The answer covers the key information needed to satisfy the user's question.
4. Helpfulness: The answer is clear, specific, and useful for the user's likely information need.
5. Concision: The answer avoids unnecessary detail; length alone should not be rewarded.

Apply the rubric holistically rather than assigning numeric scores. Correctness and relevance should matter most.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Lastly, if neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses satisfy the rubric similarly well, and "[[Both Bad]]" if both responses fail the rubric similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_NUGGETS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Use the following topic-specific nuggets as a rubric when comparing the two answers. A vital nugget represents information that should be present in a good answer. An okay nugget is useful but less essential. Prefer the answer that better covers the vital nuggets, covers more useful okay nuggets when otherwise comparable, avoids contradicting the nuggets, and stays focused on the user's question. Do not reward an answer merely for being longer.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. Lastly, if both responses cover the nuggets similarly well, or if both responses are similarly good or similarly bad under the nugget rubric, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Nuggets]
{nuggets}
[The End of Topic Nuggets]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_RUBRICS_AND_NUGGETS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Use the following topic-specific nuggets and rubric when comparing the two answers. A vital nugget represents information that should be present in a good answer. An okay nugget is useful but less essential.

Rubric:
1. Nugget coverage: The answer covers the vital nuggets and, when otherwise comparable, more useful okay nuggets.
2. Correctness: The answer is factually accurate, internally consistent, and does not contradict the nuggets.
3. Relevance: The answer directly addresses the user's question and does not drift into unrelated content.
4. Completeness: The answer covers the key information needed to satisfy the user's question.
5. Helpfulness and concision: The answer is clear, specific, and useful without unnecessary detail; length alone should not be rewarded.

Apply the rubric holistically rather than assigning numeric scores. Nugget coverage, correctness, and relevance should matter most.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. Lastly, if both responses cover the nuggets similarly well, or if both responses are similarly good or similarly bad under the rubric, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Nuggets]
{nuggets}
[The End of Topic Nuggets]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the following topic-specific rubric criteria when comparing the two answers. Each criterion describes something a strong answer should satisfy. Criteria marked mandatory or with higher weights should matter most; optional or lower-weight criteria can break ties when the answers are otherwise comparable. Prefer the answer that better satisfies the high-priority criteria, avoids contradicting them, and stays focused on the user's question. Do not reward an answer merely for being longer.

Apply the rubric holistically rather than assigning numeric scores. Be as objective as possible, and avoid any position biases: the order in which the responses were presented should not influence your decision. Lastly, if neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses satisfy the rubric similarly well, and "[[Both Bad]]" if both responses fail the rubric similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_STRICT_VITAL = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the following topic-specific rubric criteria as proxies for human nuggets. Mandatory criteria represent vital information and should dominate the decision. Optional or lower-weight criteria are not part of the strict-vital objective; use them only as tie-breakers after mandatory coverage, correctness, and relevance are essentially equal.

Compare the answers in this order:
1. Mandatory rubric coverage: prefer the answer that satisfies more mandatory criteria, especially criteria central to the user's question.
2. Correctness and relevance: penalize incorrect, misleading, unsupported, or off-topic content, especially if it contradicts mandatory criteria.
3. Direct usefulness: prefer answers that follow the user's request and answer the question clearly.
4. Optional criteria, synthesis, organization, and concision: use these only to break close ties; do not reward length for its own sake.

If one answer covers meaningfully more mandatory criteria without serious errors, choose it even if the other answer is smoother, longer, or has more optional detail. If neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_DIMENSIONS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the topic-specific rubric criteria as the main evidence for comparison. Mandatory criteria and higher-weight criteria matter most; optional criteria can break close ties. In addition to the topic rubric, evaluate these answer-quality dimensions:
1. Instruction following: follows the user's request, constraints, and expected format.
2. Relevance: directly addresses the actual question without drifting.
3. Factual accuracy: avoids incorrect, misleading, or unsupported claims beyond the rubric.
4. Grounding and citations: factual claims are supported by appropriate evidence when present; citations should be precise and useful, not decorative.
5. Completeness and balance: covers important caveats, tradeoffs, and perspectives without omitting essential context.
6. Synthesis quality: integrates facts into a coherent explanation rather than listing isolated points.
7. Calibration: states uncertainty or limitations when warranted and avoids overclaiming.
8. Clarity and organization: is readable, coherent, well-structured, and fluent.
9. Concision: provides useful detail without padding; do not reward length for its own sake.

Apply these dimensions holistically, but do not let style, length, or optional detail outweigh mandatory rubric coverage and factual correctness. Avoid any position biases: the order in which the responses were presented should not influence your decision. If neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses satisfy the rubric and quality dimensions similarly well, and "[[Both Bad]]" if both responses fail the rubric and quality dimensions similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_CHECKLIST = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question under the topic-specific rubric.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the following topic-specific rubric criteria when comparing the two answers. Criteria marked mandatory or with higher weights should matter most. Optional or lower-weight criteria can break ties when the answers are otherwise comparable.

Before deciding, internally check:
1. Which answer covers more mandatory rubric criteria?
2. Does either answer contradict the rubric, the user's question, or itself?
3. Does either answer include unsupported or irrelevant detail that should be penalized rather than rewarded?
4. If mandatory coverage is similar, which answer is more complete, calibrated, synthesized, clear, and concise?

Do not write out this checklist. Output only the final verdict. Avoid position bias and do not reward length for its own sake. If neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses satisfy the rubric similarly well, and "[[Both Bad]]" if both responses fail the rubric similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly supported by Assistant A?
2. Which mandatory criteria are clearly supported by Assistant B?
3. Which answer supports more mandatory criteria with accurate, relevant information?
4. Does either answer contain important contradictions, unsupported claims, or off-topic material that should reduce confidence in its coverage?

Prefer the answer with higher supported mandatory-criterion coverage. Do not reward an answer for merely mentioning rubric words without actually answering the question. Do not reward length, fluency, citations, or optional details unless mandatory coverage is effectively tied. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_QUALITY = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Evaluate correctness, helpfulness, completeness, accuracy, depth, and level of detail only as they affect mandatory rubric support:
1. Correctness and accuracy: the answer's statements about mandatory criteria must be true, specific, internally consistent, and not misleading.
2. Helpfulness: the answer must directly answer the user's question in a way that makes the mandatory information usable, not merely mention related facts.
3. Completeness: the answer should cover more mandatory criteria and include essential caveats needed to avoid a distorted answer.
4. Depth: deeper explanation is useful only when it clarifies why a mandatory criterion is satisfied or connects mandatory facts coherently.
5. Level of detail: precise, relevant detail can support a criterion; vague, padded, decorative, or off-topic detail should not help.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly supported by Assistant A?
2. Which mandatory criteria are clearly supported by Assistant B?
3. Which answer supports more mandatory criteria with correct, useful, complete, and precise information?
4. Does either answer contain important contradictions, unsupported claims, missing caveats, or irrelevant detail that should reduce confidence in its coverage?

Prefer the answer with higher supported mandatory-criterion coverage. Do not reward an answer for merely mentioning rubric words without actually answering the question. Do not reward length, fluency, citations, optional details, or broad background unless mandatory coverage is effectively tied. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_ACCURACY_GATE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Treat correctness and accuracy as gates for coverage. A mandatory criterion is supported only if the answer clearly provides accurate, relevant information for that criterion. Do not count a criterion when the answer is vague, speculative, contradicted, unsupported, misleading, or only loosely related to the user's question.

Use helpfulness, completeness, depth, and level of detail after the accuracy gate:
1. Helpfulness: prefer a direct answer that makes the supported mandatory information easy to use.
2. Completeness: prefer coverage of more mandatory criteria and necessary caveats.
3. Depth: prefer explanations that resolve the user's likely information need, but only when they are accurate and relevant.
4. Level of detail: prefer concrete, precise support; penalize padding, irrelevant details, and decorative citations.

Before deciding, internally estimate the number of mandatory criteria each answer accurately supports. Prefer the answer with more accurately supported mandatory criteria. If one answer contains a serious factual error or contradiction about a mandatory criterion, penalize it even if it is longer or more detailed. Optional criteria and writing quality are tie-breakers only after strict mandatory coverage is effectively tied.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher accurately supported mandatory coverage, "[[B]]" if Assistant B has higher accurately supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_HELPFUL_DEPTH = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Prioritize mandatory rubric coverage, then use the expanded quality dimensions to decide close comparisons:
1. Correctness and accuracy: claims must be factually right and must not contradict mandatory criteria.
2. Helpfulness: the response should directly satisfy what the user is asking, with clear implications or takeaways.
3. Completeness: the response should cover the important mandatory aspects of the topic without omitting essential context.
4. Depth: the response should synthesize and explain the mandatory information rather than list isolated fragments, but depth only counts when it is relevant.
5. Level of detail: enough precise detail to substantiate the answer is good; excessive or non-responsive detail is bad.

Before deciding, internally compare the answers on mandatory coverage first. If mandatory coverage differs, choose the answer with higher accurate mandatory coverage. If mandatory coverage is similar, choose the answer that is more helpful, complete, well-synthesized, and appropriately detailed for the user's question. Do not reward length, fluency, citations, optional facts, or background unless they make the mandatory answer more useful.

If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage or clearly better useful treatment of tied mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage or clearly better useful treatment of tied mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_DIMENSIONS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Use these quality dimensions only to decide whether mandatory criteria are truly supported: correctness and accuracy mean the answer is factually right and not misleading; helpfulness means it directly answers the user's information need; completeness means it covers the essential mandatory aspects and caveats; depth and level of detail mean the answer gives enough relevant specificity to substantiate the mandatory facts without padding.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly supported by Assistant A?
2. Which mandatory criteria are clearly supported by Assistant B?
3. Which answer supports more mandatory criteria with correct, accurate, helpful, complete, and appropriately detailed information?
4. Does either answer contain important contradictions, unsupported claims, missing caveats, or off-topic material that should reduce confidence in its coverage?

Prefer the answer with higher supported mandatory-criterion coverage. Do not reward an answer for merely mentioning rubric words without actually answering the question. Do not reward length, fluency, citations, or optional details unless mandatory coverage is effectively tied. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_TIEBREAK = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital and should drive the decision. Optional or lower-priority criteria are secondary.

First decide which mandatory criteria are actually supported. A mandatory criterion is supported only when the answer gives accurate, relevant, and sufficiently specific information for the user's question; do not count vague mentions, keyword matches, generic background, speculation, contradictions, or off-topic detail.

Then use these answer-quality dimensions to resolve close comparisons:
1. Relevance: directly addresses the user's question without drifting.
2. Correctness and accuracy: avoids incorrect, misleading, or unsupported claims.
3. Completeness and balance: includes essential context, caveats, tradeoffs, and perspectives needed to avoid a distorted answer.
4. Synthesis: connects the facts into a coherent explanation rather than listing isolated points.
5. Calibration: states uncertainty or limits when warranted and avoids overclaiming.
6. Clarity and organization: is readable, coherent, and easy to follow.
7. Concision: provides useful detail without padding; do not reward length for its own sake.

Prefer the answer with higher supported mandatory-criterion coverage. If mandatory coverage is close, prefer the answer that is more relevant, correct, complete, balanced, synthesized, calibrated, clear, and concise for the user's question. Do not let style, fluency, citations, optional facts, or extra length outweigh a clear mandatory-coverage difference.

If both answers support the mandatory criteria similarly well and are reliable, choose [[Both Good]]. If both answers miss most mandatory criteria, are vague, or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage or clearly better quality when mandatory coverage is close, "[[B]]" if Assistant B has higher supported mandatory coverage or clearly better quality when mandatory coverage is close, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_CALIBRATED = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Use these quality dimensions only to decide whether mandatory criteria are truly supported: relevance means the answer directly addresses the user's information need; correctness and accuracy mean the answer is factually right and not misleading; completeness and balance mean it covers the essential mandatory aspects, caveats, tradeoffs, and perspectives without distorting the answer; synthesis and depth mean it connects mandatory facts into a coherent explanation rather than listing isolated fragments; calibration means it states uncertainty or limits when warranted and avoids overclaiming; clarity and organization mean the support is easy to understand; concision and level of detail mean the answer gives enough relevant specificity to substantiate mandatory facts without padding.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly supported by Assistant A?
2. Which mandatory criteria are clearly supported by Assistant B?
3. Which answer supports more mandatory criteria with relevant, correct, accurate, complete, balanced, synthesized, calibrated, clear, and appropriately detailed information?
4. Does either answer contain important contradictions, unsupported claims, missing caveats, overclaiming, or off-topic material that should reduce confidence in its coverage?

Prefer the answer with higher supported mandatory-criterion coverage. Do not reward an answer for merely mentioning rubric words without actually answering the question. Do not reward length, fluency, citations, or optional details unless mandatory coverage is effectively tied. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage, "[[B]]" if Assistant B has higher supported mandatory coverage, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_SUPPORT_PLUS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that would receive the higher strict-vital human nugget score.

Use the topic-specific rubric as a proxy for human nuggets. Criteria marked mandatory are vital. Optional or lower-priority criteria are secondary.

Count a mandatory criterion as supported only when the answer gives information that is relevant to the user's question, factually correct, specific enough to verify, and not contradicted elsewhere in the answer. Do not count vague references, keyword overlap, unsupported speculation, irrelevant background, citations without useful content, or padded detail that does not answer the question.

Use these answer-quality dimensions as evidence for support quality, not as separate goals: correctness and accuracy, helpfulness for the user's actual information need, completeness of the mandatory facts and caveats, depth of explanation where it clarifies the mandatory facts, calibration when uncertainty or limits matter, clarity of organization, and concision without losing essential detail.

Before deciding, internally estimate:
1. Which mandatory criteria are clearly and correctly supported by Assistant A?
2. Which mandatory criteria are clearly and correctly supported by Assistant B?
3. Does either answer include a serious factual error, missing caveat, overclaim, contradiction, or off-topic material that should reduce its supported coverage?
4. If supported mandatory coverage is effectively tied, which answer communicates that support more helpfully, clearly, calibratedly, and concisely?

Prefer the answer with higher supported mandatory-criterion coverage. If coverage is effectively tied, prefer the answer with stronger support quality. Do not reward length, fluency, optional facts, or broad background unless they improve the answer to mandatory criteria. If both answers support the mandatory criteria similarly well, choose [[Both Good]]. If both answers miss most mandatory criteria or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A has higher supported mandatory coverage or stronger support quality when coverage is tied, "[[B]]" if Assistant B has higher supported mandatory coverage or stronger support quality when coverage is tied, "[[Both Good]]" if both responses satisfy the mandatory rubric criteria similarly well, and "[[Both Bad]]" if both responses miss most mandatory criteria or fail similarly badly.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_BALANCED_DIMENSIONS = """Please act as an impartial judge and compare the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the answer that is better overall for the user.

Use the topic-specific rubric as background for important facts and expectations, but do not reduce the decision to a checklist count. The rubric is one source of evidence; the side-by-side comparison should also judge the quality of the answer as a response to the user's actual question.

Compare the answers across these dimensions:
1. Relevance: directly addresses the user's question without drifting into loosely related material.
2. Completeness and balance: covers the essential context, caveats, tradeoffs, and perspectives needed for a sound answer.
3. Synthesis quality: integrates facts into a coherent explanation rather than listing isolated points.
4. Calibration: states uncertainty, limits, or conditionality when warranted and avoids overclaiming.
5. Clarity and organization: is readable, coherent, well-structured, and fluent.
6. Concision: gives useful detail without padding; do not reward length for its own sake.
7. Rubric alignment: is consistent with the topic-specific rubric and does not omit or contradict important rubric criteria.

Prefer the answer that would be more useful and trustworthy to an informed user. Strong rubric alignment matters, but a response can lose if it is poorly synthesized, poorly calibrated, padded, unclear, or not actually responsive. If neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better overall, "[[B]]" if Assistant B is better overall, "[[Both Good]]" if both responses are similarly strong, and "[[Both Bad]]" if both responses are similarly weak.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_QUESTION_FIRST_DIMENSIONS = """Please act as an impartial judge and compare the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the answer that best satisfies the user's information need.

Start from the user's question, not from the rubric. Use the topic-specific rubric to recognize important topic facts, missing information, and contradictions, but make the final side-by-side decision based on which response is the better answer to the question.

Evaluate:
1. Direct relevance: answers the actual question and avoids unrelated background.
2. Factual usefulness: gives accurate, non-misleading information that helps resolve the user's need.
3. Completeness and balance: includes essential context, caveats, tradeoffs, and perspectives without creating a distorted picture.
4. Synthesis: connects the important facts into a coherent explanation or answer.
5. Calibration: avoids overclaiming, acknowledges uncertainty or limits when the evidence is incomplete, and does not present speculation as fact.
6. Clarity and organization: makes the answer easy to understand and compare.
7. Concision: avoids padding and does not treat length as quality.
8. Rubric consistency: uses the rubric as a guardrail for topic-specific expectations, but does not prefer a mechanical rubric checklist over a better user-facing answer.

Prefer the answer that an informed human evaluator would find more relevant, complete, balanced, synthesized, calibrated, clear, and concise. Penalize answers that are verbose but unfocused, contain unsupported claims, miss crucial caveats, or merely mention rubric ideas without answering the question. If neither assistant is clearly better, distinguish between cases where both responses are good and cases where both responses are bad.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better for the user's question, "[[B]]" if Assistant B is better for the user's question, "[[Both Good]]" if both responses are similarly strong, and "[[Both Bad]]" if both responses are similarly weak.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_EXPERT_PREFERENCE = """Please act as an impartial expert judge and compare the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the response an expert human assessor should prefer.

The topic-specific rubric gives useful anchors, but this is a side-by-side preference judgment rather than a rubric-scoring task. Use the rubric to catch important omissions and contradictions, then judge which answer is stronger across multiple dimensions of answer quality.

Give substantial weight to:
1. Relevance: the response stays centered on the actual question.
2. Completeness and balance: the response covers necessary caveats, tradeoffs, and perspectives, and does not omit context that would change the answer.
3. Synthesis quality: the response explains relationships among facts and provides a coherent takeaway instead of a pile of facts.
4. Calibration: the response distinguishes what is known, likely, uncertain, or conditional, and avoids exaggerated certainty.
5. Clarity and organization: the response is easy to follow and logically structured.
6. Concision: the response includes enough useful detail but avoids filler, repeated points, and irrelevant elaboration.
7. Topic-rubric grounding: the response remains compatible with the rubric's topic-specific expectations.

A better answer may be shorter if it is more focused and well calibrated, or longer if the extra detail is necessary and well synthesized. Do not reward decorative citations, generic background, or exhaustive lists unless they improve the answer. If both responses are similarly strong, choose [[Both Good]]. If both are similarly weak, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is the expert-preferred answer, "[[B]]" if Assistant B is the expert-preferred answer, "[[Both Good]]" if both responses are similarly strong, and "[[Both Bad]]" if both responses are similarly weak.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_STRICT_SUPPORT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Your goal is to approximate a strict-vital human nugget score.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use the topic-specific rubric as a proxy for human nuggets. Treat mandatory criteria as vital nuggets. The winning answer is the one that would get more mandatory criteria marked as strict support by a human evaluator.

Internally evaluate each mandatory criterion as follows:
1. Count it as supported only if the answer clearly, specifically, and accurately provides the required information.
2. Do not count vague mentions, partial implications, generic background, irrelevant facts, or claims that are not responsive to the user's question.
3. Do not count a criterion if the answer contradicts it, overstates it, or relies on unsupported speculation.
4. Optional or lower-priority criteria, prose quality, citations, and length are tie-breakers only after strict mandatory support is essentially tied.

Prefer the answer with more strictly supported mandatory criteria. If both answers strictly support about the same mandatory criteria and are reliable, choose [[Both Good]]. If both answers miss most mandatory criteria, are vague, or are similarly unreliable, choose [[Both Bad]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A would receive the higher strict-vital score, "[[B]]" if Assistant B would receive the higher strict-vital score, "[[Both Good]]" if both responses would receive similarly high strict-vital scores, and "[[Both Bad]]" if both responses would receive similarly low strict-vital scores.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Topic Rubric]
{rubric}
[The End of Topic Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

TIE_VERDICTS = frozenset({"Tie", "Both Good", "Both Bad"})
NATIVE_PROMPT_VARIANTS = {
    "default": ARENA_JUDGE_PROMPT,
    "rich-human-voter": ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER,
    "rich-human-voter-trec": ARENA_JUDGE_PROMPT_NATIVE_RICH_HUMAN_VOTER_TREC,
}
TOPIC_RUBRIC_PROMPT_VARIANTS = {
    "default": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC,
    "strict-vital": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_STRICT_VITAL,
    "dimensions": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_DIMENSIONS,
    "checklist": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_CHECKLIST,
    "coverage-count": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COUNT,
    "coverage-quality": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_QUALITY,
    "coverage-accuracy-gate": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_ACCURACY_GATE,
    "coverage-helpful-depth": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_HELPFUL_DEPTH,
    "coverage-compact-dimensions": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_DIMENSIONS,
    "coverage-compact-tiebreak": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_TIEBREAK,
    "coverage-compact-calibrated": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_COVERAGE_COMPACT_CALIBRATED,
    "support-plus": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_SUPPORT_PLUS,
    "balanced-dimensions": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_BALANCED_DIMENSIONS,
    "question-first-dimensions": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_QUESTION_FIRST_DIMENSIONS,
    "expert-preference": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_EXPERT_PREFERENCE,
    "strict-support": ARENA_JUDGE_PROMPT_W_TOPIC_RUBRIC_STRICT_SUPPORT,
}

_VERDICT_RE = re.compile(r"\s*\[\[(A|B|Tie|Both Good|Both Bad)\]\]\s*")


def render_arena_prompt(
    *,
    query: str,
    answer_a: str,
    answer_b: str,
    rubrics: bool = False,
    nuggets: str | None = None,
    rubric: str | None = None,
    prompt_variant: str = "default",
) -> str:
    if rubric is not None and nuggets is not None:
        raise ValueError("arena prompts accept either topic rubrics or nuggets, not both")
    if rubric is not None:
        prompt = TOPIC_RUBRIC_PROMPT_VARIANTS.get(prompt_variant)
        if prompt is None:
            variants = ", ".join(sorted(TOPIC_RUBRIC_PROMPT_VARIANTS))
            raise ValueError(f"unknown topic rubric prompt variant {prompt_variant!r}; expected one of: {variants}")
        return prompt.format(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            rubric=rubric,
        )
    if nuggets is not None:
        prompt = ARENA_JUDGE_PROMPT_W_RUBRICS_AND_NUGGETS if rubrics else ARENA_JUDGE_PROMPT_W_NUGGETS
        return prompt.format(query=query, answer_a=answer_a, answer_b=answer_b, nuggets=nuggets)
    if rubrics:
        prompt = ARENA_JUDGE_PROMPT_W_RUBRICS
    else:
        prompt = NATIVE_PROMPT_VARIANTS.get(prompt_variant)
        if prompt is None:
            variants = ", ".join(sorted(NATIVE_PROMPT_VARIANTS))
            raise ValueError(f"unknown native arena prompt variant {prompt_variant!r}; expected one of: {variants}")
    return prompt.format(query=query, answer_a=answer_a, answer_b=answer_b)


def parse_verdict(text: str) -> str | None:
    match = _VERDICT_RE.fullmatch(text)
    return match.group(1) if match else None
