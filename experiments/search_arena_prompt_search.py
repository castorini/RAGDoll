from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from datasets import load_dataset
from scipy.stats import kendalltau, spearmanr

from ragdoll.arena.metrics import fit_arena_ratings, leaderboard_rows
from ragdoll.arena.prompts import ARENA_JUDGE_PROMPT, TIE_VERDICTS, parse_verdict
from ragdoll.arena.stages import RubricRecord, format_rubric_for_prompt, load_rubrics
from ragdoll.config import LocalAgentConfig
from ragdoll.jsonl import append_jsonl, read_jsonl
from ragdoll.runner import run_prompt

SEARCH_ARENA_COMPACT_DIMENSIONS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Use these quality dimensions to decide which response is better: correctness and accuracy mean the answer is factually right and not misleading; helpfulness means it directly answers the user's information need; completeness means it covers the essential aspects and caveats; depth and level of detail mean the answer gives enough relevant specificity to substantiate its claims without padding.

Before deciding, internally estimate:
1. Which response more directly satisfies the user's information need?
2. Which response is more factually correct, accurate, and well supported?
3. Which response covers the important aspects and caveats more completely?
4. Does either response contain unsupported claims, missing caveats, irrelevant detail, or padding that should reduce confidence?

Prefer the response that is more correct, helpful, complete, and appropriately detailed. Do not reward length, fluency, or extra detail unless it helps answer the user's question. If both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_HUMAN_PREFERENCE = """Please act as an impartial human preference judge comparing two search-answer assistants. Choose the response that a careful user should prefer after reading both answers.

Compare the answers holistically across these dimensions:
1. Direct usefulness: answers the user's actual question, including implicit constraints and the needed level of specificity.
2. Factual reliability: avoids false, misleading, stale, or overconfident claims.
3. Evidence use: uses sources, citations, or grounded details in a way that makes factual claims checkable; citations should support the specific claim they accompany.
4. Completeness and balance: includes the important facts, caveats, tradeoffs, and perspectives without omitting context needed for a good answer.
5. Synthesis: integrates facts into a coherent answer instead of dumping loosely related snippets.
6. Clarity and concision: is easy to read and avoids padding; do not reward length by itself.

Correctness, direct usefulness, and evidence quality are more important than style. Penalize unsupported specifics, citation misuse, evasiveness, and irrelevant detail. If neither answer is clearly preferable, output a tie.

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


SEARCH_ARENA_SEARCH_QUALITY = """Please act as an impartial judge for a search-answer arena. Your job is to choose which assistant response better satisfies the user's search or information need.

Evaluate the two answers in this order:
1. Relevance: Does the answer address exactly what the user asked, including entities, dates, locations, comparisons, or constraints?
2. Verifiable correctness: Are concrete factual claims likely correct and not contradicted by the answer itself?
3. Grounding and citation behavior: Are sources or citations used precisely and usefully when factual support is needed? Do not reward decorative citations.
4. Completeness: Does the answer include the central facts, caveats, and context needed to resolve the query?
5. Calibration: Does the answer avoid overclaiming when the evidence is uncertain, current, local, or ambiguous?
6. Presentation: Is the answer readable and concise while still giving enough detail?

Choose the response with the better evidence-backed answer to the user's need. A shorter answer can be better if it is more accurate and focused. A longer answer can be better only when the extra detail is relevant and reliable. If both are close, output a tie.

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


SEARCH_ARENA_STRICT_HELPFUL = """Please act as an impartial judge and compare two assistant responses. Choose the response that is more helpful to the user, while being strict about factual accuracy.

Use this decision rule:
- First, reject or penalize answers with clear factual errors, hallucinated specifics, wrong entities, unsupported claims, or citations that do not support the claim.
- Among answers that are factually plausible, prefer the answer that more directly and completely resolves the user's request.
- Use depth, detail, synthesis, and organization as tie-breakers only when they improve the answer.
- Penalize generic filler, hedging that avoids answering, irrelevant background, and length for its own sake.

If both responses are strong in similar ways or weak in similar ways, output a tie. Be objective and avoid position bias.

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


SEARCH_ARENA_USER_OUTCOME = """Please act as an impartial judge. Choose which assistant response gives the user the better outcome.

Think from the user's perspective:
1. Would the user get the information or decision support they asked for?
2. Is the answer specific enough to be useful without adding distracting or unrelated material?
3. Are factual claims reliable, current when needed, and supported by citations or evidence when the answer provides them?
4. Does the answer handle ambiguity, uncertainty, or caveats honestly?
5. Is the answer coherent, well organized, and not padded?

Prefer the answer that best combines relevance, factual reliability, useful completeness, synthesis, calibration, and concise clarity. Do not choose an answer just because it is longer or more polished. If the difference is not meaningful, output a tie.

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


SEARCH_ARENA_COMPACT_DEPTH = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Use these quality dimensions: correctness and accuracy; relevance to the user's exact information need; completeness over the central facts and caveats; useful depth and reasoning; synthesis into a coherent answer; and clarity without unnecessary padding.

Important calibration: do not penalize an answer merely because it is longer, more detailed, or more reasoning-oriented. Detailed reasoning is valuable when it helps solve the user's question, connects evidence, handles caveats, or explains a comparison. Penalize length only when the extra material is irrelevant, generic, unsupported, or obscures the answer.

Before deciding, internally ask:
1. Which response gives the user the more reliable and useful answer?
2. Which response better covers the important facts, caveats, and reasoning needed for this query?
3. Which response avoids factual errors, unsupported specifics, stale claims, or misleading simplifications?
4. Is either response merely polished or concise while missing important substance?

Prefer the answer with the better combination of factual reliability, direct usefulness, completeness, and useful depth. If both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_COMPACT_CALIBRATED_TIE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below.

Choose a winner only when one response is meaningfully better for the user. If the difference is small, uncertain, or mostly stylistic, output a tie.

Evaluate:
1. Correctness and accuracy: factual claims should be right, non-misleading, and internally consistent.
2. Relevance and helpfulness: the answer should address the user's actual question and constraints.
3. Completeness and balance: the answer should include important facts, caveats, and context needed to satisfy the query.
4. Useful depth: explanations, reasoning, examples, or citations help only when they substantively answer the question.
5. Clarity and concision: the answer should be readable and avoid irrelevant padding.

Do not reward length by itself, but also do not prefer a shorter answer if it omits useful context or reasoning. When both answers are plausible and neither has a clear advantage in substance, output [[Tie]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is meaningfully better, "[[B]]" if Assistant B is meaningfully better, and "[[Tie]]" for a tie.

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


SEARCH_ARENA_COMPACT_SEARCH_REASONING = """Please act as an impartial judge for a search-answer comparison. Choose which assistant response better satisfies the user.

A strong search answer should be factually reliable, directly responsive, complete enough to answer the question, and well synthesized. It may be concise or detailed. For complex, comparative, time-sensitive, local, or multi-hop questions, reward responses that reason through the answer, explain assumptions, reconcile evidence, and include relevant caveats.

Compare the responses using these dimensions:
1. Direct answer quality: gives the user the answer they asked for, not just related facts.
2. Factual reliability: avoids wrong entities, wrong dates, unsupported claims, or overconfident statements.
3. Evidence and grounding: citations or source-derived details are useful when they support specific claims, but citation count alone does not matter.
4. Completeness and synthesis: covers the central facts and connects them coherently.
5. Calibration: states uncertainty or limitations when the question requires it.
6. Readability: clear organization and no irrelevant padding.

Prefer useful substantive depth over superficial polish. If both responses would satisfy the user similarly well, or both fail similarly, output a tie.

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


SEARCH_ARENA_NATIVE_WITH_DEPTH_NOTE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Useful depth matters: for questions that require explanation, comparison, recent information, local details, or multi-step reasoning, a longer answer can be better when the added reasoning, evidence, or caveats are relevant and reliable. Do not penalize a response for being detailed unless the detail is irrelevant, unsupported, or distracting.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow length alone to influence your evaluation. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_CONTEXT_FINAL_DEPTH = """Please act as an impartial Search Arena judge. You may be shown a single-turn answer or a multi-turn conversation transcript. Use prior turns only to understand the user's latest request, then judge which assistant's final response is better for that request.

Primary decision rule:
1. First check relevance to the latest user request. Strongly penalize an answer that solves a previous request, changes the topic, ignores constraints, or gives mostly unrelated information.
2. Among relevant answers, prefer the response with stronger factual reliability, useful completeness, and substantive depth.
3. Reward reasoning, examples, citations, comparisons, caveats, and context only when they help answer the latest request.
4. Do not reward length, citation count, polish, or confident wording by itself. Do not prefer a very concise answer if it omits important substance.
5. If both final responses are similarly useful, similarly incomplete, or the difference is mainly stylistic, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of Latest User Request]
{query}
[The End of Latest User Request]

[The Start of Assistant A Conversation]
{transcript_a}
[The End of Assistant A Conversation]

[The Start of Assistant B Conversation]
{transcript_b}
[The End of Assistant B Conversation]
"""


SEARCH_ARENA_ACTIONABLE_HELPFUL = """Please act as an impartial judge comparing two search assistant responses. Choose the response that gives the user the more useful outcome, not the response that merely looks more polished.

Evaluate the final response to the user's request using these dimensions:
- Direct answer: answers the specific question or task the user asked.
- Actionability: gives concrete information, examples, names, steps, dates, tradeoffs, or recommendations when the request calls for them.
- Factual reliability: avoids false, stale, misleading, or unsupported claims.
- Completeness: covers the essential points and caveats without leaving the user unable to act.
- Synthesis: connects facts into a coherent answer rather than listing loosely related snippets.
- Concision: avoids padding, but useful detail is a strength.

Search grounding and citations are helpful only when they support the answer's factual claims. A response with fewer citations can be better if it is more directly useful and reliable. If neither response is clearly better for the user, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User Request]
{query}
[The End of User Request]

[The Start of Assistant A Final Response]
{final_answer_a}
[The End of Assistant A Final Response]

[The Start of Assistant B Final Response]
{final_answer_b}
[The End of Assistant B Final Response]
"""


SEARCH_ARENA_RELEVANCE_GATE = """Please act as an impartial judge. Choose which assistant final response better satisfies the user request.

Use a relevance gate before considering other qualities:
1. If one response directly answers the requested topic, entity, location, format, or constraint and the other mostly does not, choose the relevant response unless it contains serious factual errors.
2. If both are relevant, compare factual correctness, completeness, useful depth, synthesis, calibration, and clarity.
3. If both are off-target or both fail similarly, output a tie unless one still gives noticeably more useful information.

Important: useful depth is not the same as verbosity. Reward detailed reasoning, context, citations, or lists when they are relevant and reliable. Penalize filler, generic background, unsupported specifics, and irrelevant detail. For informational search tasks, prefer the answer that a careful human user would actually rely on.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User Request]
{query}
[The End of User Request]

[The Start of Assistant A Final Response]
{final_answer_a}
[The End of Assistant A Final Response]

[The Start of Assistant B Final Response]
{final_answer_b}
[The End of Assistant B Final Response]
"""


SEARCH_ARENA_HUMAN_SEARCH_VOTE = """You are simulating a careful human Search Arena voter. Read the user's request and the two assistant final responses. Vote for the response you would rather receive if you asked the question.

The best response usually has the strongest combination of:
- Relevance to the exact request.
- Correct and non-misleading factual content.
- Enough detail to be useful, including key caveats or comparisons when needed.
- Good synthesis rather than a raw dump of facts.
- Clear organization and readable wording.
- Honest calibration when the topic is uncertain, current, local, or disputed.

Do not use superficial rules. Longer is not automatically better, shorter is not automatically better, citations are not automatically better, and fluent wording is not automatically better. Prefer the answer that is more useful and trustworthy overall. If the two responses are close, both good, or both bad in similar ways, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User Request]
{query}
[The End of User Request]

[The Start of Assistant A Final Response]
{final_answer_a}
[The End of Assistant A Final Response]

[The Start of Assistant B Final Response]
{final_answer_b}
[The End of Assistant B Final Response]
"""


SEARCH_ARENA_SUBSTANCE_OVER_STYLE = """Please act as an impartial judge for side-by-side search answer evaluation. Choose the answer with better substance for the user's request.

Substance means:
1. It answers the user's actual question rather than an adjacent or previous question.
2. It is factually accurate and avoids hallucinated specifics.
3. It includes the important information a user needs to understand, decide, or act.
4. It explains relationships, tradeoffs, or caveats when the query is complex.
5. It is grounded enough to trust, but citations and links matter only if they support the claims.

Style matters only as a secondary factor: organization, fluency, and concision can break ties, but they should not outweigh relevance, correctness, completeness, and useful depth. Do not penalize a response for being reasoning-oriented when the reasoning helps answer the question. If both answers have similar substantive value, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of User Request]
{query}
[The End of User Request]

[The Start of Assistant A Final Response]
{final_answer_a}
[The End of Assistant A Final Response]

[The Start of Assistant B Final Response]
{final_answer_b}
[The End of Assistant B Final Response]
"""


SEARCH_ARENA_CONTEXT_FINAL_BALANCED_TIE = """Please act as an impartial Search Arena judge. You may be shown a single-turn answer or a multi-turn conversation transcript. Use prior turns to understand the user's latest request, then judge which assistant's final response is better for that request.

Choose a winner only when one final response is meaningfully better for the user. If the difference is small, uncertain, mostly stylistic, or both responses are similarly good or similarly bad, output a tie.

Evaluate in this order:
1. Relevance to the latest user request, including entities, location, format, and constraints.
2. Factual reliability and absence of misleading, stale, or unsupported claims.
3. Completeness over the central facts, caveats, and reasoning needed to satisfy the request.
4. Useful depth, examples, citations, comparisons, or explanation when they improve the answer.
5. Clarity and organization, without rewarding polish over substance.

Do not penalize a response for being longer or reasoning-oriented when the added material is relevant and reliable. Do penalize filler, source lists that do not support the answer, generic background, and answers to a previous or adjacent question.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is meaningfully better, "[[B]]" if Assistant B is meaningfully better, and "[[Tie]]" for a tie.

[The Start of Latest User Request]
{query}
[The End of Latest User Request]

[The Start of Assistant A Conversation]
{transcript_a}
[The End of Assistant A Conversation]

[The Start of Assistant B Conversation]
{transcript_b}
[The End of Assistant B Conversation]
"""


SEARCH_ARENA_CONTEXT_FINAL_ANSWER_FIRST = """Please act as an impartial Search Arena judge. Compare the two assistant final responses to the user's latest request. Prior conversation is provided after each final response only to clarify context; judge the final response in that context.

A better final response should:
- Answer the user's latest request directly.
- Be factually reliable and not misleading.
- Include the important information, caveats, and reasoning needed for the request.
- Use citations, examples, comparisons, or detail when they make the answer more useful and trustworthy.
- Be clear and organized without padding.

Relevant substance is more important than brevity, citation count, or surface polish. A longer answer can be better when it gives useful evidence, reasoning, or context. A shorter answer can be better when it is more accurate and focused. If both final responses are close in user value, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of Latest User Request]
{query}
[The End of Latest User Request]

[The Start of Assistant A Final Response]
{final_answer_a}
[The End of Assistant A Final Response]

[The Start of Assistant B Final Response]
{final_answer_b}
[The End of Assistant B Final Response]

[The Start of Assistant A Conversation Context]
{transcript_a}
[The End of Assistant A Conversation Context]

[The Start of Assistant B Conversation Context]
{transcript_b}
[The End of Assistant B Conversation Context]
"""


SEARCH_ARENA_CONTEXT_FINAL_SUBSTANCE = """Please act as an impartial Search Arena judge. You may be shown a single-turn answer or a multi-turn conversation transcript. Use prior turns only to understand the user's latest request, then choose which assistant's final response has better substance for that request.

Substance means:
1. The final response answers the actual latest request rather than an adjacent or previous request.
2. It is factually accurate, specific, and not misleading.
3. It includes the important information a user needs to understand, decide, or act.
4. It explains relationships, tradeoffs, uncertainty, or caveats when the query is complex.
5. It is grounded enough to trust, but citations and links matter only if they support the claims.

Style matters only as a secondary factor. Organization, fluency, and concision can break ties, but they should not outweigh relevance, correctness, completeness, and useful depth. Do not penalize a response for detailed reasoning when the reasoning helps answer the question. If both final responses have similar substantive value, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of Latest User Request]
{query}
[The End of Latest User Request]

[The Start of Assistant A Conversation]
{transcript_a}
[The End of Assistant A Conversation]

[The Start of Assistant B Conversation]
{transcript_b}
[The End of Assistant B Conversation]
"""


SEARCH_ARENA_CONTEXT_FINAL_REASONING_VALUE = """Please act as an impartial Search Arena judge. You may be shown a single-turn answer or a multi-turn conversation transcript. Use prior turns only to understand the user's latest request, then judge which assistant's final response is better.

First identify what the user needs: a fact, a list, a recommendation, an explanation, a comparison, current information, local information, or step-by-step help. Then compare final responses using these priorities:
1. Directly satisfies that need.
2. Gives reliable factual content with appropriate specificity.
3. Provides enough reasoning, evidence, examples, caveats, or synthesis for the type of request.
4. Avoids unsupported claims, irrelevant detail, and overconfident simplifications.
5. Is readable and not padded.

Important: for complex informational questions, a response that explains why, compares options, or connects evidence can be better than a short answer even if both mention the same headline facts. Do not reward chain-of-thought-like verbosity; reward reasoning only when it appears in the answer as useful explanation for the user. If neither response is meaningfully better, output a tie.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, and "[[Tie]]" for a tie.

[The Start of Latest User Request]
{query}
[The End of Latest User Request]

[The Start of Assistant A Conversation]
{transcript_a}
[The End of Assistant A Conversation]

[The Start of Assistant B Conversation]
{transcript_b}
[The End of Assistant B Conversation]
"""


SEARCH_ARENA_COMPACT_GROUNDED_DEPTH = """Please act as an impartial judge for a search-answer arena. Choose which assistant response better satisfies the user's information need.

Use these dimensions in order:
1. Direct relevance to the user's exact request.
2. Factual reliability, including avoiding stale, misleading, or unsupported claims.
3. Search grounding: important factual claims should be backed by useful citations, source names, concrete evidence, or verifiable details. Citations count only when they support the claims they accompany.
4. Completeness and useful depth over the key facts, caveats, comparisons, and context needed for the request.
5. Synthesis and clarity: the answer should integrate information coherently without padding.

Do not reward length by itself, but do not penalize detailed answers when the detail is relevant, grounded, and useful. For current, local, medical, legal, financial, product, or entity-specific queries, source grounding and calibration should matter more. If both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_SOURCE_TRUST = """Please act as an impartial judge comparing two search assistant responses. Choose the response a careful user should trust more.

A trustworthy search answer:
- Directly answers the user's request and constraints.
- Makes factual claims that are likely correct and not misleading.
- Shows useful grounding through citations, named sources, concrete evidence, dates, or checkable details.
- Covers the essential facts and caveats without omitting important context.
- Explains or synthesizes the answer when the query requires reasoning, comparison, or a recommendation.
- Is clear and concise, but not shallow.

Treat unsupported specificity as a serious weakness. Treat decorative or irrelevant citations as weak evidence. A response with fewer citations can win if its claims are more relevant and reliable; a response with more citations can win if those citations substantively support a fuller answer. If the difference is not meaningful, output a tie.

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


SEARCH_ARENA_CURRENTNESS_AND_COVERAGE = """Please act as an impartial judge for search-answer quality. Pick the answer that better resolves the user's question.

First decide what kind of information the user needs. For current, local, named-entity, product, institutional, policy, or time-sensitive questions, strongly prefer answers that are specific, up to date, well grounded, and calibrated. For how-to or explanatory questions, prefer answers that give correct steps or explanations with enough detail to be useful.

Compare the responses on:
1. Relevance to the exact query.
2. Correctness and currentness when currentness matters.
3. Grounding quality: citations, source-derived details, and evidence should support the claims.
4. Coverage of the important facts, caveats, and alternatives.
5. Useful reasoning and synthesis.
6. Clear organization without irrelevant padding.

Do not choose a response just because it is shorter, longer, more fluent, or has more links. Choose the response with better reliable coverage of the user's actual need. If both are close, output a tie.

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


SEARCH_ARENA_COMPACT_DEPTH_SOURCE_TIEBREAK = """Please act as an impartial judge and evaluate the quality of two AI assistant responses to the user question. Choose the assistant that better answers the question.

Use these core dimensions: factual correctness and accuracy; relevance to the user's exact information need; completeness over central facts and caveats; useful depth and reasoning; synthesis into a coherent answer; and clarity without unnecessary padding.

When both responses are relevant and plausible, use grounding as an important tie-breaker: prefer the response whose factual claims are better supported by citations, source names, dates, concrete evidence, or other checkable details. Do not reward citations that are generic, decorative, or disconnected from the claims. Do not penalize a useful detailed answer merely because it is longer.

Before deciding, internally ask:
1. Which response gives the user the more reliable and useful answer?
2. Which response better covers the facts, caveats, and reasoning needed for this query?
3. Which response is better grounded and less likely to mislead?
4. Is either response polished but shallow, concise but incomplete, or detailed but unsupported?

If both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_NATIVE_COVERAGE_FIRST = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

For search-answer comparisons, put extra weight on complete coverage of the user's actual information need. First identify every part of the request: entities, dates, locations, comparisons, constraints, subquestions, and requested format. Prefer the response that answers more of those required parts with specific, relevant, and reliable information. Do not prefer a cleaner or shorter answer if it omits important requested details, caveats, or distinctions. A detailed answer can be better when its details are on topic and help resolve the request.

Use citations, source names, dates, examples, and concrete facts as evidence only when they support the answer. Do not reward citation count, source lists, generic background, or polished wording by itself. Penalize unsupported specifics, stale or misleading claims, irrelevant padding, and answers that drift from the question.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow length alone to influence your evaluation. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_NATIVE_PARTIAL_ANSWER_PENALTY = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Use this decision process:
1. Check whether each answer directly addresses the exact question, including all subquestions and constraints.
2. Penalize serious factual errors, misleading claims, unsupported specifics, or stale information.
3. Among factually plausible answers, prefer the one with better useful coverage: more of the important facts, distinctions, examples, caveats, and context needed for the user to understand or act.
4. Use clarity, organization, and concision only as tie-breakers. Do not choose a shallow partial answer merely because it is tidy, and do not choose a long answer merely because it is long.

For search-style answers, relevant citations or concrete source-derived details can make an answer more trustworthy, but citations are valuable only when they support the claims being made. If both responses answer the question about equally well, output a tie.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

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


SEARCH_ARENA_NATIVE_DETAILED_DIMENSIONS = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail:
- Correctness: the answer should not contain false statements, contradictions, wrong entities, wrong dates, or unsupported conclusions.
- Helpfulness: the answer should satisfy the user's actual information need and leave the user better able to understand, decide, or act.
- Completeness: the answer should cover all important parts of the question, including subquestions, comparisons, caveats, and constraints.
- Accuracy: names, numbers, dates, terminology, and qualifications should be precise rather than vague, stale, or misleading.
- Depth: explanation, reasoning, examples, and context are valuable when they help answer the question or connect evidence.
- Level of detail: specific details are useful when they are relevant and reliable; irrelevant details, generic background, or padding should not help.

Prefer the response with the best overall combination of these qualities. Do not prefer a shorter answer if it is incomplete, and do not prefer a longer answer if its extra material is irrelevant, unsupported, or distracting.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_NATIVE_INTENT_COMPLIANCE = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Start by identifying the user's intent and any instructions, constraints, or expected format. Then compare the answers using these dimensions:
1. Instruction following: follows the requested task, scope, format, language, location, time frame, and constraints.
2. Correctness and accuracy: avoids false, stale, misleading, internally inconsistent, or overconfident claims.
3. Helpfulness: gives the user a usable answer rather than adjacent facts or generic commentary.
4. Completeness: addresses all important subquestions, entities, comparisons, and caveats.
5. Useful depth and detail: includes relevant reasoning, examples, sources, or concrete facts when they improve the answer; avoids padding.
6. Clarity: makes the answer easy to scan and understand without letting polish outweigh substance.

A response that violates the user's explicit instruction or answers the wrong task should usually lose unless the other response is clearly worse. Do not reward length, citation count, or fluent wording by itself. If both responses satisfy the request similarly well, or fail similarly, output a tie.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

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


SEARCH_ARENA_NATIVE_USER_EFFORT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Judge from the perspective of a user who wants the most useful answer with the least extra work. Consider:
1. Task completion: does the response actually resolve what the user asked?
2. Actionability: does it provide concrete names, steps, comparisons, dates, examples, or recommendations when the question calls for them?
3. Reliability: are factual claims correct, precise, and appropriately qualified?
4. Completeness without burden: does it include the necessary context and caveats without forcing the user to infer missing pieces or sift through irrelevant material?
5. Decision value: for ambiguous, comparative, local, or current questions, does it help the user choose, verify, or understand tradeoffs?
6. Presentation: is it organized enough to use immediately?

Use correctness, helpfulness, completeness, accuracy, depth, and level of detail as supporting criteria. Relevant detail that reduces user effort is good; irrelevant detail that increases user effort is bad. If both responses would be similarly useful to the user, output a tie.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

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


SEARCH_ARENA_NATIVE_FAILURE_MODES = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Evaluate correctness, helpfulness, completeness, accuracy, depth, and level of detail, but explicitly check for common failure modes:
- Wrong target: answers a different question, misses a key entity, ignores a location/time constraint, or omits an explicit subquestion.
- False confidence: presents uncertain, stale, speculative, or unsupported information as fact.
- Shallow answer: gives a polished but partial response that lacks important distinctions, caveats, or concrete details.
- Noisy answer: adds long generic background, source lists, or irrelevant facts that do not help the user's question.
- Misleading evidence: uses citations, names, dates, or numbers that do not actually support the claim.
- Poor calibration: fails to acknowledge ambiguity, uncertainty, limitations, or high-stakes caveats when those matter.

Prefer the answer that best avoids these failure modes while still being useful, complete, accurate, and appropriately detailed. Do not reward length or style by itself. If both responses have similar strengths and weaknesses, output a tie.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

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


SEARCH_ARENA_MULTIAXIS_ADAPTIVE = """Please act as an impartial judge comparing two assistant answers to the user's question. Choose the answer that a careful user should prefer.

Do not rely on a fixed generic checklist. First infer what kind of question this is: fact lookup, explanation, comparison, recommendation, troubleshooting, planning, local/current search, definition, list, translation, coding, or multi-part research. Then use the comparison axes that actually matter for that query.

Potential axes include:
1. Intent fit: answers the exact task the user meant, not a nearby task.
2. Constraint following: respects requested format, language, scope, location, time frame, and exclusions.
3. Answerability: recognizes ambiguity, missing information, or when the question cannot be answered confidently.
4. Directness: gives the answer early enough that the user does not have to hunt for it.
5. Requirement coverage: covers all entities, subquestions, comparisons, and edge cases the user asked about.
6. Evidence quality: supports important claims with concrete, relevant, checkable evidence when appropriate.
7. Claim precision: uses accurate names, dates, numbers, definitions, and qualifications.
8. Currentness/locality: handles time-sensitive or location-specific facts correctly.
9. Distinction quality: explains differences, tradeoffs, categories, or causal relationships when the task requires comparison.
10. Decision value: helps the user choose, act, verify, or understand consequences.
11. Robustness: avoids advice that only works under unstated assumptions.
12. Calibration: avoids false certainty, overclaiming, or burying uncertainty.
13. Noise control: avoids irrelevant background, decorative citations, repeated content, or filler.
14. Readability: organizes information so the answer is usable.

Select the answer with the better user outcome under the relevant axes. A response can win by being more complete, more precise, more actionable, more honest about uncertainty, or better aligned with the task. If both are similarly useful or similarly flawed, output a tie.

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


SEARCH_ARENA_TASK_TYPE_ADAPTIVE = """Please act as an impartial judge comparing two assistant answers. Choose the answer that best satisfies the user's query.

Use a task-type-adaptive comparison:
- For fact lookup, prefer precise, correct, directly stated facts with enough context to avoid misunderstanding.
- For lists or recommendations, prefer useful selection criteria, relevant options, coverage of constraints, and clear tradeoffs.
- For comparisons, prefer answers that identify dimensions of difference, not just isolated descriptions.
- For explanations, prefer causal structure, definitions, examples, and caveats that improve understanding.
- For current or local questions, prefer freshness, location awareness, source-aware phrasing, and uncertainty handling.
- For how-to or troubleshooting, prefer executable steps, prerequisites, failure cases, and warnings.
- For broad research questions, prefer synthesis, balanced coverage, and separation of central facts from secondary details.

Across all task types, penalize wrong-target answers, unsupported specifics, misleading confidence, missing constraints, irrelevant detail, and answers that require the user to do extra work to find the actual answer. Do not reward length, citation count, or surface polish by itself.

If one answer is better for the user's task type, choose it. If the difference is not meaningful, output a tie.

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


SEARCH_ARENA_CLAIM_AUDIT = """Please act as an impartial judge comparing two assistant answers. Choose the answer whose claims are more useful and trustworthy for the user's question.

Compare the answers by auditing their claims:
1. Identify the main claims each answer makes that matter for answering the user.
2. Penalize claims that are likely false, stale, too broad, too specific without support, internally inconsistent, or not responsive to the question.
3. Reward claims that are precise, checkable, relevant, and connected to the user's actual task.
4. Reward citations or named evidence only when they support the specific claim; ignore decorative links or vague source mentions.
5. Consider omissions: an answer can be worse if it avoids false claims but misses central facts needed by the user.
6. Consider misleading framing: an answer can be worse if individual facts are true but the overall takeaway is unbalanced or likely to mislead.

This is not a fluency contest. A concise answer can win if its claims are correct and sufficient; a detailed answer can win if the detail is relevant, reliable, and improves the user's understanding. If neither answer is clearly more trustworthy and useful, output a tie.

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


SEARCH_ARENA_USER_OUTCOME_BROAD = """Please act as an impartial judge comparing two assistant answers. Choose the answer that creates the better practical outcome for the user.

Think beyond generic correctness. Compare:
1. Need resolution: how much of the user's real problem is solved?
2. User effort saved: does the answer reduce search, inference, verification, or decision work?
3. Usefulness of structure: does the answer present information in a way the user can use immediately?
4. Completeness of decision context: does it include relevant tradeoffs, caveats, alternatives, assumptions, and consequences?
5. Specificity: does it give concrete details where needed instead of vague generalities?
6. Reliability under use: would following or relying on this answer likely work?
7. Proportionality: is the amount of detail appropriate for the query's complexity?
8. Failure cost: does the answer avoid mistakes or omissions that would materially hurt the user?

Prefer the answer with higher practical value, even if it is less polished. Do not reward a response for sounding confident, being longer, or having more citations unless that improves the user's outcome. If both would lead to about the same outcome, output a tie.

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


SEARCH_ARENA_INFORMATION_ARCHITECTURE = """Please act as an impartial judge comparing two assistant answers. Choose the answer that better turns retrieved or known information into a useful response.

Evaluate information architecture and synthesis:
1. Prioritization: puts the most important answer first and separates primary from secondary information.
2. Organization: groups facts logically for the user's task rather than dumping snippets.
3. Synthesis: connects facts, explains relationships, resolves apparent conflicts, and gives a coherent takeaway.
4. Granularity: uses the right level of detail for the question, with examples or numbers when useful.
5. Navigability: makes multi-part answers easy to scan without hiding the answer.
6. Context control: includes enough background to understand the answer but avoids generic or unrelated background.
7. Comparative framing: for comparison questions, uses dimensions that make the difference clear.
8. Evidence integration: incorporates sources or evidence into the answer instead of listing them separately.

Still penalize factual errors and wrong-target answers heavily. But when both answers are broadly factual, prefer the one that better organizes and synthesizes information for the user's purpose. If neither is meaningfully better, output a tie.

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


SEARCH_ARENA_ASSUMPTION_SENSITIVITY = """Please act as an impartial judge comparing two assistant answers. Choose the answer that is more reliable under the assumptions and ambiguities of the user's question.

Focus on dimensions that generic quality rubrics often miss:
1. Ambiguity handling: recognizes multiple interpretations and either resolves them or states assumptions.
2. Assumption exposure: makes important unstated assumptions visible instead of silently relying on them.
3. Boundary conditions: explains when the answer applies and when it may not.
4. Update sensitivity: treats facts that may change over time, by region, or by context with appropriate caution.
5. Exception handling: mentions important exceptions, edge cases, or failure modes when they matter.
6. Overgeneralization risk: avoids turning a narrow fact into a broad rule.
7. User-specific fit: stays sensitive to the user's likely context, constraints, and goal.

Use these dimensions together with factual reliability and relevance. Prefer the answer that would be safer and more useful if the user actually relied on it. If the answers are similarly reliable or similarly flawed, output a tie.

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


SEARCH_ARENA_NATIVE_TASK_TYPE_HYBRID = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Your evaluation should consider factors such as correctness, helpfulness, completeness, accuracy, depth, and level of detail. Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only uses relevant information.

Before deciding, identify the query type and apply the relevant comparison dimensions:
- Fact lookup: exact answer, precise entities/dates/numbers, and enough context to prevent misunderstanding.
- Explanation: conceptual clarity, causal structure, examples, and caveats.
- Comparison: clear dimensions of difference, tradeoffs, and a direct comparative takeaway.
- Recommendation/list: relevance of options, selection criteria, constraints, alternatives, and decision value.
- Current/local/search question: freshness, locality, source-aware phrasing, uncertainty, and verifiable details.
- How-to/troubleshooting: actionable steps, prerequisites, warnings, and likely failure cases.
- Multi-part research: coverage of all subquestions, synthesis, and separation of central facts from secondary detail.

Use these task-specific dimensions as tie-breakers and error detectors, not as a reason to overvalue verbosity. Prefer the response that better answers this user's actual task with relevant, reliable, appropriately detailed information. Do not reward length, citation count, or polish by itself.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible. Lastly, if both responses are citing the same sources of information and offer nearly identical information with minor differences, or if both responses are similarly good or similarly bad, output a tie.

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


SEARCH_ARENA_NATIVE_REQUIREMENT_MATRIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the user question displayed below. Choose the assistant that best answers the user question.

Use a requirement matrix rather than a generic quality checklist:
1. Extract the user's required answer elements: entities, subquestions, constraints, time/location, requested output form, and implied decision need.
2. Compare whether each answer satisfies those required elements.
3. For each required element, consider whether the answer is accurate, specific, supported, current when needed, and easy to use.
4. Then consider optional value: helpful examples, caveats, tradeoffs, synthesis, assumptions, and verification guidance.
5. Penalize off-target coverage, irrelevant detail, unsupported specifics, false precision, and answers that sound good but leave required elements unresolved.

Correctness, helpfulness, completeness, accuracy, depth, and level of detail matter only insofar as they satisfy the extracted requirements. A short answer can win if it satisfies the matrix; a long answer can win if its extra detail satisfies important requirements. If both answers satisfy the requirements similarly well, or fail similarly, output a tie.

Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Be as objective as possible.

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


SEARCH_ARENA_RICH_HUMAN_VOTER = """Please act as a careful human Search Arena voter. Read the user's question and the two assistant answers, then choose the answer you would rather receive as the user.

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


SEARCH_ARENA_RICH_HUMAN_VOTER_RUBRIC = """Please act as a careful human Search Arena voter. Read the user's question, the topic-specific rubric, and the two assistant answers, then choose the answer you would rather receive as the user.

Do not use a generic short-answer rubric. First infer what the user is trying to get done, then judge the answer by the dimensions that matter for that request. Use the topic-specific rubric as evidence about important facts, constraints, and expectations, but do not reduce the decision to a mechanical checklist. In search-result comparisons, a strong answer often wins because it gives more useful answer content, not because it is shorter or more polished.

Consider a broad set of possible dimensions:
- Intent match: answers the exact question, language, location, time frame, and requested scope.
- Rubric alignment: satisfies the mandatory and high-weight rubric criteria, avoids contradicting them, and uses optional criteria as tie-breakers when appropriate.
- Answer density: contains many relevant, non-duplicative pieces of useful information.
- Coverage and recall: includes the important options, entities, subquestions, examples, caveats, and perspectives the user likely needs.
- Specificity: gives concrete names, dates, numbers, prices, addresses, source names, mechanisms, or distinctions when useful.
- Explanation quality: for "how", "why", effects, tradeoffs, or research questions, explains causes, mechanisms, and implications rather than only listing facts.
- Recommendation quality: for "best", "must try", "which", or planning questions, gives useful options, criteria, and practical details.
- Evidence and trust: claims are plausible, grounded, and not misleading; citations or source-derived details help when they support the answer.
- Calibration: handles uncertainty, currentness, locality, and assumptions honestly.
- Organization: makes a rich answer easy to use through grouping, ordering, and clear takeaways.
- Noise control: extra text is a problem only when it is irrelevant, repetitive, unsupported, or distracts from the user's need.

Prefer the answer with greater useful substance for this user's question, especially when it better satisfies mandatory or high-weight rubric criteria. Do not penalize an answer merely for being long if the added material is relevant and useful. Do not reward brevity, citation count, checklist keyword matches, or fluent wording by itself. If both answers would be similarly useful or similarly flawed, distinguish between both-good and both-bad ties.

Do not write explanation or analysis. Output only the final verdict by strictly following this format:
"[[A]]" if Assistant A is better, "[[B]]" if Assistant B is better, "[[Both Good]]" if both responses are similarly strong, and "[[Both Bad]]" if both responses are similarly weak.

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


SEARCH_ARENA_ANSWER_DENSITY_RECALL = """Please act as an impartial judge for a side-by-side search answer comparison. Choose the response that gives the user more useful answer value.

Judge by useful answer units. A useful answer unit is a relevant fact, option, explanation, caveat, comparison point, source-grounded detail, step, example, address, price, date, mechanism, or recommendation that helps answer the user's question. More useful units can make an answer better when they are relevant and organized.

Decision process:
1. Identify what kind of answer the user asked for: direct fact, many examples, local recommendations, explanation, comparison, policy/current event, product choice, research starting point, troubleshooting, or planning.
2. For that task, compare how much relevant useful content each answer provides.
3. Check quality of that content: accuracy, specificity, freshness, source support, and whether it avoids misleading claims.
4. Check whether the answer covers the main subquestions and practical details the user would otherwise need to search for.
5. Penalize irrelevant padding, generic filler, hallucinated specifics, unsupported claims, and disorganized dumps.

A concise answer can win if the query is simple or the longer answer is noisy. But for broad, list-seeking, local, recommendation, or analytical queries, a comprehensive answer with many relevant details should usually beat a thin answer. If neither answer has a clear useful-content advantage, output a tie.

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


SEARCH_ARENA_QUERY_DEMAND_WEIGHTS = """Please act as an impartial judge comparing two assistant answers to the user's question. Choose the answer that best matches the demand of the query.

Before judging, infer the query demand and weight dimensions accordingly:
- If the user asks for "many", examples, papers, restaurants, products, attractions, or recommendations: prioritize breadth, concrete options, categorization, practical details, and coverage.
- If the user asks "how", "why", "effects", "impact", or "what happens": prioritize causal reasoning, mechanisms, tradeoffs, second-order effects, caveats, and synthesis.
- If the user asks a direct fact: prioritize exactness, correctness, and concise context.
- If the user asks a local or current question: prioritize location/time awareness, freshness, addresses/prices/hours when useful, and source-grounded details.
- If the user asks a comparison or choice: prioritize explicit comparison dimensions, pros/cons, decision criteria, and a clear takeaway.
- If the user asks for learning or research help: prioritize structured reading paths, foundational and advanced sources, summaries, and how each item helps.
- If the request is ambiguous: prioritize useful assumptions, clarification of scope, and answers that remain reliable under multiple interpretations.

Across all query types, reject answers that are off target, factually wrong, misleading, unsupported, or padded with irrelevant material. But do not treat length as a defect: relevant depth, many examples, and detailed organization are strengths when the query calls for them. Choose the answer whose strengths fit this specific query demand. If both are close, output a tie.

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


SEARCH_ARENA_REASONING_AND_COVERAGE = """Please act as an impartial judge comparing two search assistant answers. Choose the answer that better helps the user understand, decide, or act.

Use many comparison dimensions, but give special attention to reasoning and coverage:
- Directness: the answer addresses the user's actual request.
- Factual quality: claims are accurate, current when needed, and not misleading.
- Coverage: the answer includes the important entities, subquestions, options, factors, caveats, and examples.
- Causal and analytical reasoning: for impact, finance, policy, science, technical, or "how/why" questions, the answer explains mechanisms and implications.
- Practical utility: for recommendations and local queries, the answer gives usable names, addresses, prices, distinguishing features, and selection guidance.
- Research utility: for literature or learning queries, the answer provides enough papers/resources plus summaries and learning value.
- Synthesis: facts are connected into an explanation or organized set of choices rather than scattered.
- Evidence quality: citations, named sources, dates, or concrete details support the answer; unsupported precision is a weakness.
- Calibration: uncertainty, assumptions, and limitations are stated when they matter.
- Readability: structure makes the answer usable, but style should not outweigh substance.

Reward comprehensive, answer-rich responses when the added detail is relevant and trustworthy. Penalize only useless verbosity, repetition, or irrelevant background. If both answers have similar user value, output a tie.

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


SEARCH_ARENA_RECALL_PLUS_TRUST = """Please act as an impartial judge for Search Arena. Choose the answer that combines high recall of useful information with enough trustworthiness to rely on it.

High recall means the answer captures more of what the user likely wanted: more relevant options, facets, subquestions, examples, mechanisms, constraints, local details, caveats, or follow-up information. Trustworthiness means the content is specific, plausible, sourced or checkable when appropriate, and not misleading.

Use these dimensions as needed:
1. Exact target and scope.
2. Breadth over relevant facets.
3. Depth on the most important facets.
4. Concrete details that reduce the user's need to search again.
5. Source/citation usefulness, not just citation count.
6. Currentness, location fit, and entity precision.
7. Explanation of relationships, tradeoffs, or consequences.
8. Handling of uncertainty, assumptions, and exceptions.
9. Organization that makes a broad answer usable.
10. Absence of irrelevant padding or unsupported claims.

Prefer the higher-recall answer when its extra content is relevant and reasonably trustworthy. Prefer the shorter answer only when it is more accurate, more targeted, or the longer answer adds noise or errors. If the useful trustworthy content is about equal, output a tie.

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


SEARCH_ARENA_RICH_TIE_CALIBRATED = """Please act as a careful human voter comparing two search assistant answers. Choose a winner only when one answer would be meaningfully more useful to the user.

First infer the user's task, then compare across a wide set of relevant dimensions: exact intent fit, constraint following, answer density, breadth of coverage, useful depth, factual correctness, specificity, currentness or locality, source support, reasoning quality, recommendation value, caveats, assumptions, organization, and absence of irrelevant noise.

Important calibration:
- Do not be biased toward short answers. A long answer can be better if it is rich in relevant information.
- Do not be biased toward long answers. A long answer loses if it is repetitive, generic, unsupported, or off target.
- Do not overvalue citation count, formatting, or fluency.
- For broad, list-seeking, recommendation, research, or analytical queries, useful breadth and depth should matter a lot.
- For simple factual queries, exact correctness and directness should matter more.
- If the difference is minor, mostly stylistic, or both answers are similarly incomplete or similarly strong, output [[Tie]].

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is meaningfully better, "[[B]]" if Assistant B is meaningfully better, and "[[Tie]]" for a tie.

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


PROMPTS = {
    "native": ARENA_JUDGE_PROMPT,
    "compact-dimensions": SEARCH_ARENA_COMPACT_DIMENSIONS,
    "human-preference": SEARCH_ARENA_HUMAN_PREFERENCE,
    "search-quality": SEARCH_ARENA_SEARCH_QUALITY,
    "strict-helpful": SEARCH_ARENA_STRICT_HELPFUL,
    "user-outcome": SEARCH_ARENA_USER_OUTCOME,
    "compact-depth": SEARCH_ARENA_COMPACT_DEPTH,
    "compact-calibrated-tie": SEARCH_ARENA_COMPACT_CALIBRATED_TIE,
    "compact-search-reasoning": SEARCH_ARENA_COMPACT_SEARCH_REASONING,
    "native-depth-note": SEARCH_ARENA_NATIVE_WITH_DEPTH_NOTE,
    "context-final-depth": SEARCH_ARENA_CONTEXT_FINAL_DEPTH,
    "actionable-helpful": SEARCH_ARENA_ACTIONABLE_HELPFUL,
    "relevance-gate": SEARCH_ARENA_RELEVANCE_GATE,
    "human-search-vote": SEARCH_ARENA_HUMAN_SEARCH_VOTE,
    "substance-over-style": SEARCH_ARENA_SUBSTANCE_OVER_STYLE,
    "context-final-balanced-tie": SEARCH_ARENA_CONTEXT_FINAL_BALANCED_TIE,
    "context-final-answer-first": SEARCH_ARENA_CONTEXT_FINAL_ANSWER_FIRST,
    "context-final-substance": SEARCH_ARENA_CONTEXT_FINAL_SUBSTANCE,
    "context-final-reasoning-value": SEARCH_ARENA_CONTEXT_FINAL_REASONING_VALUE,
    "compact-grounded-depth": SEARCH_ARENA_COMPACT_GROUNDED_DEPTH,
    "source-trust": SEARCH_ARENA_SOURCE_TRUST,
    "currentness-coverage": SEARCH_ARENA_CURRENTNESS_AND_COVERAGE,
    "compact-depth-source-tiebreak": SEARCH_ARENA_COMPACT_DEPTH_SOURCE_TIEBREAK,
    "native-coverage-first": SEARCH_ARENA_NATIVE_COVERAGE_FIRST,
    "native-partial-answer-penalty": SEARCH_ARENA_NATIVE_PARTIAL_ANSWER_PENALTY,
    "native-detailed-dimensions": SEARCH_ARENA_NATIVE_DETAILED_DIMENSIONS,
    "native-intent-compliance": SEARCH_ARENA_NATIVE_INTENT_COMPLIANCE,
    "native-user-effort": SEARCH_ARENA_NATIVE_USER_EFFORT,
    "native-failure-modes": SEARCH_ARENA_NATIVE_FAILURE_MODES,
    "multiaxis-adaptive": SEARCH_ARENA_MULTIAXIS_ADAPTIVE,
    "task-type-adaptive": SEARCH_ARENA_TASK_TYPE_ADAPTIVE,
    "claim-audit": SEARCH_ARENA_CLAIM_AUDIT,
    "user-outcome-broad": SEARCH_ARENA_USER_OUTCOME_BROAD,
    "information-architecture": SEARCH_ARENA_INFORMATION_ARCHITECTURE,
    "assumption-sensitivity": SEARCH_ARENA_ASSUMPTION_SENSITIVITY,
    "native-task-type-hybrid": SEARCH_ARENA_NATIVE_TASK_TYPE_HYBRID,
    "native-requirement-matrix": SEARCH_ARENA_NATIVE_REQUIREMENT_MATRIX,
    "rich-human-voter": SEARCH_ARENA_RICH_HUMAN_VOTER,
    "rich-human-voter-rubric": SEARCH_ARENA_RICH_HUMAN_VOTER_RUBRIC,
    "answer-density-recall": SEARCH_ARENA_ANSWER_DENSITY_RECALL,
    "query-demand-weights": SEARCH_ARENA_QUERY_DEMAND_WEIGHTS,
    "reasoning-and-coverage": SEARCH_ARENA_REASONING_AND_COVERAGE,
    "recall-plus-trust": SEARCH_ARENA_RECALL_PLUS_TRUST,
    "rich-tie-calibrated": SEARCH_ARENA_RICH_TIE_CALIBRATED,
}


def stable_bucket(*parts: str, mod: int = 10_000) -> int:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod


def text_from_message(message: dict[str, Any]) -> str:
    return str(message.get("content") or "").strip()


def render_transcript(messages: list[dict[str, Any]]) -> tuple[str, str]:
    user_messages = [text_from_message(m) for m in messages if m.get("role") == "user"]
    assistant_messages = [text_from_message(m) for m in messages if m.get("role") == "assistant"]
    query = user_messages[-1] if user_messages else ""
    if len(messages) <= 2 and assistant_messages:
        return query, assistant_messages[-1]
    turns: list[str] = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "message").capitalize()
        turns.append(f"{role} {idx}: {text_from_message(message)}")
    return query, "\n\n".join(turns)


def render_parts(messages: list[dict[str, Any]]) -> tuple[str, str, str]:
    user_messages = [text_from_message(m) for m in messages if m.get("role") == "user"]
    assistant_messages = [text_from_message(m) for m in messages if m.get("role") == "assistant"]
    query = user_messages[-1] if user_messages else ""
    final_answer = assistant_messages[-1] if assistant_messages else ""
    turns = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "message").capitalize()
        turns.append(f"{role} {idx}: {text_from_message(message)}")
    return query, final_answer, "\n\n".join(turns)


def normalize_winner(winner: str) -> str:
    value = winner.strip().lower()
    if value == "model_a":
        return "A"
    if value == "model_b":
        return "B"
    return "Tie"


def verdict_to_preferred(verdict: str, model_a: str, model_b: str) -> str | None:
    if verdict == "A":
        return model_a
    if verdict == "B":
        return model_b
    return None


def task_from_row(
    row: dict[str, Any],
    row_index: int,
    prompt_name: str,
    prompt_template: str,
    rubrics_by_qid: dict[str, RubricRecord] | None = None,
) -> dict[str, Any]:
    query_a, answer_a = render_transcript(row["messages_a"])
    query_b, answer_b = render_transcript(row["messages_b"])
    _, final_answer_a, transcript_a = render_parts(row["messages_a"])
    _, final_answer_b, transcript_b = render_parts(row["messages_b"])
    query = query_a or query_b
    model_a = str(row["model_a"])
    model_b = str(row["model_b"])
    question_id = str(row["question_id"])
    rubric_record = rubrics_by_qid.get(question_id) if rubrics_by_qid is not None else None
    if rubrics_by_qid is not None and rubric_record is None:
        raise ValueError(f"missing rubric for question_id {question_id!r}")
    rubric = format_rubric_for_prompt(rubric_record) if rubric_record is not None else ""
    return {
        "task_id": f"searcharena:{prompt_name}:{row_index}:{question_id}:{model_a}:{model_b}",
        "evaluator": "search-arena",
        "instruction": prompt_template.format(
            query=query,
            rubric=rubric,
            answer_a=answer_a,
            answer_b=answer_b,
            final_answer_a=final_answer_a,
            final_answer_b=final_answer_b,
            transcript_a=transcript_a,
            transcript_b=transcript_b,
        ),
        "pair": [model_a, model_b],
        "model_a": model_a,
        "model_b": model_b,
        "human_winner": str(row["winner"]),
        "human_verdict": normalize_winner(str(row["winner"])),
        "question_id": question_id,
        "row_index": row_index,
        "turn": row.get("turn"),
        "language": row.get("language"),
        "prompt_name": prompt_name,
        "rubric_criteria_count": len(rubric_record.criteria) if rubric_record is not None else None,
        "mandatory_criteria_count": (
            sum(1 for criterion in rubric_record.criteria if str(criterion.get("tier", "")).lower() == "mandatory")
            if rubric_record is not None
            else None
        ),
    }


def load_rows(*, include_self: bool, single_turn_only: bool) -> list[dict[str, Any]]:
    dataset = load_dataset("lmarena-ai/search-arena-v1-7k", split="test")
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(dataset):
        record = dict(row)
        if not include_self and record["model_a"] == record["model_b"]:
            continue
        if single_turn_only and int(record["turn"]) != 1:
            continue
        record["_row_index"] = row_index
        rows.append(record)
    return rows


def split_name(row: dict[str, Any]) -> str:
    bucket = stable_bucket(str(row["_row_index"]), str(row["question_id"]))
    if bucket < 2000:
        return "dev"
    if bucket < 4000:
        return "val"
    return "test"


def sample_rows(rows: list[dict[str, Any]], *, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    pool = [row for row in rows if split_name(row) == split]
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in pool:
        key = tuple(sorted([str(row["model_a"]), str(row["model_b"])]))
        by_pair.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for group in by_pair.values():
        rng.shuffle(group)
    ordered: list[dict[str, Any]] = []
    while len(ordered) < n:
        progressed = False
        for key in sorted(by_pair):
            if by_pair[key]:
                ordered.append(by_pair[key].pop())
                progressed = True
                if len(ordered) >= n:
                    break
        if not progressed:
            break
    rng.shuffle(ordered)
    return ordered


def selected_rows(rows: list[dict[str, Any]], *, split: str, n: int, seed: int) -> list[dict[str, Any]]:
    if split == "all":
        return list(rows)
    return sample_rows(rows, split=split, n=n, seed=seed)


def judgment_from_task(task: dict[str, Any], verdict: str) -> dict[str, Any]:
    model_a, model_b = task["pair"]
    return {
        "status": "completed",
        "pair": [model_a, model_b],
        "judge_verdict": verdict,
        "preferred_run_id": verdict_to_preferred(verdict, model_a, model_b),
    }


def human_judgment_from_task(task: dict[str, Any]) -> dict[str, Any]:
    return judgment_from_task(task, task["human_verdict"])


def score_rows(judgments: list[dict[str, Any]], run_ids: list[str]) -> dict[str, float]:
    return fit_arena_ratings(judgments, run_ids)


def corr(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    kt = kendalltau(xs, ys)
    sp = spearmanr(xs, ys)
    return float(kt.statistic), float(kt.pvalue), float(sp.statistic), float(sp.pvalue)


def verdict_agreement(gpt_verdict: str, human_verdict: str) -> tuple[bool, bool]:
    exact = gpt_verdict == human_verdict
    if human_verdict == "Tie" or gpt_verdict in TIE_VERDICTS:
        decisive = False
    else:
        decisive = gpt_verdict == human_verdict
    return exact, decisive


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row.get("task_id")): row for row in read_jsonl(path)}


def is_usage_limit_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return (
        "usage_limit_reached" in lowered
        or "usage limit has been reached" in lowered
        or "primary-used-percent" in lowered
        or "status_code\":429" in lowered
    )


async def run_prompt_set(
    *,
    prompt_name: str,
    prompt_template: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    config: LocalAgentConfig,
    concurrency: int,
    overwrite: bool,
    stop_on_usage_limit: bool,
    rubrics_by_qid: dict[str, RubricRecord] | None = None,
) -> list[dict[str, Any]]:
    output_path = output_dir / prompt_name / "judgments.jsonl"
    failed_path = output_dir / prompt_name / "failed.jsonl"
    raw_events_dir = output_dir / prompt_name / "raw-events"
    if overwrite:
        output_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_existing(output_path)
    tasks = [
        task_from_row(row, int(row["_row_index"]), prompt_name, prompt_template, rubrics_by_qid=rubrics_by_qid)
        for row in rows
    ]
    pending = [task for task in tasks if task["task_id"] not in existing]
    print(f"[{prompt_name}] tasks={len(tasks)} pending={len(pending)} output={output_path}", flush=True)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    stop_event = asyncio.Event()

    async def one(task: dict[str, Any]) -> None:
        async with semaphore:
            if stop_event.is_set():
                return
            result = await run_prompt(
                task_id=task["task_id"],
                evaluator="search-arena",
                instruction=task["instruction"],
                raw_events_dir=raw_events_dir,
                config=config,
                metadata={
                    "prompt_name": prompt_name,
                    "question_id": task["question_id"],
                    "row_index": task["row_index"],
                    "turn": task["turn"],
                    "language": task["language"],
                    "human_winner": task["human_winner"],
                    "rubric_criteria_count": task["rubric_criteria_count"],
                    "mandatory_criteria_count": task["mandatory_criteria_count"],
                },
            )
            verdict = parse_verdict(result["output_text"].strip())
            row = {
                **result,
                "pair": task["pair"],
                "model_a": task["model_a"],
                "model_b": task["model_b"],
                "question_id": task["question_id"],
                "row_index": task["row_index"],
                "turn": task["turn"],
                "language": task["language"],
                "prompt_name": prompt_name,
                "rubric_criteria_count": task["rubric_criteria_count"],
                "mandatory_criteria_count": task["mandatory_criteria_count"],
                "human_winner": task["human_winner"],
                "human_verdict": task["human_verdict"],
                "judge_verdict": verdict,
                "preferred_run_id": verdict_to_preferred(verdict, task["model_a"], task["model_b"]) if verdict else None,
            }
            if verdict is None:
                row["status"] = "failed"
                row["error"] = row["error"] or "could not parse verdict"
            else:
                row["status"] = "completed"
                if result["status"] != "completed":
                    row["error"] = f"salvaged verdict after runner error: {row['error']}"
            async with lock:
                append_jsonl(output_path if row["status"] == "completed" else failed_path, row)
                if stop_on_usage_limit and is_usage_limit_error(row.get("error")):
                    stop_event.set()
            print(f"[{prompt_name}] {row['status']} {task['task_id']} verdict={verdict}", flush=True)
            if stop_event.is_set() and row["status"] == "failed":
                print(f"[{prompt_name}] stopping early after usage-limit error; rerun the same command to resume", flush=True)

    if pending:
        await asyncio.gather(*(asyncio.create_task(one(task)) for task in pending))
    return list(read_existing(output_path).values())


def summarize_prompt(
    *,
    prompt_name: str,
    result_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    run_ids: list[str],
    full_human_scores: dict[str, float],
    output_dir: Path,
) -> dict[str, Any]:
    task_by_id = {task["task_id"]: task for task in tasks}
    completed = [row for row in result_rows if row.get("status") == "completed" and row.get("judge_verdict")]
    common_tasks = [task_by_id[str(row["task_id"])] for row in completed if str(row["task_id"]) in task_by_id]
    human_judgments = [human_judgment_from_task(task) for task in common_tasks]
    gpt_judgments = [
        judgment_from_task(task_by_id[str(row["task_id"])], str(row["judge_verdict"]))
        for row in completed
        if str(row["task_id"]) in task_by_id
    ]
    human_scores = score_rows(human_judgments, run_ids)
    gpt_scores = score_rows(gpt_judgments, run_ids)
    xs = [gpt_scores[run_id] for run_id in run_ids]
    ys_sample = [human_scores[run_id] for run_id in run_ids]
    ys_full = [full_human_scores[run_id] for run_id in run_ids]
    tau_sample, tau_sample_p, spear_sample, spear_sample_p = corr(xs, ys_sample)
    tau_full, tau_full_p, spear_full, spear_full_p = corr(xs, ys_full)

    exact_count = 0
    decisive_total = 0
    decisive_count = 0
    gpt_ties = 0
    human_ties = 0
    for row in completed:
        gpt_verdict = str(row["judge_verdict"])
        human_verdict = str(row["human_verdict"])
        exact, decisive = verdict_agreement(gpt_verdict, human_verdict)
        exact_count += int(exact)
        if human_verdict != "Tie" and gpt_verdict not in TIE_VERDICTS:
            decisive_total += 1
            decisive_count += int(decisive)
        gpt_ties += int(gpt_verdict in TIE_VERDICTS)
        human_ties += int(human_verdict == "Tie")

    leaderboard_path = output_dir / prompt_name / "leaderboard.csv"
    with leaderboard_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "run_id", "arena_score", "n_judgments", "wins", "losses", "ties"])
        writer.writeheader()
        writer.writerows(leaderboard_rows(gpt_judgments, run_ids))

    return {
        "prompt": prompt_name,
        "n_tasks": len(tasks),
        "n_completed": len(completed),
        "exact_agreement": exact_count / len(completed) if completed else math.nan,
        "decisive_agreement": decisive_count / decisive_total if decisive_total else math.nan,
        "gpt_tie_rate": gpt_ties / len(completed) if completed else math.nan,
        "human_tie_rate": human_ties / len(completed) if completed else math.nan,
        "kendall_tau_vs_sample_human_bt": tau_sample,
        "kendall_p_vs_sample_human_bt": tau_sample_p,
        "spearman_vs_sample_human_bt": spear_sample,
        "spearman_p_vs_sample_human_bt": spear_sample_p,
        "kendall_tau_vs_full_human_bt": tau_full,
        "kendall_p_vs_full_human_bt": tau_full_p,
        "spearman_vs_full_human_bt": spear_full,
        "spearman_p_vs_full_human_bt": spear_full_p,
    }


async def main_async(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(include_self=args.include_self, single_turn_only=args.single_turn_only)
    rubrics_by_qid = load_rubrics(args.rubric_file) if args.rubric_file is not None else None
    rows_before_rubric_filter = len(rows)
    if rubrics_by_qid is not None:
        rows = [row for row in rows if str(row["question_id"]) in rubrics_by_qid]
        print(
            f"[rubrics] covered_rows={len(rows)}/{rows_before_rubric_filter} "
            f"rubrics={len(rubrics_by_qid)} source={args.rubric_file}",
            flush=True,
        )
    sample = selected_rows(rows, split=args.split, n=args.sample_size, seed=args.seed)
    all_models = sorted({str(row["model_a"]) for row in rows} | {str(row["model_b"]) for row in rows})
    full_human_tasks = [
        task_from_row(row, int(row["_row_index"]), "human", "{query}\n\n{answer_a}\n\n{answer_b}")
        for row in load_rows(include_self=args.include_self, single_turn_only=args.single_turn_only)
    ]
    all_models = sorted({str(task["model_a"]) for task in full_human_tasks} | {str(task["model_b"]) for task in full_human_tasks})
    full_human_scores = score_rows([human_judgment_from_task(task) for task in full_human_tasks], all_models)

    sample_manifest = (
        output_dir / "all-rows.jsonl"
        if args.split == "all"
        else output_dir / f"{args.split}-sample-{args.sample_size}-seed{args.seed}.jsonl"
    )
    with sample_manifest.open("w", encoding="utf-8") as handle:
        for row in sample:
            handle.write(json.dumps({
                "row_index": row["_row_index"],
                "question_id": row["question_id"],
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "winner": row["winner"],
                "turn": row["turn"],
                "language": row["language"],
                "split": args.split,
                "has_rubric": rubrics_by_qid is not None,
                "rubric_criteria_count": (
                    len(rubrics_by_qid[str(row["question_id"])].criteria) if rubrics_by_qid is not None else None
                ),
                "mandatory_criteria_count": (
                    sum(
                        1
                        for criterion in rubrics_by_qid[str(row["question_id"])].criteria
                        if str(criterion.get("tier", "")).lower() == "mandatory"
                    )
                    if rubrics_by_qid is not None
                    else None
                ),
            }, ensure_ascii=False) + "\n")

    config = LocalAgentConfig(
        agent_binary=args.agent_binary,
        model=args.model,
        thinking=args.thinking,
        timeout_seconds=args.timeout_seconds,
        cache_dir=args.cache_dir,
    )

    selected_prompts = args.prompts or list(PROMPTS)
    summary_rows: list[dict[str, Any]] = []
    for prompt_name in selected_prompts:
        if prompt_name not in PROMPTS:
            raise SystemExit(f"unknown prompt {prompt_name!r}; options: {', '.join(PROMPTS)}")
        prompt_template = PROMPTS[prompt_name]
        result_rows = await run_prompt_set(
            prompt_name=prompt_name,
            prompt_template=prompt_template,
            rows=sample,
            output_dir=output_dir,
            config=replace(config, cache_dir=args.cache_dir / prompt_name if args.cache_dir else None),
            concurrency=args.concurrency,
            overwrite=args.overwrite,
            stop_on_usage_limit=args.stop_on_usage_limit,
            rubrics_by_qid=rubrics_by_qid,
        )
        tasks = [
            task_from_row(
                row,
                int(row["_row_index"]),
                prompt_name,
                prompt_template,
                rubrics_by_qid=rubrics_by_qid,
            )
            for row in sample
        ]
        summary_rows.append(
            summarize_prompt(
                prompt_name=prompt_name,
                result_rows=result_rows,
                tasks=tasks,
                run_ids=all_models,
                full_human_scores=full_human_scores,
                output_dir=output_dir,
            )
        )
        write_summary(output_dir / "summary.csv", summary_rows)
        print_summary(summary_rows[-1])

    write_summary(output_dir / "summary.csv", summary_rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(row: dict[str, Any]) -> None:
    print(
        "[summary] {prompt}: n={n_completed}/{n_tasks} "
        "tau_sample={kendall_tau_vs_sample_human_bt:.6f} "
        "tau_full={kendall_tau_vs_full_human_bt:.6f} "
        "exact={exact_agreement:.4f} ties={gpt_tie_rate:.4f}".format(**row),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-search Search Arena judgments against human pairwise votes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/search-arena-prompt-search"))
    parser.add_argument("--split", choices=["dev", "val", "test", "all"], default="dev")
    parser.add_argument("--sample-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--model", default="openai-codex/gpt-5.5")
    parser.add_argument("--thinking", default="high")
    parser.add_argument("--agent-binary", default="pi")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/ragdoll/search-arena-prompt-search"))
    parser.add_argument("--prompt", dest="prompts", action="append", choices=sorted(PROMPTS))
    parser.add_argument("--rubric-file", type=Path)
    parser.add_argument("--single-turn-only", action="store_true")
    parser.add_argument("--include-self", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-usage-limit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
