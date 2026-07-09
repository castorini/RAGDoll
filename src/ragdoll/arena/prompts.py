from __future__ import annotations

import re

PAIRWISE_ANSWER_COMPARISON_NAIVE = """You are judging two assistant answers to the same user question. Read the user's question and both answers carefully, infer what the user is trying to accomplish, and choose the answer the user would rather receive.

This is a preference judgment, not a checklist. Judge each answer by the qualities that matter for this specific request. Prefer the answer that is more useful, better matched to the user's intent, more complete where completeness matters, and more trustworthy.

Do not apply a generic preference for short answers, long answers, polished wording, or rigid formatting. A longer answer can be better when the added content is relevant and useful. A shorter answer can be better when it answers the user's need directly without omitting important information.

Consider the following dimensions when relevant:

- Intent and style match: Does the answer address the user's actual request, including the requested format, tone, style, scope, technical level, language, and any explicit constraints?
- Directness: Does the answer actually answer the question, rather than giving generic background, evasive caveats, or irrelevant information?
- Usefulness: Does the answer give information the user can act on, apply, or learn from?
- Completeness: Does the answer cover the important parts of the question without omitting key details the user likely needs?
- Specificity: Does the answer provide concrete details, distinctions, examples, names, numbers, steps, or explanations when they would help?
- Accuracy and plausibility: Are the claims plausible, internally consistent, and not misleading? Does the answer avoid obvious factual errors, contradictions, fabricated-sounding details, or overconfident claims?
- Calibration: Does the answer handle uncertainty, assumptions, limitations, and missing information honestly?
- Explanation quality: For questions asking how, why, or what something means, does the answer explain mechanisms, reasoning, implications, and tradeoffs rather than only listing facts?
- Recommendation quality: For questions asking what to choose, use, try, or do, does the answer give practical criteria and well-motivated recommendations?
- Organization: Is the answer easy to use, with helpful structure, ordering, grouping, prioritization, or takeaways?
- Noise control: Does the answer avoid irrelevant, repetitive, distracting, or filler content?

Prefer the answer with greater useful substance for the user's actual need. Do not reward brevity, fluency, confidence, or formatting by itself.

Choose "[[Tie]]" when neither answer is meaningfully preferable and both answers are at least acceptable. Choose "[[Tie (bothbad)]]" when neither answer is meaningfully preferable because both answers are bad, unusable, substantially incorrect, unsafe, evasive, or fail the user's request in a similar way.

Do not choose a tie merely because both answers have strengths or both have weaknesses. If one answer is meaningfully more useful, accurate, complete, or better matched to the user's intent, choose that answer, even if both answers are imperfect.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better,
"[[B]]" if Assistant B is better,
"[[Tie]]" if they are effectively tied and both are at least acceptable,
"[[Tie (bothbad)]]" if they are effectively tied because both are bad.

Do not include any explanation, reasoning, or additional text outside the verdict.

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

PAIRWISE_ANSWER_COMPARISON_W_NUGGET_RUBRICS = """You are judging two assistant answers to the same user question. Read the user's question, the nugget rubric, and both answers carefully, infer what the user is trying to accomplish, and choose the answer the user would rather receive.

This is a preference judgment, not a checklist. Judge each answer by the qualities that matter for this specific request. Prefer the answer that is more useful, better matched to the user's intent, more complete where completeness matters, and more trustworthy.

Do not apply a generic preference for short answers, long answers, polished wording, or rigid formatting. A longer answer can be better when the added content is relevant and useful. A shorter answer can be better when it answers the user's need directly without omitting important information.

Consider the following dimensions when relevant:

- Intent and style match: Does the answer address the user's actual request, including the requested format, tone, style, scope, technical level, language, and any explicit constraints?
- Directness: Does the answer actually answer the question, rather than giving generic background, evasive caveats, or irrelevant information?
- Usefulness: Does the answer give information the user can act on, apply, or learn from?
- Completeness: Does the answer cover the important parts of the question without omitting key details the user likely needs?
- Specificity: Does the answer provide concrete details, distinctions, examples, names, numbers, steps, or explanations when they would help?
- Accuracy and plausibility: Are the claims plausible, internally consistent, and not misleading? Does the answer avoid obvious factual errors, contradictions, fabricated-sounding details, or overconfident claims?
- Calibration: Does the answer handle uncertainty, assumptions, limitations, and missing information honestly?
- Explanation quality: For questions asking how, why, or what something means, does the answer explain mechanisms, reasoning, implications, and tradeoffs rather than only listing facts?
- Recommendation quality: For questions asking what to choose, use, try, or do, does the answer give practical criteria and well-motivated recommendations?
- Organization: Is the answer easy to use, with helpful structure, ordering, grouping, prioritization, or takeaways?
- Noise control: Does the answer avoid irrelevant, repetitive, distracting, or filler content?

Use the nugget rubric to inform your judgment on any of the above answer qualities it captures, taking into account how its items are described, categorized, and prioritized, but do not treat it as a rigid checklist or scorecard. The user's actual question and information need still matter most. Credit semantically equivalent satisfaction of rubric items, and only credit rubric items that are addressed accurately and relevantly.

Prefer the answer with greater useful substance for the user's actual need. Do not reward brevity, fluency, confidence, or formatting by itself.

Choose "[[Tie]]" when neither answer is meaningfully preferable and both answers are at least acceptable. Choose "[[Tie (bothbad)]]" when neither answer is meaningfully preferable because both answers are bad, unusable, substantially incorrect, unsafe, evasive, or fail the user's request in a similar way.

Do not choose a tie merely because both answers have strengths or both have weaknesses. If one answer is meaningfully more useful, accurate, complete, or better matched to the user's intent, choose that answer, even if both answers are imperfect.

Output your final verdict by strictly following this format:
"[[A]]" if Assistant A is better,
"[[B]]" if Assistant B is better,
"[[Tie]]" if they are effectively tied and both are at least acceptable,
"[[Tie (bothbad)]]" if they are effectively tied because both are bad.

Do not include any explanation, reasoning, or additional text outside the verdict.

[The Start of User's Question]
{query}
[The End of User's Question]

[The Start of Nugget Rubric]
{rubric}
[The End of Nugget Rubric]

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
"""

TIE_VERDICTS = frozenset({"Tie", "Tie (bothbad)"})

_VERDICT_RE = re.compile(r"\s*\[\[(A|B|Tie|Tie \(bothbad\))\]\]\s*")


def render_arena_prompt(
    *,
    query: str,
    answer_a: str,
    answer_b: str,
    rubric: str | None = None,
) -> str:
    if rubric is not None:
        return PAIRWISE_ANSWER_COMPARISON_W_NUGGET_RUBRICS.format(
            query=query,
            answer_a=answer_a,
            answer_b=answer_b,
            rubric=rubric,
        )
    return PAIRWISE_ANSWER_COMPARISON_NAIVE.format(query=query, answer_a=answer_a, answer_b=answer_b)


def parse_verdict(text: str) -> str | None:
    match = _VERDICT_RE.fullmatch(text)
    return match.group(1) if match else None
