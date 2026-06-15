"""Prompt templates copied from upstream evaluators with provenance."""

from __future__ import annotations

import ast
import re
from typing import Any

UMBRELA_ZERO_BASIC = """You are an expert judge of a content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevance category to the query.
Here, "0" represent that the passage has nothing to do with the query, "1" represents that the passage seems related to the query but does not answer it, "2" represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and "3" represents that the passage is dedicated to the query and contains the exact answer.

Provide explanation for the relevance and give your answer with from one of the categories 0, 1, 2 or 3 only. One of the categorical values if compulsory in answer.

Instructions: Think about the question. After explaining your reasoning, provide your answer in terms of 0, 1, 2 or 3 category. Only provide the relevance category on the last line. Do not provide any further details on the last line.

###

Query: {query}
Passage: {passage}

Explanation:"""

UMBRELA_ZERO_BING = """Given a query and a passage, you must provide a score on an integer scale of 0 to 3 with the following meanings:
0 = represent that the passage has nothing to do with the query, 
1 = represents that the passage seems related to the query but does not answer it, 
2 = represents that the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information and 
3 = represents that the passage is dedicated to the query and contains the exact answer.

Important Instruction: Assign category 1 if the passage is somewhat related to the topic but not completely, category 2 if passage presents something very important related to the entire topic but also has some extra information and category 3 if the passage only and entirely refers to the topic. If none of the above satisfies give it category 0.

Query: {query}
Passage: {passage}

Split this problem into steps:
Consider the underlying intent of the search.
Measure how well the content matches a likely intent of the query (M).
Measure how trustworthy the passage is (T).
Consider the aspects above and the relative importance of each, and decide on a final score (O). Final score must be an integer value only.
Do not provide any code in result. Provide each score in the format of: ##final score: score without providing any reasoning."""

NUGGET_CREATOR_SYSTEM = "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
NUGGET_CREATOR_USER = """Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.

Search Query: {query}
Context:
{context}
Search Query: {query}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {nuggets_length}

Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""

NUGGET_SCORER_SYSTEM = "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
NUGGET_SCORER_USER = """Based on the query, label each of the {num_nuggets} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Nugget List: {nuggets}

Only return the list of labels (List[str]). Do not explain.
Labels:"""

PI_SEARCH_SYSTEM_PROMPT = """You are a retrieval agent operating inside Pi for evidence-grounded mining tasks. Use only the available retrieval tools and the user's task instructions.

Available tools:
- search: Always supply reason first, under 100 words. Use query for a concise raw search string based on the original wording or one grounded refinement. The tool returns a search_id plus the first page of results.
- read_search_results: Always supply reason first, with a brief rationale of at most 100 words. Then read a cached search result set by search_id in paginated ranked-hit chunks using offset and limit.
- read_document: Always supply reason first, with a brief rationale of at most 100 words. Then read a retrieved document by docid in paginated line-based chunks using offset and limit.

Guidelines:
- Use search for short lexical queries; start close to the task wording, then make grounded refinements only after browsing or reading.
- Use read_search_results to browse deeper ranks from an existing search result set before issuing another search.
- Use read_document to verify evidence from specific document ids before producing grounded output.
- If a document is truncated and still looks relevant, continue reading the same document with the suggested next offset before launching many new searches.
- Keep tool-call reasons specific and under 100 words."""

NUGGET_AGENTIC_CREATOR_SYSTEM = NUGGET_CREATOR_SYSTEM + "\n\n" + PI_SEARCH_SYSTEM_PROMPT

NUGGET_AGENTIC_CREATOR_USER = """Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide all the information required for the query. Leverage only the initial list of nuggets (if exists) and evidence you retrieve with the available search and read_document tools. Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.

Search Query: {query}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {nuggets_length}

Search the corpus for evidence relevant to the search query, read the most useful documents, and iteratively update the initial nugget list as you gather evidence. Only use retrieved evidence and the initial nugget list. Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""

NUGGET_ASSIGNER_SYSTEM = "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."
NUGGET_ASSIGNER_USER = """Based on the query and passage, label each of the {num_nuggets} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {context}
Nugget List: {nuggets}
Only return the list of labels (List[str]). Do not explain.
Labels:"""

NUGGET_ASSIGNER_2GRADE_USER = """Based on the query and passage, label each of the {num_nuggets} nuggets either as support or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support; otherwise, label them as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {context}
Nugget List: {nuggets}
Only return the list of labels (List[str]). Do not explain.
Labels:"""

SUPPORT_EVAL_PROMPT = """
In this task, you will evaluate whether each statement is supported by its corresponding citations. 
Note that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are not easy to discern at first glance. 
Pay close attention to the text. Read it carefully as you would when proofreading.

You will be provided with a statement and its corresponding citation. It may be helpful to ask yourself whether it is accurate to say "according to the citation" with a
statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

- "Full Support": All of the information in the statement is supported in the citation.
- "Partial Support": Only some of the information is supported in the citation, but other parts of the information are missing from the citation.
- "No Support": This citation does not support any part of the statement.

Please provide your response based on the information in the citation. If you are unsure, use your best judgment. 
Respond as either "Full Support", "Partial Support", or "No Support" with no additional information.

Statement: {statement}

Citation: {citation}

Response:
"""


def render_umbrela_prompt(*, query: str, passage: str, prompt_type: str) -> str:
    if prompt_type == "basic":
        return UMBRELA_ZERO_BASIC.format(query=query, passage=passage)
    if prompt_type == "bing":
        return UMBRELA_ZERO_BING.format(query=query, passage=passage)
    raise ValueError(f"unsupported UMBRELA prompt type: {prompt_type}")


def parse_label_list(text: str) -> list[str] | None:
    text = text.strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        match = re.search(r"\[[^\]]*\]", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def parse_umbrela_judgment(text: str) -> int | None:
    patterns = [
        r"##\s*final score\s*:\s*([0-3])",
        r"final score\s*:\s*([0-3])",
        r"relevance category\s*[:-=]?\s*([0-3])",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return int(matches[-1])
    stripped = text.strip()
    if stripped and stripped[-1] in "0123":
        return int(stripped[-1])
    return None


def parse_support_label(text: str) -> str | None:
    lowered = text.lower()
    if "full support" in lowered:
        return "FS"
    if "partial support" in lowered:
        return "PS"
    if "no support" in lowered:
        return "NS"
    return None


def list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in value]
    return []
