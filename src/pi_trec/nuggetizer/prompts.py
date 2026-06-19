from __future__ import annotations

NUGGET_CREATOR_SYSTEM = "\
You are NuggetizeLLM, an intelligent assistant that can update a list of \
atomic nuggets to best provide all the information required for the query."


NUGGET_CREATOR_USER = """\
Update the list of atomic nuggets of information (1-12 words), \
if needed, so they best provide the information required for the query. \
Leverage only the initial list of nuggets (if exists) and the provided context \
(this is an iterative process). Return only the final list of all nuggets in a \
Pythonic list format (even if no updates). Make sure there is no redundant information. \
Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), \
keeping only the most vital ones. Order them in decreasing order of importance. \
Prefer nuggets that provide more interesting information.

Search Query: {query}
Context:
{context}
Search Query: {query}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {nuggets_length}

Only update the list of atomic nuggets (if needed, else return as is). \
Do not explain. Always answer in short nuggets (not questions). \
List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""


NUGGET_SCORER_SYSTEM = "\
You are NuggetizeScoreLLM, an intelligent assistant that can label a \
list of atomic nuggets based on their importance for a given search query."


NUGGET_SCORER_USER = """\
Based on the query, label each of the {num_nuggets} nuggets either a vital or okay based on the following criteria. \
Vital nuggets represent concepts that must be present in a "good" answer; on the other hand, okay nuggets contribute \
worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format \
(type: List[str]). The list should be in the same order as the input nuggets. \
Make sure to provide a label for each nugget.

Search Query: {query}
Nugget List: {nuggets}

Only return the list of labels (List[str]). Do not explain.
Labels:"""


PI_SEARCH_SYSTEM_PROMPT = """\
You are a retrieval agent operating inside Pi for evidence-grounded mining tasks. \
Use only the available retrieval tools and the user's task instructions.

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


NUGGET_AGENTIC_CREATOR_USER = """\
Update the list of atomic nuggets of information (1-12 words), if needed, \
so they best provide all the information required for the query. \
Leverage only the initial list of nuggets (if exists) and evidence \
you retrieve with the available search and read_document tools. \
Return only the final list of all nuggets in a Pythonic list format \
(even if no updates). Make sure there is no redundant information. \
Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), \
keeping only the most vital ones. Order them in decreasing order of importance. \
Prefer nuggets that provide more interesting information.

Search Query: {query}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {nuggets_length}

Search the corpus for evidence relevant to the search query, read the most useful documents, \
and iteratively update the initial nugget list as you gather evidence. Only use retrieved evidence \
and the initial nugget list. Do not explain. Always answer in short nuggets (not questions). \
List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""


NUGGET_ASSIGNER_SYSTEM = "\
You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of \
atomic nuggets based on if they are captured by a given passage."


NUGGET_ASSIGNER_USER = """\
Based on the query and passage, label each of the {num_nuggets} nuggets either as \
support, partial_support, or not_support using the following criteria. \
A nugget that is fully captured in the passage should be labeled as support. \
A nugget that is partially captured in the passage should be labeled as partial_support. \
If the nugget is not captured at all, label it as not_support. Return the list of labels in a \
Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. \
Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {context}
Nugget List: {nuggets}
Only return the list of labels (List[str]). Do not explain.
Labels:"""


NUGGET_ASSIGNER_2GRADE_USER = """\
Based on the query and passage, label each of the {num_nuggets} nuggets either as \
support or not_support using the following criteria. A nugget that is fully captured in the passage \
should be labeled as support; otherwise, label them as not_support. Return the list of labels in a \
Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. \
Make sure to provide a label for each nugget.

Search Query: {query}
Passage: {context}
Nugget List: {nuggets}
Only return the list of labels (List[str]). Do not explain.
Labels:"""


def render_create_prompt(
    *,
    query: str,
    context: str,
    nuggets: list[str] | None = None,
    creator_max_nuggets: int = 30,
) -> str:
    initial = nuggets or []
    user = NUGGET_CREATOR_USER.format(
        query=query,
        context=context,
        nuggets=initial,
        nuggets_length=len(initial),
        creator_max_nuggets=creator_max_nuggets,
    )
    return user


def render_agentic_create_prompt(
    *,
    query: str,
    nuggets: list[str] | None = None,
    creator_max_nuggets: int = 30,
) -> str:
    initial = nuggets or []
    return NUGGET_AGENTIC_CREATOR_USER.format(
        query=query,
        nuggets=initial,
        nuggets_length=len(initial),
        creator_max_nuggets=creator_max_nuggets,
    )


def render_score_prompt(*, query: str, nuggets: list[str]) -> str:
    return NUGGET_SCORER_USER.format(query=query, nuggets=nuggets, num_nuggets=len(nuggets))


def render_assign_prompt(
    *,
    query: str,
    context: str,
    nuggets: list[str],
    assign_mode: str,
) -> str:
    template = NUGGET_ASSIGNER_2GRADE_USER if assign_mode == "support-grade-2" else NUGGET_ASSIGNER_USER
    return template.format(query=query, context=context, nuggets=nuggets, num_nuggets=len(nuggets))


def clean_response(text: str) -> str:
    """Strip Markdown code fences and trim (mirrors the reference `_clean_response`)."""
    return text.replace("```python", "").replace("```", "").strip()


_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "0": "\0"}


def parse_label_list(text: str) -> list[str] | None:
    """Extract a Pythonic list of strings from a model response.

    Faithful port of the reference nuggetizer parser: locate the outermost
    ``[ ... ]``, then tolerate single- or double-quoted items, bare words, and
    surrounding prose/code fences. Returns ``None`` when no list can be located.
    """
    cleaned = clean_response(text)
    open_index = cleaned.find("[")
    close_index = cleaned.rfind("]")
    if open_index == -1 or close_index == -1 or close_index <= open_index:
        return None

    body = cleaned[open_index + 1 : close_index]
    items: list[str] = []
    i, n = 0, len(body)
    while i < n:
        # Skip separators / whitespace between items.
        while i < n and (body[i] == "," or body[i].isspace()):
            i += 1
        if i >= n:
            break
        char = body[i]
        if char in ("'", '"'):
            quote = char
            i += 1
            buf: list[str] = []
            closed = False
            while i < n:
                current = body[i]
                if current == "\\" and i + 1 < n:
                    buf.append(_ESCAPES.get(body[i + 1], body[i + 1]))
                    i += 2
                    continue
                if current == quote:
                    i += 1
                    closed = True
                    break
                buf.append(current)
                i += 1
            if not closed:
                return None  # unterminated string => malformed list
            items.append("".join(buf))
        else:
            # Bare word (e.g. an unquoted label) up to the next comma.
            buf = []
            while i < n and body[i] != ",":
                buf.append(body[i])
                i += 1
            trimmed = "".join(buf).strip()
            if trimmed:
                items.append(trimmed)
    return items
