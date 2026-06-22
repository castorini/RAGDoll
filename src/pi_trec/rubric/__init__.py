from __future__ import annotations

from typing import NoReturn


def _not_implemented() -> NoReturn:
    raise NotImplementedError("rubric commands are not implemented on this branch")


async def author(config: object) -> NoReturn:
    _not_implemented()


async def grade(config: object) -> NoReturn:
    _not_implemented()


def compute_scores(config: object) -> NoReturn:
    _not_implemented()


async def rubric_eval_pipeline(config: object) -> NoReturn:
    _not_implemented()
