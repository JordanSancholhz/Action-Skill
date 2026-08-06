"""Prompt templates for Search-QA/Search-R1 style environments."""

SEARCH_TEMPLATE_NO_HIS = """
You are an expert search agent answering a factual question.
Your question is: {task_description}

Reason carefully, then choose exactly one action:
1. If information is missing or uncertain, issue one query using
   <search>your query</search>.
2. If you have enough evidence, answer using
   <answer>your final answer</answer>.

Do not emit both actions in the same response.
"""


SEARCH_TEMPLATE = """
You are an expert search agent answering a factual question.
Your question is: {task_description}

You have already taken {step_count} step(s). The recent interaction history is:
{memory_context}

Reason carefully, then choose exactly one action:
1. If information is missing or uncertain, issue one query using
   <search>your query</search>.
2. If you have enough evidence, answer using
   <answer>your final answer</answer>.

Do not emit both actions in the same response.
"""


SEARCH_TEMPLATE_WITH_MEMORY_NO_HIS = """
You are an expert search agent answering a factual question.
Your question is: {task_description}

## Retrieved Search Skills

{retrieved_memories}

Reason carefully, then choose exactly one action:
1. If information is missing or uncertain, issue one query using
   <search>your query</search>.
2. If you have enough evidence, answer using
   <answer>your final answer</answer>.

Do not emit both actions in the same response.
"""


SEARCH_TEMPLATE_WITH_MEMORY = """
You are an expert search agent answering a factual question.
Your question is: {task_description}

## Retrieved Search Skills

{retrieved_memories}

## Current Progress

You have already taken {step_count} step(s). The recent interaction history is:
{memory_context}

Reason carefully, then choose exactly one action:
1. If information is missing or uncertain, issue one query using
   <search>your query</search>.
2. If you have enough evidence, answer using
   <answer>your final answer</answer>.

Do not emit both actions in the same response.
"""
