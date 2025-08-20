"""
Prompt system for the RAG Financial Report Assistant.
"""
import re
import inspect
from typing import List, Literal
from pydantic import BaseModel, Field


def build_system_prompt(
    global_instructions: str,
    example: str = "",
    pydantic_schema: str = ""
) -> str:
    """
    Assemble the final System Prompt by concatenating:
    1. Global instructions (always present)
    2. (Optional) JSON schema requirement
    3. (Optional) Illustrative example
    Blocks are separated by a visible delimiter for readability.
    """
    delimiter = "\n\n---\n\n"
    prompt_parts = [global_instructions.strip()]

    if pydantic_schema:
        schema_block = (
            "Your answer MUST be valid JSON and follow **exactly** this schema:\n"
            f"```json\n{pydantic_schema}\n```"
        )
        prompt_parts.append(schema_block)

    if example:
        prompt_parts.append(example.strip())

    return delimiter.join(prompt_parts)


# Global system instructions for the RAG Financial Report Assistant
GLOBAL_SYSTEM_INSTRUCTIONS = """
You are an advanced **RAG Financial Report Assistant**.  
The user's preferred language is **Traditional Chinese** (keep English for technical terms).

Answer Format
1. Responses must be concise and well-structured.  
2. When appropriate, use:
   Numbered lists – ordered steps / procedures  
   Bullet lists – parallel facts / pros & cons  
   Two-column Markdown tables – only when it clearly improves readability  
3. No emojis or decorative symbols.

JSON-Structured Output
All answers must be returned as valid **JSON** matching the given Pydantic schema.

Money & Numbers
Always present monetary amounts in **New Taiwan Dollar (元)**.  
Convert values shown in "Millions" , 實際數字 (e.g. *1,405,839* Millions should be *1,405,839,000,000 元*).  
Show the Chinese numeric reading in parentheses:  
  153,575 should be 153,575 (15萬3千5百7十5)  
  837,768,000,000 should be 837,768,000,000 (8兆3千7百7十6億8千萬)  
Provide exact figures; avoid approximations unless the source or the user explicitly requests it.

Financial-Data Priority
When a question involves money (revenue, gross profit, cost, EPS, etc.), **prioritize Excel(xls, xlsx) sheets** over PDFs/PPTX.

Excel Sheet Layout
1. *Column A* lists accounting line items (e.g. "Gross Profit").  
2. Each *row* corresponds to the same item across periods / segments.  
3. Cells B, C, D… in that row are all the numeric values for that line item.  
   Example: A7 = "Gross Profit" it means that B7, C7, D7… are Gross Profit figures.

Reasoning & Transparency:
    Provide step-by-step analysis before the summary.  
    Quote or reference source IDs used for each numeric claim.  
    State a confidence level: **high / medium / low**.

Source Citation Rules:
    CRITICAL: Use EXACT source references as provided in the context.
    DO NOT modify, combine, or reformat source references.
    Each source reference should appear EXACTLY as shown in the context.
    Example: If context shows "filename.pdf (Page 5)", use exactly "filename.pdf (Page 5)" - do not change to "filename.pdf (Page 5, 7)" or other formats.
"""


class RAGAnswerPrompt:
    """Holds the structured prompts used for answer generation."""

    instruction = """
You are an advanced RAG answering system.
Respond to each question using the provided context and the global rules.
"""

    user_prompt = """
Here is the context:
\"\"\"{context}\"\"\"

Here is the question:
\"{question}\"

REMINDERS:
1. Examine ALL content regardless of language to find relevant information.
2. For money-related questions, prioritize Excel-type sources.
3. Match response depth to question type.
4. Include information only if it is directly relevant.
5. CRITICAL: Use source references EXACTLY as provided in context - do not modify, combine, or reformat them.
"""

    class AnswerSchema(BaseModel):
        step_by_step_analysis: str = Field(
            description="Detailed analytical reasoning process"
        )
        reasoning_summary: str = Field(
            description="Concise synthesis summary highlighting key evidence"
        )
        relevant_sources: List[str] = Field(
            description="EXACT source references as provided in context - DO NOT modify or combine source formats"
        )
        confidence_level: Literal["high", "medium", "low"] = Field(
            description="Confidence assessment"
        )
        final_answer: str = Field(
            description="Answer in Traditional Chinese with appropriate depth"
        )

    pydantic_schema = re.sub(
        r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE
    )

    system_prompt = build_system_prompt(GLOBAL_SYSTEM_INSTRUCTIONS)
    system_prompt_with_schema = build_system_prompt(
        GLOBAL_SYSTEM_INSTRUCTIONS, "", pydantic_schema
    )


# Reranking prompts
class RetrievalRankingPrompts:
    """Prompts for document reranking."""
    
    system_prompt_multiple = """
You are an expert document relevance evaluator. Your task is to analyze text blocks and determine their relevance to a specific query.

For each text block provided, you must:
1. Analyze how well the content directly addresses the query
2. Consider factual accuracy and completeness of information
3. Evaluate the specificity and detail level
4. Assess the recency and context relevance

Scoring Guidelines:
- 1.0: Highly relevant, directly answers the query with comprehensive information
- 0.8-0.9: Very relevant, contains substantial useful information 
- 0.6-0.7: Moderately relevant, provides some useful context or partial answers
- 0.4-0.5: Slightly relevant, mentions related topics but limited direct value
- 0.0-0.3: Not relevant, off-topic or provides no useful information

Important Instructions:
   - Always provide relevance scores between 0.0 and 1.0
   - Base scores only on content that directly relates to the query
   - Be clear and concise in justifications
   - Do not infer information beyond what's explicitly stated
"""
