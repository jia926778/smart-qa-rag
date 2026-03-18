"""Text-to-SQL Agent — conditional branch in the RAG graph.

When the QueryAnalyzer detects a ``data_query`` intent (questions about
numbers, rankings, aggregations, comparisons over structured data), this
agent generates a SQL query, executes it against the SQLite store, and
formats the results into a natural language answer.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from app.agents.state import GraphState
from app.config import Settings
from app.services.sql_store import SQLStore
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TEXT_TO_SQL_PROMPT = """\
You are an expert SQL query generator.  Given a user's natural language
question and a database schema, generate a valid SQLite SELECT query.

Database schema:
{schema}

Rules:
1. Generate ONLY a valid SQLite SELECT statement.  No explanations.
2. Use double quotes for table and column names that may contain special characters.
3. Use LIMIT 50 at the end unless the user specifies otherwise.
4. For aggregation questions, use appropriate GROUP BY, ORDER BY, SUM, AVG, COUNT, etc.
5. Handle Chinese column names correctly.
6. If the question cannot be answered with the available tables, reply with: NO_SQL
7. Reply with ONLY the SQL query (or NO_SQL), nothing else.

User question: {question}

SQL:
"""

_RESULT_FORMAT_PROMPT = """\
You are a helpful data analyst.  Given a user's question, a SQL query,
and its results, provide a clear and concise natural language answer in Chinese.

User question: {question}

SQL query executed: {sql}

Query results (columns: {columns}):
{rows}

Rules:
1. Answer in Chinese.
2. Summarize the data clearly, highlighting key numbers.
3. If the data contains a table, present it in Markdown table format.
4. Cite the source table(s) used.
5. If results are empty, explain that no matching data was found.
"""


def build_sql_agent(settings: Settings, sql_store: SQLStore):
    """Return a LangGraph node function for Text-to-SQL."""

    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": 0,
        "max_tokens": 1024,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    llm = ChatOpenAI(**kwargs)

    async def sql_agent_node(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        collection_name = state.get("collection_name", "default")

        # 1. Get database schema
        schema = sql_store.get_schema(collection_name)
        if schema == "(no structured data tables)":
            logger.info("SQL Agent: no structured data available, skipping.")
            return {
                "sql_result": None,
                "sql_query": None,
            }

        # 2. Generate SQL query
        gen_prompt = _TEXT_TO_SQL_PROMPT.format(schema=schema, question=question)
        try:
            response = await llm.ainvoke(gen_prompt)
            sql = response.content.strip()

            # Clean up markdown fences
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if sql.startswith("sql"):
                sql = sql[3:].strip()

            if sql == "NO_SQL" or not sql.upper().startswith("SELECT"):
                logger.info("SQL Agent: question not suitable for SQL.")
                return {"sql_result": None, "sql_query": None}

        except Exception as exc:
            logger.error("SQL generation failed: %s", exc)
            return {"sql_result": None, "sql_query": None}

        # 3. Execute SQL
        result = sql_store.execute_sql(collection_name, sql)
        if result.get("error"):
            logger.warning("SQL execution error: %s", result["error"])
            # Try to self-correct once
            correction = await _self_correct_sql(llm, schema, question, sql, result["error"])
            if correction:
                result = sql_store.execute_sql(collection_name, correction)
                sql = correction
                if result.get("error"):
                    logger.warning("SQL self-correction also failed: %s", result["error"])
                    return {"sql_result": None, "sql_query": sql}

        # 4. Format results into natural language
        columns = result.get("columns", [])
        rows = result.get("rows", [])

        # Build row text
        if rows:
            row_text_parts = []
            for row in rows[:20]:  # Limit display
                row_text_parts.append(" | ".join(str(v) for v in row))
            row_text = "\n".join(row_text_parts)
        else:
            row_text = "(empty result set)"

        format_prompt = _RESULT_FORMAT_PROMPT.format(
            question=question,
            sql=sql,
            columns=", ".join(columns),
            rows=row_text,
        )

        try:
            format_response = await llm.ainvoke(format_prompt)
            formatted_answer = format_response.content
        except Exception as exc:
            logger.error("SQL result formatting failed: %s", exc)
            formatted_answer = f"查询结果：\n{row_text}"

        logger.info(
            "SQL Agent: executed SQL (%d rows returned) for question: %s",
            len(rows), question[:50],
        )

        return {
            "sql_result": {
                "query": sql,
                "columns": columns,
                "rows": rows[:20],
                "row_count": len(rows),
                "formatted_answer": formatted_answer,
            },
            "sql_query": sql,
        }

    return sql_agent_node


async def _self_correct_sql(
    llm: ChatOpenAI,
    schema: str,
    question: str,
    bad_sql: str,
    error: str,
) -> str | None:
    """Attempt to fix a SQL query that resulted in an error."""
    prompt = (
        f"The following SQL query resulted in an error:\n"
        f"SQL: {bad_sql}\n"
        f"Error: {error}\n\n"
        f"Database schema:\n{schema}\n\n"
        f"Original question: {question}\n\n"
        f"Please generate a corrected SQL query.  Reply with ONLY the SQL, no explanation."
    )
    try:
        response = await llm.ainvoke(prompt)
        sql = response.content.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        if sql.upper().startswith("SELECT"):
            return sql
    except Exception:
        pass
    return None
