"""文本转 SQL 智能体 — RAG 图中的条件分支节点。

当查询分析器检测到 ``data_query`` 意图（关于数字、排名、聚合、结构化数据比较的问题）时，
本智能体生成 SQL 查询，在 SQLite 存储中执行，并将结果格式化为自然语言答案。

主要职责：
    1. 根据用户问题和数据库模式生成 SQL 查询语句。
    2. 执行 SQL 查询并获取结构化数据结果。
    3. 将查询结果格式化为自然语言答案。
    4. 处理 SQL 执行错误并尝试自动修正。

工作流程：
    获取模式 → 生成SQL → 执行查询 → 格式化结果 → 返回答案
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

# 文本转 SQL 提示词模板
# 指导 LLM 根据用户问题和数据库模式生成有效的 SQL 查询
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

# 结果格式化提示词模板
# 指导 LLM 将 SQL 查询结果转换为清晰的自然语言答案
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
    """构建并返回文本转 SQL 智能体的 LangGraph 节点函数。

    Args:
        settings (Settings): 应用配置对象，包含 LLM 模型参数、API 密钥等配置。
        sql_store (SQLStore): SQL 存储服务实例，用于获取模式和执行查询。

    Returns:
        Callable: 异步的 LangGraph 节点函数，接收 GraphState 作为输入，
            返回包含 sql_result 和 sql_query 的状态更新字典。

    Note:
        该函数内部定义了 sql_agent_node 异步函数，
        该函数会根据用户问题生成 SQL 查询，执行并格式化结果。
    """

    # 构建 LLM 调用参数，使用 temperature=0 确保生成结果稳定
    kwargs: Dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": 0,
        "max_tokens": 1024,
        "openai_api_key": settings.OPENAI_API_KEY,
    }
    # 如果配置了自定义 API 端点，则添加到参数中
    if settings.OPENAI_API_BASE:
        kwargs["openai_api_base"] = settings.OPENAI_API_BASE
    # 初始化 ChatOpenAI 实例
    llm = ChatOpenAI(**kwargs)

    async def sql_agent_node(state: GraphState) -> Dict[str, Any]:
        """文本转 SQL 节点的核心逻辑。

        Args:
            state (GraphState): RAG 图的当前状态，包含用户问题和集合名称等信息。

        Returns:
            Dict[str, Any]: 状态更新字典，包含以下键：
                - sql_result (Dict, optional): SQL 查询结果，包含查询语句、列名、行数据和格式化答案
                - sql_query (str, optional): 生成的 SQL 查询语句

        Note:
            如果没有可用的结构化数据表或问题不适合 SQL 查询，将返回 None 值。
        """
        # 从状态中提取必要信息
        question = state["question"]  # 用户原始问题
        collection_name = state.get("collection_name", "default")  # 集合名称

        # 步骤 1: 获取数据库模式
        schema = sql_store.get_schema(collection_name)
        # 如果没有可用的结构化数据表，跳过 SQL 查询
        if schema == "(no structured data tables)":
            logger.info("SQL Agent: no structured data available, skipping.")
            return {
                "sql_result": None,
                "sql_query": None,
            }

        # 步骤 2: 生成 SQL 查询
        gen_prompt = _TEXT_TO_SQL_PROMPT.format(schema=schema, question=question)
        try:
            response = await llm.ainvoke(gen_prompt)
            sql = response.content.strip()

            # 清理可能的 markdown 代码块标记
            if sql.startswith("```"):
                sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            if sql.startswith("sql"):
                sql = sql[3:].strip()

            # 如果 LLM 返回 NO_SQL 或不是有效的 SELECT 语句，跳过 SQL 查询
            if sql == "NO_SQL" or not sql.upper().startswith("SELECT"):
                logger.info("SQL Agent: question not suitable for SQL.")
                return {"sql_result": None, "sql_query": None}

        except Exception as exc:
            # SQL 生成失败，记录错误并返回
            logger.error("SQL generation failed: %s", exc)
            return {"sql_result": None, "sql_query": None}

        # 步骤 3: 执行 SQL 查询
        result = sql_store.execute_sql(collection_name, sql)
        if result.get("error"):
            logger.warning("SQL execution error: %s", result["error"])
            # 尝试自动修正 SQL 查询（仅一次）
            correction = await _self_correct_sql(llm, schema, question, sql, result["error"])
            if correction:
                # 使用修正后的 SQL 重新执行
                result = sql_store.execute_sql(collection_name, correction)
                sql = correction
                if result.get("error"):
                    logger.warning("SQL self-correction also failed: %s", result["error"])
                    return {"sql_result": None, "sql_query": sql}

        # 步骤 4: 将查询结果格式化为自然语言
        columns = result.get("columns", [])  # 列名列表
        rows = result.get("rows", [])  # 行数据列表

        # 构建行文本，限制显示前 20 行
        if rows:
            row_text_parts = []
            for row in rows[:20]:  # 限制显示数量
                row_text_parts.append(" | ".join(str(v) for v in row))
            row_text = "\n".join(row_text_parts)
        else:
            row_text = "(empty result set)"

        # 构建格式化提示词
        format_prompt = _RESULT_FORMAT_PROMPT.format(
            question=question,
            sql=sql,
            columns=", ".join(columns),
            rows=row_text,
        )

        try:
            # 调用 LLM 格式化查询结果
            format_response = await llm.ainvoke(format_prompt)
            formatted_answer = format_response.content
        except Exception as exc:
            # 格式化失败时，使用原始行文本作为答案
            logger.error("SQL result formatting failed: %s", exc)
            formatted_answer = f"查询结果：\n{row_text}"

        # 记录 SQL 查询结果日志
        logger.info(
            "SQL Agent: executed SQL (%d rows returned) for question: %s",
            len(rows), question[:50],
        )

        # 返回 SQL 查询结果
        return {
            "sql_result": {
                "query": sql,  # SQL 查询语句
                "columns": columns,  # 列名列表
                "rows": rows[:20],  # 行数据（限制 20 行）
                "row_count": len(rows),  # 总行数
                "formatted_answer": formatted_answer,  # 格式化的自然语言答案
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
    """尝试修正执行错误的 SQL 查询。

    当 SQL 查询执行失败时，调用 LLM 尝试生成修正后的 SQL 语句。

    Args:
        llm (ChatOpenAI): 大语言模型实例。
        schema (str): 数据库模式描述。
        question (str): 用户原始问题。
        bad_sql (str): 执行失败的 SQL 语句。
        error (str): 错误信息。

    Returns:
        str | None: 修正后的 SQL 语句，如果修正失败则返回 None。

    Note:
        该函数仅尝试修正一次，不会进行多次迭代修正。
    """
    # 构建修正提示词，包含错误信息和原始上下文
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
        # 清理可能的 markdown 代码块标记
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # 验证是否为有效的 SELECT 语句
        if sql.upper().startswith("SELECT"):
            return sql
    except Exception:
        # 修正失败，返回 None
        pass
    return None
