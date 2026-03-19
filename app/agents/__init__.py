"""LangGraph 多智能体 RAG 管道模块。

本模块提供了基于 LangGraph 的多智能体 RAG（检索增强生成）系统实现。

模块组成：
    - graph: 工作流图构建器，组装所有智能体节点
    - generator: 答案生成智能体，基于检索结果生成回答
    - evaluator: 质量评估智能体，评估答案质量并决定是否重试
    - sql_agent: 文本转 SQL 智能体，处理结构化数据查询
    - query_analyzer: 查询分析智能体，分析用户问题意图
    - retriever_agent: 检索智能体，从知识库检索相关文档
    - state: 状态定义，包含 GraphState 和相关类型

使用示例：
    from app.agents import build_rag_graph

    graph = build_rag_graph(settings, retriever, sql_store)
    result = await graph.ainvoke({"question": "用户问题"})
"""
