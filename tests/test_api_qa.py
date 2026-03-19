"""
问答 API 测试模块

本模块测试问答相关的 REST API 端点，包括：
- 问答查询端点
- 空问题验证
- 健康检查端点
"""

from __future__ import annotations


class TestQAApi:
    """问答 API 端点测试类"""

    def test_ask_endpoint(self, client):
        """
        测试问答查询端点。

        测试场景：发送 POST 请求进行问答查询
        预期结果：返回 200 状态码，响应包含答案和耗时字段
        """
        resp = client.post(
            "/api/v1/qa/ask",
            json={"question": "What is RAG?", "collection_name": "default"},
        )
        assert resp.status_code == 200  # 验证请求成功
        data = resp.json()
        assert "answer" in data  # 验证包含答案字段
        assert "elapsed_ms" in data  # 验证包含耗时字段

    def test_ask_empty_question(self, client):
        """
        测试空问题验证。

        测试场景：发送空字符串作为问题
        预期结果：返回 422 状态码（验证错误）
        """
        resp = client.post(
            "/api/v1/qa/ask",
            json={"question": "", "collection_name": "default"},
        )
        assert resp.status_code == 422  # 验证返回验证错误

    def test_health_check(self, client):
        """
        测试健康检查端点。

        测试场景：发送 GET 请求检查服务健康状态
        预期结果：返回 200 状态码，状态为 "ok"
        """
        resp = client.get("/health")
        assert resp.status_code == 200  # 验证服务正常
        assert resp.json()["status"] == "ok"  # 验证健康状态正确
