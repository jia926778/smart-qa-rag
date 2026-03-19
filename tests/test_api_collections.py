"""
集合管理 API 测试模块

本模块测试集合相关的 REST API 端点，包括：
- 创建集合
- 列出所有集合
- 删除集合
- 获取集合统计信息
"""

from __future__ import annotations


class TestCollectionsApi:
    """集合 API 端点测试类"""

    def test_create_collection(self, client):
        """
        测试创建集合端点。

        测试场景：发送 POST 请求创建新集合
        预期结果：返回 200 状态码，响应包含正确的集合名称
        """
        resp = client.post(
            "/api/v1/collections/",
            json={"name": "test", "description": "Test collection"},
        )
        assert resp.status_code == 200  # 验证请求成功
        data = resp.json()
        assert data["name"] == "test"  # 验证返回的集合名称正确

    def test_list_collections(self, client):
        """
        测试列出集合端点。

        测试场景：发送 GET 请求获取所有集合列表
        预期结果：返回 200 状态码，响应为列表格式
        """
        resp = client.get("/api/v1/collections/")
        assert resp.status_code == 200  # 验证请求成功
        data = resp.json()
        assert isinstance(data, list)  # 验证返回的是列表

    def test_delete_collection(self, client):
        """
        测试删除集合端点。

        测试场景：发送 DELETE 请求删除指定集合
        预期结果：返回 200 状态码表示删除成功
        """
        resp = client.delete("/api/v1/collections/default")
        assert resp.status_code == 200  # 验证删除成功

    def test_collection_stats(self, client):
        """
        测试获取集合统计信息端点。

        测试场景：发送 GET 请求获取集合的统计信息
        预期结果：返回 200 状态码，响应包含文档数量字段
        """
        resp = client.get("/api/v1/collections/default/stats")
        assert resp.status_code == 200  # 验证请求成功
        data = resp.json()
        assert "documents_count" in data  # 验证包含文档计数字段
