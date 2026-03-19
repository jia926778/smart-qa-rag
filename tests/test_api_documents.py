"""
文档管理 API 测试模块

本模块测试文档相关的 REST API 端点，包括：
- 上传文档
- 列出文档
- 删除文档
"""

from __future__ import annotations

import io


class TestDocumentsApi:
    """文档 API 端点测试类"""

    def test_upload_document(self, client):
        """
        测试上传文档端点。

        测试场景：发送 POST 请求上传文本文件
        预期结果：返回 200 状态码，响应包含文件名和分块数量
        """
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")},
            data={"collection_name": "default"},
        )
        assert resp.status_code == 200  # 验证上传成功
        data = resp.json()
        assert data["filename"] == "test.txt"  # 验证返回的文件名正确
        assert data["chunks_count"] == 3  # 验证返回的分块数量正确

    def test_list_documents(self, client):
        """
        测试列出文档端点。

        测试场景：发送 GET 请求获取指定集合中的文档列表
        预期结果：返回 200 状态码，响应为包含一个文档的列表
        """
        resp = client.get("/api/v1/documents/default")
        assert resp.status_code == 200  # 验证请求成功
        data = resp.json()
        assert isinstance(data, list)  # 验证返回的是列表
        assert len(data) == 1  # 验证列表包含一个文档

    def test_delete_document(self, client):
        """
        测试删除文档端点。

        测试场景：发送 DELETE 请求删除指定文档
        预期结果：返回 200 状态码，响应包含删除的分块数量
        """
        resp = client.delete("/api/v1/documents/default/test.txt")
        assert resp.status_code == 200  # 验证删除成功
        data = resp.json()
        assert data["deleted_chunks"] == 3  # 验证返回删除的分块数
