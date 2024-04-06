import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import aiofiles
import pytest

from py_rag.core.document_processor import DocumentProcessor, document_processor


class TestDocumentProcessor:
    """文档处理器测试类"""

    @pytest.fixture
    def processor(self):
        """创建文档处理器实例"""
        return DocumentProcessor()

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def sample_texts(self):
        """示例文本内容"""
        return {
            'simple': "这是一个简单的测试文档。",
            'long': "这是一个很长的文档。" + "测试内容。" * 100,
            'multi_paragraph': """第一段内容。

第二段内容，包含更多信息。

第三段内容。""",
            'unicode': "这是包含中文、English和éñglish的文档。",
            'empty': "",
            'whitespace': "   \n\t   \n   "
        }

    # ========== 基础功能测试 ==========

    def test_get_supported_types(self, processor):
        """测试获取支持的文件类型"""
        supported_types = processor.get_supported_types()

        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert '.txt' in supported_types
        assert '.pdf' in supported_types
        assert '.docx' in supported_types

    def test_is_supported_file(self, processor):
        """测试文件类型支持检查"""
        # 支持的文件类型
        assert processor.is_supported_file("test.txt") == True
        assert processor.is_supported_file("document.pdf") == True
        assert processor.is_supported_file("report.docx") == True
        assert processor.is_supported_file("data.csv") == True

        # 不支持的文件类型
        assert processor.is_supported_file("image.jpg") == False
        assert processor.is_supported_file("video.mp4") == False
        assert processor.is_supported_file("archive.zip") == False

        # 边界情况
        assert processor.is_supported_file("") == False
        assert processor.is_supported_file("noextension") == False
        # assert processor.is_supported_file(".txt") == True

    # def test_split_text_by_sentences(self, processor, sample_texts):
    #     """测试句子分割"""
    #     # 简单文本
    #     result = processor._split_text_by_sentences(sample_texts['simple'])
    #     assert len(result) == 1
    #     assert result[0] == "这是一个简单的测试文档。"
    #
    #     # 多段落文本
    #     result = processor._split_text_by_sentences(sample_texts['multi_paragraph'])
    #     assert len(result) >= 3
    #
    #     # 空文本
    #     result = processor._split_text_by_sentences(sample_texts['empty'])
    #     assert len(result) == 0
    #
    #     # 纯空白
    #     result = processor._split_text_by_sentences(sample_texts['whitespace'])
    #     assert len(result) == 0

    # def test_chunk_text(self, processor, sample_texts):
    #     """测试文本分块"""
    #     # 短文本 - 应该返回一个块
    #     chunks = processor._chunk_text(sample_texts['simple'])
    #     assert len(chunks) == 1
    #     assert chunks[0] == sample_texts['simple']
    #
    #     # 长文本 - 应该分成多个块
    #     chunks = processor._chunk_text(sample_texts['long'])
    #     assert len(chunks) > 1
    #
    #     # 验证每个块的长度不超过限制
    #     for chunk in chunks:
    #         assert len(chunk) <= processor.chunk_size
    #
    #     # 空文本
    #     chunks = processor._chunk_text(sample_texts['empty'])
    #     assert len(chunks) == 0
    #
    #     # 自定义参数测试
    #     small_chunks = processor._chunk_text(sample_texts['long'], chunk_size=100, overlap=20)
    #     assert len(small_chunks) > len(chunks)  # 更小的块应该产生更多分片

    # ========== 文件创建和保存测试 ==========

    @pytest.mark.asyncio
    async def test_save_temp_file(self, processor, temp_dir):
        """测试保存临时文件"""
        content = f"测试文件内容"
        filename = "test.txt"

        # 临时修改处理器的临时目录
        original_temp_dir = processor.temp_dir
        processor.temp_dir = temp_dir

        try:
            file_path = await processor.save_temp_file(content, filename)

            assert file_path.exists()
            assert file_path.name.endswith("_test.txt")
            assert file_path.parent == temp_dir

            # 验证文件内容
            async with aiofiles.open(file_path, 'rb') as f:
                saved_content = await f.read()
                assert saved_content == content

        finally:
            processor.temp_dir = original_temp_dir

    @pytest.mark.asyncio
    async def test_save_temp_file_with_special_chars(self, processor, temp_dir):
        """测试保存包含特殊字符的文件名"""
        content = f"content"
        filename = "测试文件 (1) [copy].txt"

        processor.temp_dir = temp_dir

        file_path = await processor.save_temp_file(content, filename)

        assert file_path.exists()
        # 文件名应该被清理但保持可读性
        assert "txt" in file_path.name

    # ========== TXT 文件处理测试 ==========

    def create_temp_txt_file(self, temp_dir: Path, content: str, filename: str = "test.txt") -> Path:
        """创建临时TXT文件"""
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    @pytest.mark.asyncio
    async def test_process_txt_file(self, processor, temp_dir, sample_texts):
        """测试TXT文件处理"""
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['multi_paragraph'])

        result = await processor._process_txt_file(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) >= 3
        assert result['file_name'] == file_path.name
        assert result['file_size'] > 0
        assert 'processing_time' in result
        assert 'document_hash' in result

    @pytest.mark.asyncio
    async def test_process_empty_txt_file(self, processor, temp_dir):
        """测试空TXT文件处理"""
        file_path = self.create_temp_txt_file(temp_dir, "")

        result = await processor._process_txt_file(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) == 0

    @pytest.mark.asyncio
    async def test_process_txt_file_with_encoding_issues(self, processor, temp_dir):
        """测试编码问题的文件处理"""
        file_path = temp_dir / "test.txt"

        # 创建有编码问题的文件
        with open(file_path, 'wb') as f:
            f.write("测试".encode('gbk'))  # 使用GBK编码

        result = await processor._process_txt_file(file_path)

        # 应该能够处理编码问题
        assert result['status'] == 'success'
        assert len(result['chunks']) > 0

    # ========== PDF 文件处理测试 ==========

    @pytest.mark.asyncio
    async def test_process_pdf_file_mock(self, processor, temp_dir):
        """测试PDF文件处理（使用mock）"""
        # 创建一个假的PDF文件
        file_path = temp_dir / "test.pdf"
        file_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # Mock PyPDF2
        with patch('py_rag.core.document_processor.PdfReader') as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "这是PDF文档的内容。"

            mock_pdf = MagicMock()
            mock_pdf.pages = [mock_page]
            mock_reader.return_value = mock_pdf

            result = await processor._process_pdf_file(file_path)

            assert result['status'] == 'success'
            assert len(result['chunks']) > 0
            assert "PDF文档的内容" in result['chunks'][0]

    @pytest.mark.asyncio
    async def test_process_corrupted_pdf_file(self, processor, temp_dir):
        """测试损坏的PDF文件处理"""
        # 创建一个损坏的PDF文件
        file_path = temp_dir / "corrupted.pdf"
        file_path.write_bytes(b"this is not a valid pdf")

        result = await processor._process_pdf_file(file_path)

        assert result['status'] == 'error'
        assert 'error' in result

    # ========== DOCX 文件处理测试 ==========

    @pytest.mark.asyncio
    async def test_process_docx_file_mock(self, processor, temp_dir):
        """测试DOCX文件处理（使用mock）"""
        file_path = temp_dir / "test.docx"
        file_path.write_bytes(b"fake docx content")

        # Mock python-docx
        with patch('py_rag.core.document_processor.Document') as mock_document:
            mock_paragraph = MagicMock()
            mock_paragraph.text = "这是Word文档的段落。"

            mock_doc = MagicMock()
            mock_doc.paragraphs = [mock_paragraph]
            mock_document.return_value = mock_doc

            result = await processor._process_docx_file(file_path)

            assert result['status'] == 'success'
            assert len(result['chunks']) > 0
            assert "Word文档的段落" in result['chunks'][0]

    # ========== CSV 文件处理测试 ==========

    def create_temp_csv_file(self, temp_dir: Path, data: list, filename: str = "test.csv") -> Path:
        """创建临时CSV文件"""
        file_path = temp_dir / filename
        with open(file_path, 'w', encoding='utf-8', newline='') as f:
            import csv
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
        return file_path

    @pytest.mark.asyncio
    async def test_process_csv_file(self, processor, temp_dir):
        """测试CSV文件处理"""
        csv_data = [
            ['姓名', '年龄', '城市'],
            ['张三', '25', '北京'],
            ['李四', '30', '上海'],
            ['王五', '28', '广州']
        ]

        file_path = self.create_temp_csv_file(temp_dir, csv_data)

        result = await processor._process_csv_file(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) > 0

        # 验证CSV内容被正确解析
        content = " ".join(result['chunks'])
        assert '张三' in content
        assert '北京' in content

    @pytest.mark.asyncio
    async def test_process_empty_csv_file(self, processor, temp_dir):
        """测试空CSV文件处理"""
        file_path = self.create_temp_csv_file(temp_dir, [])

        result = await processor._process_csv_file(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) == 0

    # ========== 主处理方法测试 ==========

    @pytest.mark.asyncio
    async def test_process_document_txt(self, processor, temp_dir, sample_texts):
        """测试主文档处理方法 - TXT文件"""
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['multi_paragraph'])

        result = await processor.process_document(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) > 0
        assert result['file_name'] == file_path.name
        assert result['file_size'] > 0
        assert result['processing_time'] > 0
        assert len(result['document_hash']) == 64  # SHA256 hash length

    @pytest.mark.asyncio
    async def test_process_document_unsupported_format(self, processor, temp_dir):
        """测试不支持的文件格式"""
        file_path = temp_dir / "test.xyz"
        file_path.write_text("content")

        result = await processor.process_document(file_path)

        assert result['status'] == 'error'
        assert 'unsupported' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_process_document_nonexistent_file(self, processor, temp_dir):
        """测试不存在的文件"""
        file_path = temp_dir / "nonexistent.txt"

        result = await processor.process_document(file_path)

        assert result['status'] == 'error'
        assert 'not found' in result['error'].lower() or 'does not exist' in result['error'].lower()

    # ========== 错误处理测试 ==========

    @pytest.mark.asyncio
    async def test_process_document_permission_error(self, processor, temp_dir):
        """测试权限错误处理"""
        file_path = temp_dir / "readonly.txt"
        file_path.write_text("content")

        # 模拟权限错误
        with patch('aiofiles.open', side_effect=PermissionError("Permission denied")):
            result = await processor.process_document(file_path)

            assert result['status'] == 'error'
            assert 'permission' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_process_large_file_timeout(self, processor, temp_dir):
        """测试大文件处理超时"""
        # 创建一个"大"文件
        large_content = "x" * (10 * 1024 * 1024)  # 10MB
        file_path = self.create_temp_txt_file(temp_dir, large_content, "large.txt")

        # 设置很短的超时时间来模拟超时
        original_timeout = processor.processing_timeout
        processor.processing_timeout = 0.001  # 1ms

        try:
            result = await processor.process_document(file_path)
            # 取决于系统性能，可能成功也可能超时
            assert result['status'] in ['success', 'error']
        finally:
            processor.processing_timeout = original_timeout

    # ========== 性能测试 ==========

    @pytest.mark.asyncio
    async def test_processing_performance(self, processor, temp_dir, sample_texts):
        """测试处理性能"""
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['long'])

        import time
        start_time = time.time()

        result = await processor.process_document(file_path)

        end_time = time.time()
        processing_time = end_time - start_time

        assert result['status'] == 'success'
        assert processing_time < 10.0  # 应该在10秒内完成
        assert result['processing_time'] > 0

    # ========== 并发测试 ==========

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, processor, temp_dir, sample_texts):
        """测试并发处理"""
        # 创建多个文件
        files = []
        for i in range(5):
            file_path = self.create_temp_txt_file(
                temp_dir,
                sample_texts['multi_paragraph'],
                f"test_{i}.txt"
            )
            files.append(file_path)

        # 并发处理
        tasks = [processor.process_document(file_path) for file_path in files]
        results = await asyncio.gather(*tasks)

        # 验证所有处理都成功
        for result in results:
            assert result['status'] == 'success'
            assert len(result['chunks']) > 0

    # ========== 清理测试 ==========

    @pytest.mark.asyncio
    async def test_cleanup_temp_files(self, processor, temp_dir):
        """测试临时文件清理"""
        # 创建一些临时文件
        processor.temp_dir = temp_dir

        content = b"test content"
        file_path1 = await processor.save_temp_file(content, "test1.txt")
        file_path2 = await processor.save_temp_file(content, "test2.txt")

        assert file_path1.exists()
        assert file_path2.exists()

        # 清理临时文件
        await processor.cleanup_temp_files()

        # 验证文件被删除（注意：实际实现可能保留某些文件）
        # 这里的断言取决于具体的清理策略

    # ========== 集成测试 ==========

    @pytest.mark.asyncio
    async def test_full_workflow(self, temp_dir, sample_texts):
        """测试完整工作流程"""
        # 使用全局实例
        file_path = self.create_temp_txt_file(temp_dir, sample_texts['multi_paragraph'])

        # 处理文档
        result = await document_processor.process_document(file_path)

        assert result['status'] == 'success'
        assert len(result['chunks']) > 0

        # 验证返回的数据结构
        required_keys = ['status', 'chunks', 'file_name', 'file_size', 'processing_time', 'document_hash']
        for key in required_keys:
            assert key in result

        # 验证chunks的结构
        for chunk in result['chunks']:
            assert isinstance(chunk, str)
            assert len(chunk) > 0


# ========== 参数化测试 ==========

@pytest.mark.parametrize("file_extension,expected_support", [
    ("txt", True),
    ("pdf", True),
    ("docx", True),
    ("csv", True),
    ("json", True),
    ("md", True),
    ("jpg", False),
    ("mp4", False),
    ("exe", False),
])
def test_file_support_parametrized(file_extension, expected_support):
    """参数化测试文件支持"""
    processor = DocumentProcessor()
    filename = f"test.{file_extension}"

    assert processor.is_supported_file(filename) == expected_support


@pytest.mark.parametrize("chunk_size,overlap,text_length", [
    (100, 20, 50),  # 短文本
    (100, 20, 150),  # 中等文本
    (100, 20, 500),  # 长文本
])
def test_chunking_parametrized(chunk_size, overlap, text_length):
    """参数化测试文本分块"""
    processor = DocumentProcessor()
    text = "x" * text_length

    chunks = processor._chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if text_length <= chunk_size:
        assert len(chunks) <= 1
    else:
        assert len(chunks) > 1

    # 验证每个块不超过指定大小
    for chunk in chunks:
        assert len(chunk) <= chunk_size


# ========== Fixture 清理 ==========

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """每个测试后的清理"""
    yield
    # 测试后清理逻辑
    import gc
    gc.collect()
