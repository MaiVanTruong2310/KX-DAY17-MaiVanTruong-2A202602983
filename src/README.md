# Day 17 implementation

**Sinh viên:** Mai Văn Trường — **MSSV:** 2A202602983

Chạy từ thư mục gốc dự án bằng Python 3.11+:

```bash
python src/benchmark.py
python -m pytest src/test_agents.py -q
```

Benchmark mặc định dùng chế độ offline, không cần API key hay thư viện ngoài Python. Cài `pytest` để chạy kiểm thử. Hai tập `data/conversations.json` và `data/advanced_long_context.json` được giữ nguyên.

Hồ sơ mẫu của sinh viên nằm ở `state/profiles/MaiVanTruong-2A202602983/User.md`. Agent đọc hồ sơ này khi gọi với `user_id="MaiVanTruong-2A202602983"`. Các hồ sơ khác được tạo trong `state/` khi chạy và không được đưa lên Git.

Muốn dùng model thật, cài LangChain và gói tích hợp tương ứng, tạo `.env` ở thư mục gốc rồi đặt `LIVE_MODE=1`, `LLM_PROVIDER`, `LLM_MODEL` và API key/base URL tương ứng. Các provider được hỗ trợ: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`. Chế độ live chưa được dùng để tính các số liệu trong báo cáo.

- `model_provider.py`, `config.py`: cấu hình và tạo model theo provider.
- `memory_store.py`: ước lượng token, lưu `User.md`, trích fact có quy tắc, nén thread.
- `agent_baseline.py`: giữ toàn bộ lịch sử trong một thread, quên ở thread mới.
- `agent_advanced.py`: dùng hồ sơ bền vững, summary và các message gần nhất.
- `benchmark.py`: so sánh hai agent trên hai tập dữ liệu trong state tạm độc lập.
- `test_agents.py`: kiểm tra lưu hồ sơ, đính chính, nhiễu, nhớ qua lần khởi tạo mới và hiệu quả compact.

Xem [báo cáo kết quả](../REPORT.md) để biết số liệu, cách đo và giới hạn.
