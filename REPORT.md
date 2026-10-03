# Báo cáo Day 17 — Memory Systems for AI Agent

**Họ tên:** Mai Văn Trường

**Mã sinh viên:** 2A202602983

## Thiết kế

Baseline giữ đầy đủ tin nhắn theo `thread_id` trong bộ nhớ tiến trình. Một thread mới không đọc được fact ở thread cũ. Advanced giữ tin nhắn gần nhất, nén nội dung cũ thành các ý chính và lưu các fact ổn định vào `state/profiles/<user>/User.md`. Hồ sơ nằm trên đĩa nên vẫn dùng được sau khi tạo lại agent. Khi có lời đính chính rõ ràng, giá trị mới thay thế giá trị cũ; câu hỏi, ví dụ đi họp ở Hà Nội và câu đùa chuyển nghề không được ghi thành fact.

Benchmark dùng cùng dữ liệu và cùng cách ước lượng token cho hai agent. Các câu recall được hỏi trên thread mới. `Agent tokens only` là tổng ước lượng token của đầu vào và đầu ra mỗi lượt. `Prompt tokens processed` là tổng kích thước ngữ cảnh được đưa vào mỗi lần trả lời. Hai chỉ số này không phải số token do nhà cung cấp API tính phí.

## Kết quả chạy offline

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2.986 | 13.554 | 0% | 0% | 0 | 0 |
| Advanced | 2.718 | 19.034 | 100% | 100% | 283 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 2.606 | 21.919 | 0% | 0% | 0 | 0 |
| Advanced | 2.586 | 7.564 | 100% | 100% | 224 | 7 |

Ở tập chuẩn, Advanced xử lý nhiều prompt token hơn Baseline khoảng 40% vì phải đọc thêm `User.md`, trong khi thread chưa đủ dài để compact. Ở stress test, 7 lần compact giúp giảm khoảng 65% prompt token so với Baseline. Cả hai tập đều cho thấy hồ sơ bền vững cải thiện recall qua thread mới. File memory tăng 283 và 224 byte tương ứng; với thời gian sử dụng dài hơn, file vẫn có thể tiếp tục tăng và cần chính sách dọn dẹp.

`Response quality` ở đây chỉ là điểm heuristic offline: 80% dựa trên số chuỗi mong đợi xuất hiện, 20% dựa trên câu trả lời ngắn và có nội dung. Vì vậy 100% **không chứng minh** chất lượng hội thoại tự nhiên hay độ đúng của model thật. Phần trích fact cũng dựa trên mẫu câu tiếng Việt, có thể bỏ sót cách diễn đạt khác hoặc ghi sai khi câu quá phức tạp. Summary giữ các mốc chủ đề chính, có thể mất chi tiết không thuộc các mốc đó. Một hệ thống triển khai thật cần extraction có độ tin cậy, kiểm tra xung đột và đánh giá thủ công bổ sung.

## Kiểm chứng

Chạy `python -m pytest src/test_agents.py -q`: **6 passed**. Các test bao gồm read/write/edit `User.md`, compact lặp lại, nhớ qua thread và qua lần khởi tạo agent mới, xử lý lời đính chính, bỏ qua nhiễu và giảm prompt load trên thread dài.
