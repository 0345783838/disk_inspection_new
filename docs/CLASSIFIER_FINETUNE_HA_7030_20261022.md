# Fine-tune HA trên canonical revision 10

Người dùng cho phép ngày 22/10/2026: “thử finetune đi”. Lượt này tại
`.work/classifier_finetune_ha_7030_20261022/`, báo cáo <http://127.0.0.1:8795/>.

Nguồn duy nhất là `DATASET_CLASSIFIER`, revision 10, 5.383 crop:
3.778 train / 1.605 validation. Nhãn, hash và split trùng hoàn toàn lượt
ImageNet `.work/classifier_camera_7030_legacy_20261022/`; xác nhận trong
`start_verification.json`. Không bổ sung ảnh, đổi nhãn hoặc chia lại dữ liệu.
Manifest gốc, ảnh input và code được đóng băng theo hash.

Khởi tạo từ HA PT đang dùng: `classifier_fullha_20261005/selected_v2/`
`c_p_ha_all_reviewed_20261005.pt`, SHA-256
`d8b82f801195cb9a3f24587f7cc5dac4e36200a1e6a336964bebf855a1835091`.
PT/ONNX parent khớp argmax trước train. Giữ head 3 lớp và đủ 236 tensor;
mọi trọng số được học, lr=1e-5, AdamW, cosine, max 300 epoch/patience 100.
Khóa running mean/variance/counter BatchNorm, vẫn học affine; 117 buffer
được kiểm tra nguyên vẹn mỗi lần save. Không dùng distillation hoặc stage khác.

Sampling và chọn best giữ nguyên lượt ImageNet: HA/new/legacy 40/40/20%,
hai camera cân bằng trong HA/new; class HA/new NG/no_disk/OK 30/30/40%,
legacy 70/15/15%. Best theo macro-F1 validation có trọng số 40/40/20%.
So best với HA gốc và ImageNet best trên cùng toàn bộ rows/split.

Gate không đổi: không tăng bỏ sót NG, no_disk thành OK, NG ảo hoặc tổng lỗi
trong từng nguồn và từng camera HA/new; weighted macro-F1 phải tăng.
Không tự triển khai model dù đạt gate. Kết quả cuối nằm trong `RESULTS.md`,
`release_decision.json`, `completion_verification.json`, `comparison.html`
và `error_review.html` của run. Pipeline còn kiểm tra PT/ONNX, replay
28 ảnh White với hardware tắt và hash file runtime.

HA validation chỉ đo khả năng giữ kiến thức, vì parent từng học HA.
Split theo vị trí rãnh vẫn chung frame context; dùng val chọn best nên đây
không phải test độc lập theo khay hoặc thời gian. Rà tính nhất quán nhãn còn
chưa hoàn tất. Không dùng kết quả train để xác nhận nhãn đúng.

Hoàn tất: 211 epoch, best 111, dừng sau 100 epoch không cải thiện. Trên cùng
validation, tổng lỗi HA gốc / ImageNet / fine-tune là 28 / 51 / 15; NG ảo
14 / 6 / 2, NG bỏ sót 1 / 31 / 7, no_disk → OK 11 / 8 / 0. Fine-tune tốt hơn
ImageNet về cả ba nguồn nhưng chưa vượt gate HA gốc vì bỏ sót NG tăng.
Giữ nguyên model cài đặt; không triển khai. Không resume lượt đã hoàn tất.

Kiểm tra hoàn tất đạt: integrity canonical, toàn bộ ảnh input và frozen code,
PT/ONNX argmax trên đủ 1.605 val, 28 White hardware-off, debug/production hai
camera, geometry/counts và hash runtime. Heartbeat đã tạm dừng.
Trang `val_best_errors.html` chỉ gồm 15 lỗi validation của fine-tune best;
`error_review.html` gồm khác biệt của cả ba model trên cả train và val.
