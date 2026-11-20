# Kiểm tra hai đĩa quá chéo bằng chênh khoảng cách caliper

Người dùng xác nhận so cùng một cặp đĩa tại hai vị trí đo và yêu cầu triển khai
ngày 09/11/2026. Ngưỡng đầu tiên 10 px được nâng lên 20 px; sau khi xem hai ảnh
chênh 21,5 px, người dùng yêu cầu nâng tiếp lên **30 px**.
Cho phép đĩa lệch nhẹ; không yêu cầu hai khoảng cách bằng nhau.

## Cấu hình và quyết định

Trong `SERVICE/config/config.env`:

```ini
MAX_CALIPER_GAP_DELTA_PX = 30
```

`abs(d1 - d2) > 30` thì NG; bằng 30 vẫn qua phép kiểm tra này. Giá trị phải hữu
hạn và không âm. Đọc khi SERVICE khởi động; sửa config rồi khởi động lại SERVICE.
Nếu dùng ứng dụng đóng gói, sửa `plugin/config/config.env` cạnh backend đang chạy.
Config cũ thiếu khóa này dùng mặc định 20 px. Debug cũng dùng config SERVICE,
không phụ thuộc việc cửa sổ Debug cũ có gửi tham số mới hay không.

Giữ nguyên phép đo hiện hành: khoảng cách Euclid giữa **tâm hai đĩa kề nhau**
trên ảnh band sau nắn phối cảnh, không phải độ rộng đĩa hay khe mép-mép.
Pixel thuộc ảnh band thực tế; đổi scale/nắn ảnh/vị trí caliper cần kiểm lại ngưỡng.
Không chuẩn hóa theo phần trăm độ rộng khe và không đổi sang góc vật lý.

Hai scanline trong band trên: caliper 1 (75% chiều cao) với 2 (25%).
Hai scanline trong band dưới: caliper 3 (25%) với 4 (75%).
Không so giữa hai band/camera, không so khoảng cách khác vị trí sau khi sort theo độ lớn.

## Ghép đúng đĩa, kết quả và ảnh hiển thị

Production ghép đĩa bằng connected component của mask segmentation đã làm sạch.
Một component phải xuất hiện đúng một lần ở mỗi scanline. Chỉ so một khoảng nếu
hai đĩa đó kề nhau trên **cả hai** scanline. Việc toàn bộ dải cùng nghiêng/dịch X
không tự làm đổi danh tính đĩa sang đĩa bên cạnh.

Nếu một điểm bị thiếu, component đứt hoặc nhiều đĩa dính chung, các khoảng không
ghép chắc chắn được sẽ bỏ qua và báo số lượng trong `skipped_gaps`/`unmatched_points`.
Các khoảng còn lại vẫn kiểm tra; không nối qua điểm thiếu để so nhầm khoảng khác.
Ảnh ghi `PARTIAL` khi chỉ so được một phần. Khi không còn khoảng hợp lệ hoặc
không đủ hai đĩa, trả NG do không đủ phép đo, phân biệt với NG do vượt ngưỡng.
Phép kiểm tra một phần không chứng minh mọi cặp đĩa đều đã được kiểm tra.

Giữ ưu tiên NG của classifier/count/khoảng cách cũ. Chỉ cần một khoảng chênh quá
30 px thì kết quả White là NG, kể cả trước đó classifier báo thiếu đĩa/Warning.
Giữ `ERROR_001` cho luồng PLC/API hiện có, bổ sung nguyên nhân trong `ErrorDesc`.
White production và Debug đều có `CaliperGapChecks`, gồm band, cặp caliper,
trạng thái, ngưỡng, từng khoảng/delta, khoảng lớn nhất và độ phủ ghép cặp.
Khoảng vượt ngưỡng được vẽ khung đỏ qua hai scanline, ghi `D=...`.

## Kiểm chứng

- 97 test runtime White/UV/caliper/geometry/segmentor qua, gồm biên 10/20/30 px,
  vượt biên, đĩa nghiêng cùng hướng, thiếu điểm, component dính/đứt, API Debug.
- Lượt thử **20 px** trên 28 ảnh White cũ, không camera/PLC:
  trước 4 OK / 15 NG / 9 Warning; sau 4 OK / 17 NG / 7 Warning.
  Hai ảnh Warning chuyển NG có khoảng chênh 21,5 px. Người dùng đã xem và nâng
  ngưỡng lên 30 px. Không có ảnh OK chuyển NG ở lượt 20 px.
  48/56 band không vượt ngưỡng, 8/56 band có khoảng vượt ngưỡng.
- Lượt cuối **30 px**: 4 OK / 15 NG / 9 Warning, không đổi kết quả tổng so với
  trước khi thêm kiểm tra. Hai ảnh 21,5 px giữ Warning thiếu đĩa. 55/56 band
  không vượt ngưỡng; một band Cam2 chênh 30,5 px vượt ngưỡng nhưng ảnh vốn NG.
  Median paired whole-software ratio 1,022x, max 1,324x; cùng giới hạn đo như trên.
- Hình học, counts, khoảng cách cũ và model/manifest hash không đổi;
  production/Debug khớp cả hai camera. Median paired whole-software ratio
  1,008x, max 1,198x trong lượt 20 px; không phải bảo đảm độ trễ máy thật.
- Kết quả trên ảnh lưu là kiểm chứng tích hợp và áp dụng đúng ngưỡng,
  không phải đánh giá độ chính xác với nhãn chéo đĩa đã xác nhận tại nhà máy.

Evidence: `.work/caliper_gap_delta_20261109/`: `runtime_tests.json`,
`white_replay.json`, `protected_before.json`, ảnh có đánh dấu và bản source trước sửa.
`white_replay_20px.json` và `*_threshold20.png` giữ nguyên lượt xem trước ở 20 px;
`white_replay.json` là lượt cuối theo config 30 px.
Hai file `Cam1_frame09_after.png`/`Cam1_frame10_after.png` đã gửi trong chat
cũng là ảnh lịch sử ở ngưỡng 20 px, không phải kết quả ngưỡng cuối.
`initial_x_matching_replay.json` là thử nghiệm ghép X ban đầu đã bị loại vì nhầm
đĩa khi cả dải nghiêng; kết quả chính thức dùng component mask.

Không có training, chỉnh dataset/nhãn hay thay model. Classifier hiện hành vẫn
là original HA best111; giữ nguyên mọi model retained. Bản ZIP ngày 07/11 là
lịch sử và không tự có kiểm tra mới chỉ vì source đã thay đổi.

Kiểm tra transport APP phát hiện config Debug cũ có namespace `xmlns:ns0` trên
root `configuration`, bị .NET ConfigurationManager từ chối. Chuyển namespace
về phần `assemblyBinding` đúng chuẩn; cây XML sau parse và toàn bộ giá trị,
binding redirect, camera/PLC giữ nguyên. Source `App.config` không đổi.
Backup và receipt: `DiskInspection.exe.config.before_namespace_repair`,
`config_namespace_repair.json` trong thư mục evidence. Bộ smoke lần này dùng
thư mục riêng, không sửa harness lịch sử.

SERVICE standalone mới đã kiểm tra 28 ảnh qua API, Debug cả hai camera, hai
request White đồng thời và actual APP transport/9 ca gộp White/UV; đã cài vào
`APP/DiskInspection/bin/Debug/plugin/` với config 30 px. Backend thử đã thoát.
Plugin cũ nằm tại `.work/caliper_gap_delta_20261109/backup_plugin/`.
`completion.json`, `packaged_verification.json` và `installed_inventory.json`
ghi nhận kết quả; Debug có `RELEASE_VERIFICATION.json`/`SHA256SUMS.txt` mới.
