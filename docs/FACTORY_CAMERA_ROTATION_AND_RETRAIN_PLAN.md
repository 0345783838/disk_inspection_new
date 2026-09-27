# Kế hoạch chuẩn hóa ảnh camera và train lại 3 model

Ngày rà soát: **2026-09-27**

## 1. Kết luận ngắn

Phương án nên triển khai là:

1. Giữ cách gá hiện tại của hai camera.
2. Chuẩn hóa ảnh về cùng hướng ngang như luồng cũ trước khi gọi AI:
   - **Camera 1: xoay 90° theo chiều kim đồng hồ.**
   - **Camera 2: xoay 90° ngược chiều kim đồng hồ.**
3. White và UV của cùng một camera phải dùng **chính xác cùng phép biến đổi**.
4. Chỉ giữ ba hàng point thuộc camera đó; vùng overlap với camera còn lại phải bị loại trước khi chia hàng.
5. Fine-tune/train lại cả ba model trên ảnh nhà máy sau khi đã chuẩn hóa:
   - `d_p_n.onnx`: point detector.
   - `c_p.onnx`: classifier `ok` / `ng` / `no_disk`.
   - `s_p.onnx`: disk segmentor.

Đây là phương án ít rủi ro hơn việc sửa toàn bộ luồng hình học để xử lý ảnh dọc. Phép xoay 90° bằng `cv2.rotate` chỉ là transpose + flip, không nội suy nên không làm giảm chất lượng ảnh.

## 2. Dữ liệu đã kiểm tra

### Ảnh mới tại nhà máy

- `C:/Users/CH Computer/Downloads/HA/Cam1`: 10 ảnh White.
- `C:/Users/CH Computer/Downloads/HA/Cam2`: 10 ảnh White.
- `C:/Users/CH Computer/Downloads/HA/UV1`: 1 ảnh UV Camera 1.
- `C:/Users/CH Computer/Downloads/HA/UV2`: 1 ảnh UV Camera 2.
- `C:/Users/CH Computer/Downloads/HA/HA`: 14 ảnh của một góc/FOV khác.
- Kích thước tất cả ảnh đã xem: `2592 x 1944`.

### Ảnh của luồng đã phát triển

- `F:/working/disk_checking/APP/test_input`.
- Kích thước: `2592 x 1944`.
- Ba hàng point nằm ngang, mỗi hàng 25 point.

Số ảnh hiện có chỉ đủ để xác nhận hướng xử lý và đánh giá khả thi. Chưa đủ độ đa dạng để train model dùng ổn định trong production.

## 3. Kết quả kiểm tra thực tế

### 3.1. Model detector hiện tại rất nhạy với hướng ảnh

Chạy thử `d_p_n.onnx` hiện tại với `confidence=0.1`, `IoU=0.1`:

| Dữ liệu | Hướng đưa vào model | Số box | Nhận xét |
|---|---:|---:|---|
| Ảnh cũ đã phát triển | nguyên bản | 75 | đúng 3 x 25 |
| Ảnh cũ đã phát triển | xoay 90° | 0 | model không bất biến theo phép xoay |
| Factory Cam2 mẫu | nguyên bản | 0 | không thể dùng thẳng ảnh dọc |
| Factory Cam2 mẫu | xoay clockwise | 94 | bắt cả vùng overlap/thừa |
| Factory Cam2 mẫu | xoay counter-clockwise | 79 | gần đúng ba hàng chính + box vùng thừa |

Kết luận: phải chuẩn hóa hướng ảnh trước inference. Không nên kỳ vọng augmentation xoay trong lúc train tự giải quyết toàn bộ vấn đề hình học ở runtime.

### 3.2. Camera 1

Sau khi xoay **clockwise**, trên 10 ảnh model hiện tại trả về `75-81` box. Ba cụm chính rất ổn định:

- Hàng 1: khoảng `y = 205`, 25 point.
- Hàng 2: khoảng `y = 1132`, 25 point.
- Hàng 3: khoảng `y = 2049`, 25 point.
- Cụm thừa/overlap: khoảng `y = 2323`, có khoảng 0-6 box tùy ảnh.

Đây là hướng chuẩn hóa đúng cho Camera 1. Vùng overlap sau khi xoay nằm ở phía dưới ảnh chuẩn hóa và cần loại khỏi tập box trước khi gọi `split_rows`.

Nếu xoay Camera 1 theo hướng ngược lại, model trả về khoảng `104-107` box và xuất hiện bốn cụm lớn; hướng này không phù hợp với ba hàng thuộc Camera 1.

### 3.3. Camera 2

Sau khi xoay **counter-clockwise**, trên 10 ảnh model hiện tại trả về `74-79` box. Các cụm điển hình:

- Hàng 1: khoảng `y = 61`, hiện detector cũ thường chỉ bắt được 23 point.
- Hàng 2: khoảng `y = 969`, 25 point.
- Hàng 3: khoảng `y = 1885`, 25 point.
- Cụm thừa/overlap: khoảng `y = 2178`, có khoảng 1-6 box tùy ảnh.

Đây là hướng chuẩn hóa đối xứng hợp lý với Camera 1 và đưa phần overlap về cuối ảnh. Tuy nhiên hàng đầu tiên nằm sát biên ảnh, cần xác minh riêng:

- Nếu đủ 25 point còn nhìn thấy trong ảnh raw, detector mới phải được label/train để lấy đủ 25.
- Nếu một phần point đã bị cắt vật lý ngoài FOV, model không thể phục hồi pixel không tồn tại. Khi đó phải chỉnh ROI/FOV trong giới hạn cơ khí hoặc thay rule bắt buộc 25 bằng logic slot hình học có kiểm soát.

Nếu xoay Camera 2 clockwise, ảnh tạo ra bốn cụm lớn và bắt nhiều vùng overlap hơn, không phù hợp với mục tiêu ba hàng của Camera 2.

### 3.4. Ảnh UV

Ảnh UV mới có cùng hướng dọc và cùng FOV với ảnh White tương ứng. Phản xạ trên đĩa và kim loại khác đáng kể so với dữ liệu cũ.

White đang sinh ra `CropBox`, `UvBox1`, `UvBox2`, `Mid1`, `Mid2`; các tọa độ này được dùng trực tiếp cho ảnh UV. Vì vậy:

- Không được chỉ xoay White mà không xoay UV.
- Không được crop White và UV bằng hai ROI khác nhau.
- Tốt nhất tạo một hàm chuẩn hóa theo camera rồi dùng lại cho cả hai loại ánh sáng.

Chỉ có một ảnh UV cho mỗi camera nên chưa đủ để đánh giá độ ổn định của threshold/mask UV trong nhiều điều kiện.

### 3.5. Thư mục `HA`

Ảnh trong `HA` có góc nhìn và tỷ lệ khay khác ảnh `Cam1/Cam2`. Không nên trộn trực tiếp chúng vào training set chính như cùng một domain.

Có thể dùng chúng làm:

- dữ liệu tham khảo;
- hard-negative cho detector nếu được label đúng;
- tập kiểm tra khả năng tổng quát riêng.

Nếu muốn dùng để train chính thức, cần xác nhận rõ ảnh đó đến từ cấu hình camera nào và normalize về đúng canonical frame trước.

## 4. Chi phí thời gian của phép xoay

Benchmark trên 10 ảnh BMP mới, kích thước `2592 x 1944`:

- Decode BMP trung bình: khoảng **12.9 ms/ảnh**.
- `cv2.rotate` 90° trung bình: khoảng **7.8 ms/ảnh**.
- Median rotate: khoảng **7.8 ms/ảnh**.
- P95 rotate: khoảng **8.8 ms/ảnh**.
- Max đo được: khoảng **10.2 ms/ảnh**.

Chi phí này nhỏ so với inference và thời gian bật/chờ đèn của một cycle. Nếu hai camera được xử lý song song, tác động lên cycle tổng còn nhỏ hơn nữa.

Ảnh BGR giải nén chiếm khoảng 15.1 MB. Cần dispose đúng lúc; không giữ đồng thời quá nhiều bản raw + rotated + rendered để tránh tăng RAM không cần thiết.

## 5. Thiết kế preprocessing đề xuất

### 5.1. Canonical frame

Đầu vào AI sau chuẩn hóa phải luôn thỏa mãn:

- Ba hàng point nằm ngang.
- Thứ tự trái sang phải nhất quán.
- Chỉ chứa ba hàng thuộc camera đang kiểm tra.
- White và UV của cùng camera có cùng kích thước và cùng hệ tọa độ pixel.

### 5.2. Biến đổi theo camera

| Camera | Rotation | Vùng cần bỏ sau xoay | Ghi chú |
|---|---|---|---|
| Cam1 | 90° clockwise | cụm overlap ở cuối/trục Y lớn | ba hàng chính hiện bắt đủ 25 |
| Cam2 | 90° counter-clockwise | cụm overlap ở cuối/trục Y lớn | hàng đầu nằm sát mép, phải xác minh đủ 25 point |

Không nên hard-code ROI cuối cùng chỉ từ 10 ảnh hiện tại. Trước tiên cần thu thêm ảnh khi máy chạy ở các vị trí X/Y biên, sau đó chốt ROI có margin an toàn.

### 5.3. Nơi nên đặt preprocessing

Ưu tiên đặt trong APP, ngay sau khi nhận frame từ camera và trước khi:

- hiển thị/saving ảnh dùng để debug;
- gọi `/check_disk_white`;
- gọi `/check_disk_uv`.

Lý do:

- APP biết frame đến từ Cam1 hay Cam2.
- API hiện tại không truyền `camera_id`, trong khi hai camera cần hai hướng xoay khác nhau.
- Service tiếp tục nhận canonical frame như luồng cũ, giảm thay đổi và rủi ro.

Phải dùng cùng helper ở cả production và Debug Form. Không được chỉ sửa `MainController`, nếu không debug và production sẽ cho kết quả khác nhau.

Gợi ý cấu trúc:

```text
raw frame Cam1/Cam2
        |
        v
NormalizeFrame(cameraId)
  - rotate theo camera
  - crop/mask vùng overlap theo cấu hình
  - giữ cùng transform cho White và UV
        |
        v
canonical horizontal frame
        |
        +--> White API --> geometry
        |
        +--> UV API + geometry White
```

Nếu buộc phải normalize trong SERVICE thì phải bổ sung `camera_id` hoặc `orientation` vào cả API White, UV và Debug. Không được đoán camera bằng kích thước ảnh vì hai camera có cùng resolution.

## 6. Thay đổi source cần làm trước/sau training

### 6.1. APP

- Tạo một helper duy nhất, ví dụ `CameraFrameNormalizer`.
- Cấu hình theo camera:
  - `Cam1Rotation = Clockwise90`.
  - `Cam2Rotation = CounterClockwise90`.
  - ROI/mask overlap riêng cho từng camera.
- Áp dụng helper cho cả White và UV.
- Áp dụng trong production và Debug Form.
- Lưu được cả raw frame và normalized frame trong chế độ debug để điều tra lỗi.
- Thêm unit test bằng ảnh nhỏ có marker ở bốn góc để xác nhận chiều xoay và ROI.
- Thêm integration test bảo đảm White/UV sau normalize cùng kích thước và mapping.

### 6.2. SERVICE

`split_rows` hiện chọn hai khoảng cách Y lớn nhất để chia ba hàng. Với cụm overlap nhỏ thứ tư, cách này có thể vẫn chia được ba nhóm nhưng nhóm cuối sẽ bị nhiễm các box overlap.

Cần sửa theo một trong hai cách, ưu tiên cách đầu:

1. APP crop/mask overlap chắc chắn trước API; SERVICE chỉ nhận đúng ba hàng.
2. SERVICE cluster tất cả các hàng trước, sau đó chọn đúng ba cluster phù hợp với cấu hình vị trí/khoảng cách/số point và loại cluster overlap.

Không nên chỉ kiểm tra tổng số box `>= 75`. Điều kiện cần kiểm tra ở mức từng hàng:

- đúng ba cluster được chọn;
- số point/slot mỗi hàng phù hợp;
- khoảng cách giữa ba hàng hợp lý;
- độ dốc và kích thước box không bất thường;
- box ngoài ROI không được đưa vào rectify/classify/segment.

Luồng hiện tại đã được sửa để xử lý ba hàng point và bốn mặt classify quanh ba hàng. Khi làm tiếp phải giữ contract này:

- detector: 3 hàng point;
- classifier: 4 vùng `(row1-bottom, row2-top, row2-bottom, row3-top)`;
- segmentor: 2 band giữa `row1-row2` và `row2-row3`;
- UV: 2 vùng được sinh từ hai hàng ngoài;
- `no_disk` là Warning, không đổi thành NG nếu spacing/count vẫn hợp lệ.

## 7. Kế hoạch dữ liệu và train lại ba model

### 7.1. Nguyên tắc chung

- Chỉ label/train trên ảnh đã qua đúng pipeline normalize sẽ chạy ở production.
- Không resize/crop thủ công khác với runtime mà không ghi lại cấu hình.
- Thu dữ liệu từ cả Cam1 và Cam2.
- Bao phủ thay đổi ánh sáng, phản xạ, vị trí X/Y, rung, bụi, khay khác nhau và thời điểm khác nhau.
- Chia train/validation/test theo **cycle/khay/phiên chụp**, không random từng crop từ cùng một frame, để tránh leakage.
- Có thể dùng tỷ lệ bắt đầu `70/15/15`; test set phải được khóa và không dùng để chỉnh threshold.

20 ảnh White hiện tại và 2 ảnh UV chỉ là tập feasibility. Cần thu thêm tối thiểu vài trăm full-frame đa dạng mỗi camera; riêng classifier cần hàng nghìn crop được cân bằng theo class nếu có thể. Chất lượng và độ đa dạng quan trọng hơn một con số cố định.

### 7.2. Model 1 - point detector `d_p_n.onnx`

Mục tiêu:

- Bắt đủ 25 point trên từng hàng thuộc camera.
- Không bắt point của vùng overlap đã bị loại.
- Ổn định khi hàng nằm gần biên ảnh, đặc biệt Cam2.

Công việc:

- Normalize tất cả ảnh trước khi label.
- Label point của đúng ba hàng sở hữu bởi camera.
- Nếu ROI vẫn còn vật thể/point thừa, label policy phải nhất quán: crop chúng khỏi ảnh hoặc đánh dấu ignore; không lúc label lúc bỏ.
- Bổ sung ảnh có thay đổi vị trí máy X/Y và các biên cơ khí.
- Bổ sung hard-negative gồm ốc, gối đỡ, phản xạ kim loại và vùng overlap.
- Fine-tune từ checkpoint hiện tại trước; chỉ train from scratch nếu fine-tune không hội tụ hoặc class/architecture thay đổi lớn.
- Export ONNX và kiểm tra output schema giống detector hiện tại.

Chỉ số cần theo dõi ngoài mAP:

- recall theo từng hàng;
- tỷ lệ frame có đủ 3 x 25 point;
- false positive trong vùng overlap;
- tỷ lệ sai thứ tự point sau sort;
- kết quả riêng Cam1 và Cam2.

### 7.3. Model 2 - classifier `c_p.onnx`

Classes hiện tại cần giữ:

- `ok`;
- `ng`;
- `no_disk`.

Dữ liệu classifier phải được sinh bằng đúng hàm crop runtime sau normalize + rectify, trên bốn mặt:

- row 1 - bottom;
- row 2 - top;
- row 2 - bottom;
- row 3 - top.

Công việc:

- Sinh crop từ cả hai camera và tất cả vị trí hàng/mặt.
- Kiểm tra orientation của từng crop, đặc biệt logic rotate crop phía top nếu code hiện tại đang dùng.
- Cân bằng class; không để `ok` áp đảo `ng` và `no_disk`.
- Thu đủ các loại NG thực tế, không gộp toàn bộ vào một ít ảnh mô phỏng.
- Kiểm tra confusion matrix và threshold theo class.
- Ưu tiên recall của `ng`; với `no_disk`, kiểm tra đúng policy trả Warning.
- Export ONNX và xác nhận index-to-label khớp `ClassifyResult` trong source.

### 7.4. Model 3 - segmentor `s_p.onnx`

Segmentor hiện xử lý hai band giữa ba hàng point. Dataset phải lấy đúng hai band sau normalize + rectify:

- band giữa row 1 và row 2;
- band giữa row 2 và row 3.

Công việc:

- Label mask biên/diện tích đĩa chính xác trong điều kiện ánh sáng White mới.
- Bao phủ phản xạ mạnh, nền kim loại, đĩa trong suốt, thiếu đĩa và lỗi thật.
- Giữ đúng cách chia ảnh lớn thành các phần như runtime.
- Đánh giá mask bằng IoU/Dice nhưng bắt buộc đánh giá thêm kết quả caliper cuối cùng.
- Kiểm tra số pair, khoảng cách min/max và tỷ lệ pass/NG trên held-out cycles.
- Export ONNX và kiểm tra kích thước input/output, postprocess và threshold.

### 7.5. UV không phải model ONNX riêng nhưng phải hiệu chỉnh lại

Luồng UV hiện dùng HSV threshold + làm sạch mask + caliper. Sau khi normalize và thay domain ảnh, cần:

- thu nhiều ảnh UV hơn cho mỗi camera;
- tune lại lower/upper HSV và min area trên validation set;
- xác minh số disk/pair UV khớp các midpoint từ White;
- kiểm tra vùng phản xạ sáng trên kim loại không trở thành false edge;
- không tune threshold trực tiếp trên test set.

## 8. Tiêu chí nghiệm thu đề xuất

### Preprocessing

- Cam1/Cam2 đều ra ảnh ngang đúng chiều.
- White/UV của cùng camera có cùng shape và cùng mapping.
- Vùng overlap không đi vào ba hàng được chọn.
- Không có crop làm mất point hợp lệ khi máy ở các vị trí X/Y cho phép.

### Detector

- Báo cáo riêng tỷ lệ frame đủ 25 point cho từng hàng, từng camera.
- Không có false positive overlap đi vào `split_rows`.
- Không chỉ xem tổng box; phải kiểm tra đúng ba cluster.
- Cam2 hàng sát biên phải được kiểm tra trên dữ liệu thật đủ lớn.

### Classifier

- Có precision/recall/F1 riêng cho `ok`, `ng`, `no_disk`.
- Các vị trí hàng/mặt và hai camera đều xuất hiện trong test set.
- `ng` không bị bỏ sót ở mức vượt ngưỡng nghiệp vụ thống nhất với QA.
- `no_disk` tạo Warning đúng như contract hiện tại.

### Segmentor + caliper

- So sánh mask và số pair với ground truth.
- So sánh min/max distance với đo tay hoặc golden sample.
- Đánh giá quyết định cuối OK/NG, không chỉ IoU mask.
- Kiểm tra riêng các ảnh phản xạ mạnh.

### End-to-end

- Debug và production cho cùng kết quả trên cùng một ảnh.
- Hai camera chạy song song, không đảo nhầm hướng.
- White geometry áp đúng lên UV.
- Chạy thử liên tục trên held-out production cycles; ghi latency, RAM và tỷ lệ lỗi theo camera.
- Phép xoay nên giữ dưới khoảng 15 ms/ảnh trên máy production; đo lại bằng Release build.

## 9. Checklist để Codex tiếp tục trên máy công ty

### Pha A - xác minh và chốt preprocessing

- [ ] Copy toàn bộ dữ liệu mới vào thư mục dataset có version; không sửa file raw.
- [ ] Xác nhận mapping chính xác `UV1 -> Cam1`, `UV2 -> Cam2`.
- [ ] Tạo script/command normalize batch với Cam1 CW và Cam2 CCW.
- [ ] Render contact sheet raw/normalized để người vận hành xác nhận chiều.
- [ ] Đếm thủ công Cam2 hàng sát biên: raw frame thực sự có đủ 25 point hay không.
- [ ] Thu thêm ảnh ở giới hạn X/Y và khi máy rung/chuyển vị trí.
- [ ] Chốt ROI/mask overlap có margin an toàn bằng config, không hard-code rải rác.

### Pha B - sửa APP/SERVICE

- [ ] Implement `CameraFrameNormalizer` dùng chung cho production và Debug Form.
- [ ] Áp cùng transform cho White và UV.
- [ ] Bổ sung config rotation/ROI theo camera.
- [ ] Bổ sung unit/integration test cho orientation, ROI và White/UV mapping.
- [ ] Làm `split_rows` robust với cluster thừa hoặc bảo đảm APP đã crop sạch overlap.
- [ ] Giữ nguyên contract ba hàng point/bốn mặt classify/hai band segment/hai vùng UV.
- [ ] Lưu raw + normalized trong debug để truy vết.

### Pha C - chuẩn bị dataset

- [ ] Tách dataset theo camera, light, date/session và trạng thái.
- [ ] Label detector trên canonical frame.
- [ ] Chạy pipeline sinh classifier crops rồi review/correct labels.
- [ ] Chạy pipeline sinh segment bands rồi label masks.
- [ ] Tạo split train/val/test theo session/tray.
- [ ] Viết dataset manifest chứa version preprocessing và class mapping.

### Pha D - train/export

- [ ] Fine-tune detector; đánh giá 3 x 25 và overlap FP.
- [ ] Fine-tune classifier 3 class; kiểm tra confusion matrix.
- [ ] Fine-tune segmentor; đánh giá caliper end-to-end.
- [ ] Tune UV HSV/min-area trên validation data mới.
- [ ] Export ba ONNX và chạy parity test giữa framework và ONNX Runtime.
- [ ] Không overwrite model production; lưu model theo version và checksum.

### Pha E - nghiệm thu tại máy

- [ ] Chạy Debug Form trên bộ test khóa.
- [ ] Chạy production flow trên cùng ảnh và so sánh kết quả.
- [ ] Chạy thử hai camera đồng thời.
- [ ] Kiểm tra memory/dispose và cycle latency.
- [ ] Chạy dry-run đủ số cycle đã thống nhất với QA/production.
- [ ] Chuẩn bị rollback về model/config cũ.

## 10. Cảnh báo quan trọng trước khi tiếp tục

Workspace hiện có các thay đổi chưa commit từ các công việc trước. Codex chạy tiếp không được reset hoặc ghi đè chúng:

- `APP/DiskInspection/Models/EnvironmentConfig.cs`
- `APP/DiskInspection/Utils/EnvReader.cs`
- `APP/DiskInspection/Views/DebugWindows/DebugWindow.xaml.cs`
- `APP/Tests/FlowTests.cs`
- `APP/Tests/FlowTests.csproj`
- `SERVICE/src/service/check_disk_service_yolo.py`
- `SERVICE/tests/test_inspection_flow.py`

Trước khi sửa tiếp, chạy `git status` và đọc diff hiện tại. Đặc biệt không được quay lại logic hai hàng point hoặc làm mất class `no_disk`.

## 11. Quyết định cuối cùng

**Có thể triển khai tốt bằng phương án xoay ảnh + loại overlap + fine-tune/train lại dữ liệu mới.** Phần xoay gần như không đáng kể về thời gian và giúp tái sử dụng phần lớn luồng hình học hiện tại.

Hai việc phải giải quyết trước khi gọi là production-ready:

1. Xác minh Camera 2 có đủ 25 point thật ở hàng sát mép hay đã bị cắt khỏi FOV.
2. Thu thêm dữ liệu White/UV đủ đa dạng; tập hiện tại quá nhỏ để kết luận độ ổn định của ba model và UV threshold.

