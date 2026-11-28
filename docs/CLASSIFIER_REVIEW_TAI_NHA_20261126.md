# Chạy lại bộ rà nhãn classifier ở nhà

Ngày bàn giao 26/11/2026. Ảnh mới đã có sẵn trên máy ở nhà. Mục tiêu là chạy lại replay White offline với best17, tạo crop đúng như SERVICE và mở giao diện để người dùng xác nhận hoặc sửa nhãn. Dừng ở bước review cho đến khi người dùng xác nhận đã rà xong và yêu cầu train.

## Tin nhắn để gửi Codex ở nhà

> Đọc toàn bộ file hướng dẫn này. Ảnh mới đã có trên máy ở nhà tại `<điền đường dẫn thư mục ảnh>`. Kiểm tra workspace và model hiện có. Chạy lại replay White offline, dựng giao diện rà nhãn như mô tả và gửi tao link local đã kiểm tra. Tao sẽ tự xác nhận hoặc sửa nhãn. Chờ tao báo xong rồi mới chia train/val và fine-tune.

## Những gì workspace ở nhà cần có

| Thành phần | Dùng để làm gì |
|---|---|
| Thư mục ảnh mới trên máy nhà | Dùng trực tiếp làm đầu vào. `ANH2111.zip` chỉ cần nếu đó là nơi chứa ảnh hoặc nếu muốn đối chiếu CRC với bản ZIP. |
| `SERVICE/src/`, `SERVICE/config/config.env`, `SERVICE/requirements.txt` | Luồng xử lý White, cấu hình và môi trường Python. Dùng đúng bản source/config từ máy xưởng để tái lập. |
| `APP/DiskInspection/bin/Debug/DiskInspection.exe.config` | Lấy rotation và ROI. Cấu hình đã dùng là Cam1 xoay 90° thuận chiều kim đồng hồ, Cam2 xoay 90° ngược chiều, ROI trống cho cả hai. Nếu chỉ có `APP/DiskInspection/App.config`, đối chiếu bốn giá trị này trước khi dùng. |
| Ba file ONNX bên dưới | Best17 để gợi ý nhãn, detector để tìm vị trí rãnh và segmenter để chạy đến hết luồng White. |
| `SERVICE/training/classifier_dataset.py`, `adopt_classifier_dataset.py`, `export_current_classifier_crops.py` trong thư mục `SERVICE/training/` | Ghi nhãn qua canonical, tính hash ảnh giải mã và dùng lại hàm chuẩn hóa ảnh. |
| Các file trong `.work/classifier_new_data_20261126/` | Lấy trực tiếp `prepare.py`, `finalize.py`, `index.html`, `review.js`, `review_store.py`, `serve_review.py` và `README_VI.md` nếu workspace ở nhà thiếu phần `.work/`. Có thể tạo lại `frames/`, `items.json`, CSV và báo cáo từ ảnh sẵn ở nhà. Lấy `verify_review.py` và `verify_ui.cjs` nếu muốn chạy lại kiểm thử lưu nhãn. |
| `DATASET_CLASSIFIER/` và `.work/ACTIVE_CLASSIFIER_DATASET.json` | Một dataset canonical có thể ghi ở máy nhà. Sửa `path` trong registry sang đường dẫn tuyệt đối thực tế. Đọc revision từ `manifest.json`. |

Ba file ONNX đã dùng đều nằm trong `SERVICE/config/models/`. Kiểm tra bằng `Get-FileHash <đường_dẫn> -Algorithm SHA256`.

| Model | SHA-256 |
|---|---|
| `c_p.onnx` · YOLO11n-cls best17 ngày 19/11, lượt train bỏ legacy | `79f95f774a9bd4fed783eec3b7d577a5d6d4ec347ec4bb70e4b05997a418b38e` |
| `d_p_ha_20260930.onnx` · detector rãnh | `b04223c1ce295416e4c7e6f7bf6342499be7b87a81689ac8ceee06e19c413bbe` |
| `s_p_edge_yolo11n_20261120.onnx` · segmenter White | `9509091f61fea6f9c21cc7fad2527e5e25c99e53995bb6a9741cfbbc2bb07996` |

Gói `classifier-home-no-legacy-r228-best17.zip` ngày 25/11 có checkpoint PT best17, snapshot **15.358 train / 6.635 val** và mã tham chiếu. Gói đó thiếu ba ONNX ở trên, mã runtime đầy đủ và bộ review ngày 26/11. `DATASET_BEST17_NO_LEGACY/` trong gói là snapshot `is_canonical=false`. Nếu máy nhà chỉ có snapshot này và người dùng chọn dùng nó làm nền, Codex cần tạo đúng một `DATASET_CLASSIFIER/` làm việc bằng cách giữ nguyên ID, nhãn, hash, nhóm và split của 21.993 bản ghi, khởi tạo manifest canonical, catalog và CSV bằng mã quản lý dataset, rồi đăng ký đường dẫn mới. Giữ nguyên snapshot gốc. Nếu ở nhà đã có canonical mới hơn, dùng manifest hiện tại và bảo toàn các sửa đổi ở đó. Nếu thiếu runtime hoặc model, báo chính xác file còn thiếu trước khi chạy replay.

## Chạy lại replay

1. Kiểm kê `*_Cam1_White_Raw.png` và `*_Cam2_White_Raw.png` theo ngày, lượt chụp và camera. Chỉ đưa **White Raw** vào classifier. UV và Result JPG có thể để lại trong nguồn để đối chiếu khi cần. Ở máy xưởng, ảnh được xếp theo `NEW_DATA/<ngày>/<lượt chụp>/<tên ảnh>`. Trỏ `SOURCE` trong `prepare.py` vào thư mục ảnh trên máy nhà. Nếu cách xếp thư mục khác, điều chỉnh cùng quy tắc kiểm kê ở `prepare.py` và `finalize.py`. Nếu ảnh đã giải nén và không có ZIP, cho phần đối chiếu CRC chạy có điều kiện và ghi đúng số ảnh đã đối chiếu CRC trong báo cáo. Vẫn lưu SHA-256 của từng White Raw, kiểm tra khả năng giải mã và báo rõ số cặp Cam1/Cam2 đủ hoặc thiếu. Với batch ảnh khác, dùng origin phù hợp để tránh trùng ID.
2. Tạo môi trường Python trong workspace. Replay gốc dùng Python 3.10.1, OpenCV 4.6.0, NumPy 1.26.4 và ONNX Runtime 1.23.2 trên CPU. Cài theo `SERVICE/requirements.txt`. Chạy `DiskCheckingService.check_disk_white` trực tiếp trên ảnh đã áp dụng rotation/ROI và vòng mã hóa, giải mã JPEG bằng OpenCV để mô phỏng APP. Chặn PLC bằng các hook trong `prepare.py`. Giữ detector, hoàn thiện vị trí slot và crop của SERVICE. Crop rộng **1,8**, vùng top xoay **180°**, classifier nhận **224×224** và thứ tự lớp `ng`, `no_disk`, `ok`. Lưu crop PNG đúng pixel đầu vào classifier, cả ba xác suất và ảnh ngữ cảnh có đánh dấu slot. Báo riêng ảnh SERVICE dừng trước khi thu đủ crop.
3. Khởi chạy trong root workspace. Dùng thư mục kết quả sạch chỉ chứa các file mã cần chạy. Lượt thử hai ảnh giúp kiểm tra rotation, crop, slot và model trước khi chạy hết. Script bỏ qua từng frame đã có `replay.json` khi chạy tiếp với cùng đầu vào. Đối với mã copy từ máy xưởng, tạo lại `protected_before.json` ở máy nhà. Nếu workspace mới chưa có `.work/ACTIVE_CLASSIFIER_TRAINING.json`, cho phép bỏ qua riêng file bảo vệ tùy chọn đó và ghi rõ trong log. Các hash của manifest, catalog, CSV, config và model vẫn được kiểm tra.

```powershell
$review = '.work\classifier_new_data_20261126'
$py = '.\SERVICE\.venv\Scripts\python.exe'
& $py -u -B "$review\prepare.py" --limit 2
# Kiểm tra hai frame thử, sau đó chạy hết.
& $py -u -B "$review\prepare.py"
& $py -u -B "$review\finalize.py"
& $py -u -B "$review\serve_review.py" --port 8828
```

Nếu giữ nguyên đúng batch ở máy xưởng, mốc đối chiếu là **270 lượt, 540 ảnh White, 54.000 crop**. Best17 gợi ý **47 NG**, **3.863 Trống**, **50.090 OK**. Trong 47 NG có 24 Cam1 và 23 Cam2. Đây là gợi ý để review. Nếu đếm khác, kiểm tra nguồn, config, phiên bản tiền xử lý và hash model, rồi báo nguyên nhân hoặc phần còn chưa đối chiếu được. Chuẩn bị khoảng **6,4 GB** cho ảnh đầu ra của lượt đủ 540 frame, cộng dung lượng môi trường và dataset.

## Mở giao diện và lưu nhãn

- Mở <http://127.0.0.1:8828/> trên máy nhà. Kiểm tra lưới crop, ngữ cảnh rãnh, ảnh khay và trang kết quả replay. Mặc định ưu tiên ca báo NG, ca có cờ hình học và confidence thấp. Bộ lọc **OK kiểm tra mẫu** lấy tối đa 50 crop OK confidence từ 95% trở lên từ các ảnh khác nhau cho mỗi ngày/camera. Batch gốc có 300 mẫu.
- **1 NG**, **2 Trống**, **3 OK**, **Enter** lưu xác nhận nhãn đang hiện, **U** chưa rõ, **X** loại ảnh, **Ctrl+Z** hoàn tác. Mỗi lần lưu thật đi qua `CanonicalDataset.save` để manifest, thư mục lớp, catalog và CSV đồng bộ. Nhãn hợp lệ của crop mới có `role=reference_only` và `split=null`. Ca chưa rõ và loại nằm trong `excluded/`. Những gợi ý chưa được xác nhận vẫn ở trang review. Việc bấm bỏ qua chỉ chuyển ảnh.
- Khi kiểm thử thao tác ghi, dùng dataset thử. Nếu mang theo hai file kiểm thử ở trên, chạy `verify_review.py` ngoài tiến trình server chính với Python và Node.js. Trên dataset thật, xác nhận API hiển thị đúng revision và tổng crop rồi bàn giao cho người dùng. Trang chỉ cần chạy trên loopback `127.0.0.1`.

Nếu sau này đưa nhãn từ máy nhà về máy xưởng, chuyển **manifest canonical ở nhà, `items.json` của lượt replay và các crop đã rà cần nhập**. Giữ nguyên byte của `items.json` khi nén hoặc giải nén. Khi ảnh White Raw đã có ở máy xưởng, dùng đường dẫn local tới ảnh có cùng SHA-256. Đối chiếu từng ID, hash và provenance với manifest hiện hành, rồi cập nhật đường dẫn nguồn về máy nhận. Báo các trường hợp xung đột hoặc khác pixel để người dùng quyết định. Áp dụng các nhãn đã xác nhận bằng `CanonicalDataset.save` và bảo toàn các sửa đổi mới tại máy nhận.

Training là bước riêng sau khi người dùng xác nhận. Lúc đó chốt đúng revision và hash manifest, chia crop mới theo nhóm khay/lượt chụp/hai camera và các ảnh trùng hoặc lật, rồi so với best17 trên cùng tập đánh giá. Ưu tiên lọt NG, sau đó NG ảo và Trống → OK, báo theo camera. Ở bước bàn giao này chỉ cần replay xong và mở được giao diện rà nhãn.
