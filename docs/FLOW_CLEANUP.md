# Điều chỉnh luồng kiểm tra — 2026-09-05

## Phạm vi

Giữ nguyên các mốc chờ đèn, chu kỳ poll PLC, timeout và các model/ngưỡng đang cấu hình.
Giữ quy tắc production: Warning gửi OK cho PLC; mixing ưu tiên trong nhóm NG;
UV được chấp nhận khi không có đĩa UV hoặc số đĩa UV khớp White ở cả hai vùng.
Không thêm lệnh reset các bit kết quả: trách nhiệm reset phía PLC giữ như hiện tại.

## Thay đổi chính

- Caliper trả dữ liệu riêng cho mỗi lần đo, không lưu kết quả ảnh vào instance dùng chung.
- White, UV và Swagger dùng chung pipeline; debug chỉ thêm ảnh trung gian và dùng tham số yêu cầu.
- Response White dùng 0/1/2 rõ ràng, không chấp nhận boolean; thiếu định vị trả NG có mã lỗi và ảnh.
- APP bỏ qua xử lý UV của camera không định vị được khay sau một kết quả White NG hợp lệ.
  Timeout, response sai hoặc lỗi API là lỗi hệ thống: dừng chu kỳ, không giả lập NG sản phẩm.
- `InspectionSequence` sở hữu các tác vụ chụp, xử lý và tắt đèn; chờ các tác vụ hoàn tất trước khi thoát.
  Lệnh tắt đèn và gửi kết quả PLC phải trả thành công trước khi chuyển bước/ghi nhận hoàn thành.
- Một task poll PLC điều phối tuần tự; Stop/đóng ứng dụng chờ startup và chu kỳ hiện tại kết thúc.
- Camera được cấu hình Software Trigger và kiểm tra trạng thái khởi động.
- API đọc trigger chỉ đọc Modbus một lần; thao tác PLC dùng chung client được tuần tự hóa.
- Lưu ảnh dùng snapshot riêng của chu kỳ và hàng đợi; lỗi một ảnh không bỏ qua các ảnh sau.
  File JPEG dùng đuôi `.jpg`; tên file/thư mục bổ sung phần lẻ giây để tránh ghi đè.
- Debug xử lý lỗi bằng `try/finally`, sử dụng camera đang chọn, không ghi cứng Camera 1.
- HTTP dùng helper và DTO có kiểu; tài nguyên ảnh tạm được giải phóng theo vòng đời tác vụ.
- Sửa đọc/ghi số thập phân trong cấu hình theo invariant culture và nhận cả `KEY=value`/`KEY = value`.

## Kiểm chứng

Chạy trong thư mục `SERVICE`:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Chạy từ thư mục gốc project bằng MSBuild của Visual Studio 2022:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' APP\DiskInspection.sln /t:Build /p:Configuration=Debug /p:Platform=x64 /verbosity:minimal /nologo
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' APP\Tests\FlowTests.csproj /t:Build /verbosity:minimal /nologo
.\APP\Tests\bin\FlowTests.exe
```

Các test dùng ảnh giả lập và thiết bị giả: kiểm tra xử lý đồng thời, thứ tự đèn và timing,
lỗi chụp/API/đèn, cancellation, snapshot ảnh, contract API và đồng nhất debug/production.
Đã khởi tạo các model ONNX thật và kiểm tra một ảnh trống, không kết nối PLC/camera.
Đã đối chiếu response production cũ/mới trên fixture White OK, NG và Warning, gồm cả ảnh kết quả.

Chưa xác nhận với ảnh khay thực tế và phần cứng. Trước khi đưa vào máy, cần chạy các ca
OK/NG/Warning/mixing, mất kết nối, Stop giữa chu kỳ và xác nhận handshake với chương trình PLC.

## Bản build và triển khai

Build WPF được tạo trong `APP/DiskInspection/bin/x64/Debug`.
Service Python và APP phải được cập nhật cùng nhau vì contract lỗi và luồng bỏ qua UV đã thay đổi.
Lượt sửa này không thay thế `plugin/main.exe` trong bộ cài đã đóng gói và không chạy máy kiểm tra.
Nếu dùng bộ cài có `plugin/main.exe`, cần đóng gói lại service từ source mới trước khi triển khai.

Bản sao source trước sửa chỉ lưu cục bộ tại `.work/flow-cleanup-before.zip` và không nằm trong Git.
