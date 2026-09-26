# Disk Inspection

Hệ thống kiểm tra đĩa gồm ứng dụng WPF (`APP`) điều khiển camera/PLC và dịch vụ Python (`SERVICE`) xử lý ảnh. Hai phần cần dùng cùng phiên bản vì giao tiếp và luồng xử lý được định nghĩa ở cả hai bên.

## Cấu trúc

- `APP/DiskInspection.sln`: ứng dụng .NET Framework 4.8, build x64 bằng Visual Studio 2022. `APP/dll` chứa DLL SDK camera cần cho build; NuGet khôi phục các gói trong `APP/packages`.
- `SERVICE/main.py`: API Python, cấu hình tại `SERVICE/config/config.env`. Ba model được cấu hình chạy nằm trong `SERVICE/config/models` và được lưu trong Git.
- `APP/Tests` và `SERVICE/tests`: kiểm tra luồng bằng thiết bị/ảnh giả lập.
- `docs/FLOW_CLEANUP.md`: mô tả luồng đã chỉnh, lệnh kiểm chứng và các bước cần xác nhận trên máy thật.

## Chạy và kiểm chứng

Từ thư mục `SERVICE`, tạo môi trường Python, cài `requirements.txt`, rồi chạy `python main.py`. Service dùng đường dẫn tương đối trong `config.env`, vì vậy cần chạy từ thư mục này. Ứng dụng WPF cần camera, PLC và các thiết lập máy phù hợp để chạy đầy đủ.

Các lệnh build và test nằm trong [tài liệu luồng](docs/FLOW_CLEANUP.md). Thư mục `APP/DiskInspection/bin` là đầu ra build/bộ cài cũ và không được commit. Nếu triển khai bằng bộ cài có `plugin/main.exe`, cần đóng gói lại service từ source hiện tại.
