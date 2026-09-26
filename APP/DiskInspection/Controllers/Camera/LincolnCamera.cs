using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MvCamCtrl.NET;
using NLog;

namespace DiskInspection.Controllers.Camera
{
    public enum CameraWorkMode
    {
        LiveStream,       // Chụp liên tục, tự động đẩy ảnh lên UI
        SoftwareTrigger   // Nằm chờ lệnh, gọi hàm mới chụp 1 tấm
    }

    public class LincolnCamera : IDisposable
    {
        private readonly object _synLock = new object();
        private static readonly NLog.Logger _logger = NLog.LogManager.GetLogger("debug");

        private MyCamera _cam;
        public string SN { get; private set; } = "";

        // Buffer dùng chung
        private uint _nBufSizeForDriver = 0;
        private byte[] _pBufForDriver = new byte[0];
        private uint _nBufSizeForSaveImage = 0;
        private byte[] _pBufForSaveImage = new byte[0];

        // --- CÁC SỰ KIỆN CHO UI ---
        public event EventHandler<string> OnCameraError;
        public event EventHandler<Bitmap> OnLiveFrameReceived; // Sự kiện nhả ảnh khi ở chế độ Live Stream

        // Quan trọng: Phải giữ biến này ở cấp class để Garbage Collector không xoá mất callback
        private MyCamera.cbOutputExdelegate _imageCallbackDelegate;
        public bool IsGrabbing { get; private set; }

        public LincolnCamera(string serialNumber)
        {
            this.SN = serialNumber;
            _imageCallbackDelegate = new MyCamera.cbOutputExdelegate(ImageCallbackProcess);
            GetDeviceBySN(serialNumber);
        }

        public LincolnCamera(int index)
        {
            _imageCallbackDelegate = new MyCamera.cbOutputExdelegate(ImageCallbackProcess);
            var list = GetListCamInfo();
            if (index < list.Count)
            {
                this.SN = list[index].SN;
                GetDeviceByIdx(index, list);
            }
            else
            {
                NotifyError("Camera index out of range", 0);
            }
        }

        public bool IsOpen() => _cam != null;

        /// <summary>
        /// Liệt kê tất cả thiết bị camera trong mạng và USB
        /// </summary>
        public static List<CamInfo> GetListCamInfo()
        {
            List<CamInfo> listCamInfo = new List<CamInfo>();
            MyCamera.MV_CC_DEVICE_INFO_LIST pDeviceList = new MyCamera.MV_CC_DEVICE_INFO_LIST();

            int ret = MyCamera.MV_CC_EnumDevices_NET(MyCamera.MV_GIGE_DEVICE | MyCamera.MV_USB_DEVICE, ref pDeviceList);

            if (ret != MyCamera.MV_OK)
            {
                _logger.Warn($"Enumerate devices failed! Error code: {ret:X8}");
                return listCamInfo;
            }

            for (int i = 0; i < pDeviceList.nDeviceNum; i++)
            {
                MyCamera.MV_CC_DEVICE_INFO device = (MyCamera.MV_CC_DEVICE_INFO)Marshal.PtrToStructure(pDeviceList.pDeviceInfo[i], typeof(MyCamera.MV_CC_DEVICE_INFO));
                CamInfo camInfo = new CamInfo();

                if (device.nTLayerType == MyCamera.MV_GIGE_DEVICE)
                {
                    IntPtr buffer = Marshal.UnsafeAddrOfPinnedArrayElement(device.SpecialInfo.stGigEInfo, 0);
                    MyCamera.MV_GIGE_DEVICE_INFO gigeInfo = (MyCamera.MV_GIGE_DEVICE_INFO)Marshal.PtrToStructure(buffer, typeof(MyCamera.MV_GIGE_DEVICE_INFO));
                    camInfo.Name = !string.IsNullOrEmpty(gigeInfo.chUserDefinedName) ? gigeInfo.chUserDefinedName : gigeInfo.chManufacturerName;
                    camInfo.SN = gigeInfo.chSerialNumber;
                }
                else if (device.nTLayerType == MyCamera.MV_USB_DEVICE)
                {
                    IntPtr buffer = Marshal.UnsafeAddrOfPinnedArrayElement(device.SpecialInfo.stUsb3VInfo, 0);
                    MyCamera.MV_USB3_DEVICE_INFO usbInfo = (MyCamera.MV_USB3_DEVICE_INFO)Marshal.PtrToStructure(buffer, typeof(MyCamera.MV_USB3_DEVICE_INFO));
                    camInfo.Name = !string.IsNullOrEmpty(usbInfo.chUserDefinedName) ? usbInfo.chUserDefinedName : usbInfo.chManufacturerName;
                    camInfo.SN = usbInfo.chSerialNumber;
                }

                camInfo.DeviceIndex = i;
                camInfo.DeviceInfo = device;
                listCamInfo.Add(camInfo);
            }

            return listCamInfo;
        }

        private void GetDeviceBySN(string sn)
        {
            var list = GetListCamInfo();
            foreach (var info in list)
            {
                if (info.SN == sn)
                {
                    GetDeviceByIdx(info.DeviceIndex, list);
                    return;
                }
            }
            NotifyError($"Cannot find camera with SN: {sn}", 0);
        }

        private void GetDeviceByIdx(int idx, List<CamInfo> currentList)
        {
            try
            {
                if (idx >= currentList.Count) return;

                _cam = new MyCamera();
                MyCamera.MV_CC_DEVICE_INFO device = currentList[idx].DeviceInfo;

                int nRet = _cam.MV_CC_CreateDevice_NET(ref device);
                if (nRet != MyCamera.MV_OK)
                {
                    _cam = null;
                    NotifyError("Create device failed", nRet);
                    return;
                }

                nRet = _cam.MV_CC_OpenDevice_NET();
                if (nRet != MyCamera.MV_OK)
                {
                    _cam.MV_CC_DestroyDevice_NET();
                    _cam = null;
                    NotifyError("Device open fail. Might be used by another app.", nRet);
                    return;
                }

                // Chạy thiết lập mặc định ngay sau khi Open
                SetupDefault();
            }
            catch (Exception ex)
            {
                _logger.Error(ex, "Exception in GetDeviceByIdx");
            }
        }

        // Đoạn thiết lập khởi tạo mặc định:
        private void SetupDefault()
        {
            if (_cam == null) return;

            // Tối ưu mạng GigE
            int nPacketSize = _cam.MV_CC_GetOptimalPacketSize_NET();
            if (nPacketSize > 0) _cam.MV_CC_SetIntValue_NET("GevSCPSPacketSize", (uint)nPacketSize);

            // Cài đặt mặc định ban đầu là chế độ Continuous (Live) để sẵn sàng
            _cam.MV_CC_SetEnumValue_NET("AcquisitionMode", 2); // 2 = Continuous
            _cam.MV_CC_SetEnumValue_NET("TriggerMode", 0);     // 0 = Off

            // Đăng ký Callback hứng ảnh tự động
            _cam.MV_CC_RegisterImageCallBackEx_NET(_imageCallbackDelegate, IntPtr.Zero);
        }

        /// <summary>
        /// Hàm cốt lõi: Chuyển đổi linh hoạt giữa Live Stream và Trigger
        /// </summary>
        public bool SetWorkMode(CameraWorkMode mode)
        {
            if (_cam == null) return false;

            bool wasGrabbing = false;
            if (IsGrabbing)
            {
                Stop();
                wasGrabbing = true;
            }

            int nRet;
            if (mode == CameraWorkMode.LiveStream)
            {
                nRet = _cam.MV_CC_SetEnumValue_NET("TriggerMode", 0);
                // BẬT LẠI CALLBACK khi xem Live
                _cam.MV_CC_RegisterImageCallBackEx_NET(_imageCallbackDelegate, IntPtr.Zero);
            }
            else
            {
                nRet = _cam.MV_CC_SetEnumValue_NET("TriggerMode", 1);
                if (nRet == MyCamera.MV_OK)
                    nRet = _cam.MV_CC_SetEnumValue_NET("TriggerSource", 7);

                // CỰC KỲ QUAN TRỌNG: HUỶ CALLBACK ĐỂ TRÁNH CRASH KHI GỌI TRIGGER!
                if (nRet == MyCamera.MV_OK)
                    nRet = _cam.MV_CC_RegisterImageCallBackEx_NET(null, IntPtr.Zero);
            }

            if (nRet != MyCamera.MV_OK)
            {
                NotifyError("Lỗi khi chuyển chế độ Camera", nRet);
                return false;
            }

            if (wasGrabbing) return Start();
            return true;
        }

        public bool PrepareForInspection(float exposure)
        {
            if (!IsOpen() || !SetWorkMode(CameraWorkMode.SoftwareTrigger)) return false;
            if (_cam.MV_CC_SetFloatValue_NET("ExposureTime", exposure) != MyCamera.MV_OK) return false;
            return IsGrabbing || Start();
        }

        public void SetExposureTime(float value)
        {
            if (_cam != null) _cam.MV_CC_SetFloatValue_NET("ExposureTime", value);
        }

        public bool Start()
        {
            if (_cam == null || IsGrabbing) return false;
            int nRet = _cam.MV_CC_StartGrabbing_NET();
            if (nRet == MyCamera.MV_OK)
            {
                IsGrabbing = true;
                return true;
            }
            NotifyError("Lỗi Start Grabbing", nRet);
            return false;
        }

        public bool Stop()
        {
            if (_cam == null || !IsGrabbing) return false;
            _cam.MV_CC_StopGrabbing_NET();
            IsGrabbing = false;
            return true;
        }

        /// <summary>
        /// Dùng riêng cho chế độ Software Trigger (Pick & Place Auto)
        /// Bóp cò và lấy 1 ảnh trả về ngay lập tức.
        /// </summary>
        public Bitmap TriggerAndGetFrame(int timeoutMs = 1000)
        {
            if (_cam == null || !IsGrabbing) return null;

            lock (_synLock)
            {
                // 1. Phát lệnh chụp
                int nRet = _cam.MV_CC_SetCommandValue_NET("TriggerSoftware");
                if (nRet != MyCamera.MV_OK) return null;

                // 2. Chờ lấy ảnh
                return GrabFrameInternal(timeoutMs);
            }
        }

        /// <summary>
        /// Hàm này chạy ngầm dưới Background Thread do SDK tự gọi khi có ảnh (Chế độ Live Stream)
        /// </summary>
        private void ImageCallbackProcess(IntPtr pData, ref MyCamera.MV_FRAME_OUT_INFO_EX pFrameInfo, IntPtr pUser)
        {
            // Chỉ xử lý event nếu UI có đăng ký lắng nghe (để tiết kiệm CPU)
            if (OnLiveFrameReceived != null)
            {
                lock (_synLock) // Đảm bảo an toàn bộ nhớ chung
                {
                    Bitmap bmp = ConvertToBitmap(pData, ref pFrameInfo);
                    if (bmp != null)
                    {
                        OnLiveFrameReceived.Invoke(this, bmp);
                    }
                }
            }
        }

        // Hàm helper gom chung logic lấy ảnh từ bộ nhớ đệm
        private Bitmap GrabFrameInternal(int timeoutMs)
        {
            MyCamera.MVCC_INTVALUE stParam = new MyCamera.MVCC_INTVALUE();
            int nRet = _cam.MV_CC_GetIntValue_NET("PayloadSize", ref stParam);

            // Validate an toàn, tránh mảng rỗng
            if (nRet != MyCamera.MV_OK || stParam.nCurValue == 0)
            {
                _logger.Error("Lỗi: Không lấy được PayloadSize từ Camera.");
                return null;
            }

            EnsureBufferCapacity(stParam.nCurValue);

            MyCamera.MV_FRAME_OUT_INFO_EX stFrameInfo = new MyCamera.MV_FRAME_OUT_INFO_EX();

            // SỬ DỤNG GCHandle ĐỂ KHÓA CHẶT MẢNG BYTE TRONG RAM (Chống GC di chuyển)
            GCHandle handle = GCHandle.Alloc(_pBufForDriver, GCHandleType.Pinned);
            try
            {
                IntPtr pData = handle.AddrOfPinnedObject();
                nRet = _cam.MV_CC_GetOneFrameTimeout_NET(pData, _nBufSizeForDriver, ref stFrameInfo, timeoutMs);

                if (nRet != MyCamera.MV_OK)
                {
                    _logger.Warn($"Trigger rỗng hoặc Timeout. Mã lỗi: {nRet:X8}");
                    return null;
                }

                return ConvertToBitmap(pData, ref stFrameInfo);
            }
            finally
            {
                // Luôn luôn phải giải phóng Handle dù có lỗi hay không
                if (handle.IsAllocated) handle.Free();
            }
        }

        // Logic convert Pixel -> Bitmap tối ưu
        private Bitmap ConvertToBitmap(IntPtr pSrcData, ref MyCamera.MV_FRAME_OUT_INFO_EX stFrameInfo)
        {
            if (stFrameInfo.nFrameLen == 0) return null;
            EnsureBufferCapacity(stFrameInfo.nFrameLen);

            // Khóa RAM mảng nhận ảnh
            GCHandle handleDst = GCHandle.Alloc(_pBufForSaveImage, GCHandleType.Pinned);
            try
            {
                IntPtr pDstData = handleDst.AddrOfPinnedObject();

                MyCamera.MV_PIXEL_CONVERT_PARAM stConvertParam = new MyCamera.MV_PIXEL_CONVERT_PARAM
                {
                    nWidth = stFrameInfo.nWidth,
                    nHeight = stFrameInfo.nHeight,
                    pSrcData = pSrcData,
                    nSrcDataLen = stFrameInfo.nFrameLen,
                    enSrcPixelType = stFrameInfo.enPixelType,
                    pDstBuffer = pDstData,
                    nDstBufferSize = _nBufSizeForSaveImage
                };

                bool isMono = stFrameInfo.enPixelType.ToString().Contains("Mono");
                stConvertParam.enDstPixelType = isMono ? MyCamera.MvGvspPixelType.PixelType_Gvsp_Mono8 : MyCamera.MvGvspPixelType.PixelType_Gvsp_BGR8_Packed;

                if (_cam.MV_CC_ConvertPixelType_NET(ref stConvertParam) != MyCamera.MV_OK) return null;

                Bitmap bmp;
                if (isMono)
                {
                    bmp = new Bitmap(stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nWidth, PixelFormat.Format8bppIndexed, pDstData);
                    ColorPalette cp = bmp.Palette;
                    for (int i = 0; i < 256; i++) cp.Entries[i] = Color.FromArgb(i, i, i);
                    bmp.Palette = cp;
                }
                else
                {
                    bmp = new Bitmap(stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nWidth * 3, PixelFormat.Format24bppRgb, pDstData);
                }

                return CloneBitmap(bmp);
            }
            finally
            {
                if (handleDst.IsAllocated) handleDst.Free();
            }
        }

        private void EnsureBufferCapacity(uint payloadSize)
        {
            if (payloadSize > _nBufSizeForDriver)
            {
                _nBufSizeForDriver = payloadSize;
                _pBufForDriver = new byte[_nBufSizeForDriver];
                _nBufSizeForSaveImage = _nBufSizeForDriver * 3 + 2048;
                _pBufForSaveImage = new byte[_nBufSizeForSaveImage];
            }
        }

        private Bitmap CloneBitmap(Bitmap src)
        {
            Bitmap dest = new Bitmap(src.Width, src.Height, src.PixelFormat);
            dest.SetResolution(src.HorizontalResolution, src.VerticalResolution);
            using (Graphics g = Graphics.FromImage(dest)) g.DrawImageUnscaled(src, 0, 0);
            return dest;
        }

        private void NotifyError(string message, int errorCode)
        {
            string fullMsg = errorCode == 0 ? message : $"{message} (Code: {errorCode:X8})";
            _logger.Error(fullMsg);
            OnCameraError?.Invoke(this, fullMsg);
        }

        public void Close()
        {
            if (_cam != null)
            {
                Stop();
                _cam.MV_CC_CloseDevice_NET();
                _cam.MV_CC_DestroyDevice_NET();
                _cam = null;
            }
        }

        public void Dispose()
        {
            Close();
            GC.SuppressFinalize(this);
        }
    }

    public class CamInfo
    {
        public string Name { get; set; }
        public string SN { get; set; }
        public int DeviceIndex { get; set; }
        public MyCamera.MV_CC_DEVICE_INFO DeviceInfo { get; set; } // Giữ lại object device để khởi tạo nhanh
    }
}
