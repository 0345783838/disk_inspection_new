using DiskInspection.Controllers.APIs;
using DiskInspection.Controllers.Camera;
using DiskInspection.Controllers.PLC;
using DiskInspection.Models;
using DiskInspection.Security;
using DiskInspection.Utils;
using DiskInspection.Views.ActivationWindows;
using Emgu.CV;
using Emgu.CV.Structure;
using NLog;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace DiskInspection.Controllers
{
    /// <summary>
    /// Điều phối toàn bộ luồng kiểm tra: PLC trigger → chụp ảnh → AI → kết quả.
    /// </summary>
    class MainController
    {
        // ─── Dependencies ────────────────────────────────────────────────────────
        private static readonly Logger _logger = LogManager.GetCurrentClassLogger();
        private readonly Properties.Settings _param = Properties.Settings.Default;
        private readonly MainWindow _mainWindow;

        // ─── Camera ──────────────────────────────────────────────────────────────
        private CameraManager _cameraManager;
        private LincolnCamera _camera1;
        private LincolnCamera _camera2;

        // ─── Timers ───────────────────────────────────────────────────────────────
        private System.Timers.Timer _statusTimer;

        // ─── State ────────────────────────────────────────────────────────────────
        private volatile bool _isRunning;
        private readonly object _stateGate = new object();
        private Task _inspectionLoop = Task.CompletedTask;
        private Task _stopTask = Task.CompletedTask;
        private Task _startup = Task.CompletedTask;
        private bool _stopRequested;
        private readonly InspectionImageWriter _imageWriter = new InspectionImageWriter();
        private CancellationTokenSource _inspectCts;
        public bool ServiceIsRunning { get; private set; }

        // ─── Last captured frames (dùng để update UI và lưu ảnh) ─────────────────
        // Mỗi cặp (origin, result) được bảo vệ bởi 1 lock duy nhất
        private readonly object _cam1Lock = new object();
        private readonly object _cam2Lock = new object();

        private BitmapSource _cam1WhiteOrigin, _cam1WhiteResult;
        private BitmapSource _cam1UvOrigin, _cam1UvResult;
        private BitmapSource _cam2WhiteOrigin, _cam2WhiteResult;
        private BitmapSource _cam2UvOrigin, _cam2UvResult;

        // ─── Constants ───────────────────────────────────────────────────────────
        private const int PlcPollIntervalMs = 50;
        private const int StatusPollIntervalMs = 2000;
        private const string LicensePath = @"plugin\license.dat";

        // ─────────────────────────────────────────────────────────────────────────

        public MainController(MainWindow window)
        {
            _mainWindow = window;

        }

        // ═════════════════════════════════════════════════════════════════════════
        // 1. KHỞI ĐỘNG CHƯƠNG TRÌNH
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Khởi động AI engine và chờ nó sẵn sàng trong khoảng thời gian timeout.
        /// </summary>
        public bool StartAIService(int timeoutMs, string loadingMessage)
        {
            ServiceIsRunning = false;
            _mainWindow.SetLoadingService(loadingMessage);
            _logger.Info("Starting AI service...");
            AppLogger.Instance.Info("Loading Program...", "SYSTEM");

            AIServiceController.CloseProcessExisting();
            AIServiceController.Start();

            int steps = timeoutMs / 1000;
            for (int i = 0; i < steps; i++)
            {
                Thread.Sleep(1000);
                if (APICommunication.CheckAPIStatus(_param.ApiUrlAi, 200))
                {
                    _logger.Info("AI service started successfully.");
                    AppLogger.Instance.Info("Loaded Program Successfully!", "SYSTEM");
                    ServiceIsRunning = true;
                    return true;
                }
            }

            _logger.Error("AI service failed to start within timeout.");
            return false;
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 2. BẮT ĐẦU KIỂM TRA
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Kiểm tra điều kiện và bắt đầu vòng lặp kiểm tra.
        /// </summary>
        public bool Start()
        {
            TaskCompletionSource<bool> startup;
            lock (_stateGate)
            {
                if (!_inspectionLoop.IsCompleted || !_stopTask.IsCompleted || !_startup.IsCompleted)
                    return false;
                startup = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                _startup = startup.Task;
                _stopRequested = false;
            }
            Thread.Sleep(500);
            try
            {
                // PLC communication is hosted by the AI service and must be connected after it starts.
                if (!CheckAndStartAI() || !CheckAndStartPLC() || !CheckAndStartCamera())
                {
                    ShutdownAllLights();
                    StopAllCameras();
                    return false;
                }
                lock (_stateGate)
                {
                    if (_stopRequested) return false;
                    _inspectCts?.Dispose();
                    _inspectCts = new CancellationTokenSource();
                    var token = _inspectCts.Token;
                    _isRunning = true;
                    StartStatusTimer();
                    _inspectionLoop = Task.Run(() => RunInspectionLoopAsync(token));
                }
                AppLogger.Instance.Info("Cameras, PLC and AI are ready.", "SYSTEM");
                return true;
            }
            catch (Exception ex)
            {
                ShutdownAllLights();
                StopAllCameras();
                ShowAndLogError($"System startup failed: {ex.Message}", "SYSTEM");
                return false;
            }
            finally { startup.TrySetResult(true); }
        }

        // ─── Kiểm tra từng hệ thống con ──────────────────────────────────────────

        private bool CheckAndStartAI()
        {
            if (APICommunication.CheckAPIStatus(_param.ApiUrlAi))
                return true;

            if (!_mainWindow.ShowWarning("AI engine is not running. Restart?")) return false;
            bool restarted = StartAIService(timeoutMs: 20000, "Restarting AI engine...");

            if (!restarted)
                _mainWindow.ShowError("Failed to restart AI engine. Please contact vendor.");

            return restarted;
        }

        private bool CheckAndStartCamera()
        {
            //return true;
            _cameraManager = CameraManager.GetInstance();
            _camera1 = _cameraManager.GetCamera1();
            _camera2 = _cameraManager.GetCamera2();

            if (!_camera1.IsOpen())
            {
                _camera1 = null;
                _mainWindow.ShowError($"Cannot open Camera 1 (SN: {_param.Cam1Sn})");
                return false;
            }
            if (!_camera2.IsOpen())
            {
                _camera2 = null;
                _mainWindow.ShowError($"Cannot open Camera 2 (SN: {_param.Cam2Sn})");
                return false;
            }

            if (!_camera1.PrepareForInspection(_param.Cam1Exposure) ||
                !_camera2.PrepareForInspection(_param.Cam2Exposure))
            {
                _mainWindow.ShowError("Cannot configure cameras for software trigger.");
                return false;
            }
            return true;
        }

        private bool CheckAndStartPLC()
        {
            //return true;
            if (PlcController.CheckPlcConnection(_param.ApiUrlCom))
                return true;

            bool connected = PlcController.ConnectPlc(_param.ApiUrlCom, _param.PlcIp, _param.PlcPort);
            if (!connected)
                _mainWindow.ShowError("Cannot connect to PLC. Please check the connection.");

            return connected;
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 3. VÒNG LẶP PLC — một task sở hữu toàn bộ chu kỳ
        // ═════════════════════════════════════════════════════════════════════════

        private async Task RunInspectionLoopAsync(CancellationToken token)
        {
            string fault = null;
            try
            {
                while (true)
                {
                    await Task.Delay(PlcPollIntervalMs, token).ConfigureAwait(false);
                    var trigger = await Task.Run(() => PlcController.CheckTrigger(_param.ApiUrlCom, 1000)).ConfigureAwait(false);
                    token.ThrowIfCancellationRequested();
                    if (trigger.Item1 == TriggerState.Error)
                        throw new InvalidOperationException("Cannot read PLC trigger.");
                    if (trigger.Item2)
                        await RunInspectionCycleAsync(token).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException) when (token.IsCancellationRequested)
            {
                AppLogger.Instance.Info("Inspection cancelled.", "SYSTEM");
            }
            catch (Exception ex)
            {
                fault = ex.Message;
                _logger.Error(ex, "Inspection stopped due to a system fault.");
                AppLogger.Instance.Error(fault, "SYSTEM");
            }
            finally
            {
                _isRunning = false;
                StopStatusTimer();
                if (!await Task.Run(() => ShutdownAllLights()).ConfigureAwait(false))
                    fault = (fault == null ? "" : fault + " ") + "Cannot confirm that all lights are off.";
                StopAllCameras();
                if (fault != null) _mainWindow.NotifyInspectionFault(fault);
            }
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 4. MỘT CHU KỲ KIỂM TRA ĐẦY ĐỦ
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Thực hiện 1 chu kỳ kiểm tra hoàn chỉnh:
        ///   1. Reset trigger
        ///   2. Chụp ảnh (White → UV) — tuần tự theo đèn
        ///   3. Gửi AI — 2 camera song song
        ///   4. Cập nhật UI và PLC
        /// </summary>
        private async Task RunInspectionCycleAsync(CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            if (!PlcController.ResetTrigger(_param.ApiUrlCom, 1000))
                throw new InvalidOperationException("Cannot reset PLC trigger.");

            _mainWindow.UpdateInspectingMode();
            App.ImageViewer.ClearImages();
            App.ImageViewer.HideViewer();
            ClearCycleImages();
            Bitmap white1 = null, white2 = null, uv1 = null, uv2 = null;
            InspectionResponse whiteResult1 = null, whiteResult2 = null;
            CameraInspectionResult cam1Result = null, cam2Result = null;
            var cycleTime = Stopwatch.StartNew();
            var sequence = new InspectionSequence(ControlLightAsync);
            try
            {
                await sequence.RunAsync(
                    async ct =>
                    {
                        (white1, white2) = await CaptureFromBothCamerasAsync(ct).ConfigureAwait(false);
                        if (white1 == null || white2 == null)
                            throw new InvalidOperationException("Camera did not return both White frames.");
                        UpdateCapturedFrameUI(white1, white2, InspectionLight.White);
                    },
                    async ct =>
                    {
                        (whiteResult1, whiteResult2) = await RunWhiteAIAsync(white1, white2, ct).ConfigureAwait(false);
                        ct.ThrowIfCancellationRequested();
                        UpdateUIWithWhiteResults(whiteResult1, whiteResult2);
                    },
                    async ct =>
                    {
                        (uv1, uv2) = await CaptureFromBothCamerasAsync(ct).ConfigureAwait(false);
                        if (uv1 == null || uv2 == null)
                            throw new InvalidOperationException("Camera did not return both UV frames.");
                        UpdateCapturedFrameUI(uv1, uv2, InspectionLight.Uv);
                    },
                    async ct =>
                    {
                        (cam1Result, cam2Result) = await RunUVAIAsync(uv1, uv2, whiteResult1, whiteResult2, ct).ConfigureAwait(false);
                        ct.ThrowIfCancellationRequested();
                    },
                    _param.WaitWhiteLightOn, _param.WaitWhiteLightOff,
                    _param.WaitUvLightOn, _param.WaitUvLightOff, token).ConfigureAwait(false);

                token.ThrowIfCancellationRequested();
                cycleTime.Stop();
                cam1Result.Duration = cam2Result.Duration = cycleTime.Elapsed;
                var finalStatus = cam1Result.Result == InspectionResult.Failed || cam2Result.Result == InspectionResult.Failed
                    ? InspectionResult.Failed
                    : cam1Result.Result == InspectionResult.Warning || cam2Result.Result == InspectionResult.Warning
                        ? InspectionResult.Warning : InspectionResult.Passed;

                // Preserve production policy: Warning sends OK; mixing takes priority among NG results.
                bool acknowledged = finalStatus != InspectionResult.Failed
                    ? PlcController.OnOkSignal(_param.ApiUrlCom)
                    : cam1Result.UvErrorCode == ErrorCode.ERROR_003 || cam2Result.UvErrorCode == ErrorCode.ERROR_003
                        ? PlcController.OnErrorMixing(_param.ApiUrlCom)
                        : PlcController.OnErrorAbnormal(_param.ApiUrlCom);
                if (!acknowledged)
                    throw new InvalidOperationException("PLC did not acknowledge the inspection result.");

                UpdateUIWithUvResults(cam1Result, cam2Result);
                _mainWindow.UpdateInspectionStatus(finalStatus);
                _mainWindow.UpdateStatistics(finalStatus);
                var timestamp = _mainWindow.UpdateTimeStamp();
                if (finalStatus == InspectionResult.Failed) App.ImageViewer.ShowFirstErrorImage();
                QueueImages(timestamp, finalStatus);
                AppLogger.Instance.Info($"Inspection {finalStatus}; cycle completed in {cycleTime.ElapsedMilliseconds} ms.", "SYSTEM");
            }
            finally
            {
                white1?.Dispose();
                white2?.Dispose();
                uv1?.Dispose();
                uv2?.Dispose();
            }
        }

        private async Task<(Bitmap cam1, Bitmap cam2)> CaptureFromBothCamerasAsync(CancellationToken token)
        {
            token.ThrowIfCancellationRequested();

            var tasks = new[]
            {
                Task.Run(() => _camera1.TriggerAndGetFrame()),
                Task.Run(() => _camera2.TriggerAndGetFrame())
            };
            try
            {
                var frames = await Task.WhenAll(tasks).ConfigureAwait(false);
                return (frames[0], frames[1]);
            }
            catch
            {
                foreach (var task in tasks)
                    if (task.Status == TaskStatus.RanToCompletion) task.Result?.Dispose();
                throw;
            }
        }

        private Task<bool> ControlLightAsync(InspectionLight light, bool on)
        {
            return Task.Run(() => light == InspectionLight.White
                ? PlcController.ControlWhiteLight(_param.ApiUrlCom, on, 1000)
                : PlcController.ControlUvLight(_param.ApiUrlCom, on, 1000));
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 6. XỬ LÝ AI
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Kết quả kiểm tra của 1 camera (gộp White + UV).
        /// </summary>

        private class CameraInspectionResult
        {
            public InspectionResult Result { get; set; }
            public TimeSpan Duration { get; set; }

            public BitmapSource UvResultBitmap { get; set; }

            public double MinDiskDistance { get; set; }
            public double MaxDiskDistance { get; set; }
            public int UvDiskCount { get; set; }

            public string WhiteErrorDesc { get; set; }
            public string WhiteErrorCode { get; set; }
            public string UvErrorDesc { get; set; }
            public string UvErrorCode { get; set; }
            public int WhitePassed { get; set; }
            public bool UvPassed { get; set; }
        }

        /// <summary>
        /// Gửi AI White cho cả 2 camera song song.
        /// Lỗi API được chuyển thành lỗi hệ thống; không giả lập kết quả NG.
        /// </summary>
        private async Task<(InspectionResponse cam1, InspectionResponse cam2)> RunWhiteAIAsync(
            Bitmap frame1, Bitmap frame2, CancellationToken token)
        {
            var results = await Task.WhenAll(Inspect(frame1), Inspect(frame2)).ConfigureAwait(false);
            return (results[0], results[1]);

            async Task<InspectionResponse> Inspect(Bitmap frame)
            {
                using (var image = new Image<Bgr, byte>(frame))
                {
                    var result = await APICommunication.InspectWhiteLightAsync(_param.ApiUrlAi, image.Mat, token).ConfigureAwait(false);
                    if (result == null || result.Result < 0 || result.Result > 2 || string.IsNullOrWhiteSpace(result.ResImg))
                        throw new InvalidOperationException("Invalid White inspection response.");
                    if (!result.HasUvGeometry && result.Result != (int)InspectionResult.Failed)
                        throw new InvalidOperationException("White inspection returned no UV geometry.");
                    return result;
                }
            }
        }

        private async Task<(CameraInspectionResult cam1, CameraInspectionResult cam2)> RunUVAIAsync(
            Bitmap frame1, Bitmap frame2, InspectionResponse white1, InspectionResponse white2, CancellationToken token)
        {
            var results = await Task.WhenAll(Inspect(frame1, white1), Inspect(frame2, white2)).ConfigureAwait(false);
            return (BuildResult(white1, results[0]), BuildResult(white2, results[1]));

            async Task<InspectionUvResponse> Inspect(Bitmap frame, InspectionResponse white)
            {
                // A tray rejected before localization cannot be inspected under UV.
                if (!white.HasUvGeometry) return null;
                using (var image = new Image<Bgr, byte>(frame))
                {
                    var result = await APICommunication.InspectUvLightAsync(_param.ApiUrlAi, image.Mat,
                        white.CropBox, white.UvBox1, white.UvBox2, white.Mid1, white.Mid2, token).ConfigureAwait(false);
                    if (result == null || string.IsNullOrWhiteSpace(result.ResImg))
                        throw new InvalidOperationException("Invalid UV inspection response.");
                    return result;
                }
            }
        }

        /// <summary>
        /// Ghép kết quả White và UV thành 1 object kết quả hoàn chỉnh cho 1 camera.
        /// </summary>
        private static CameraInspectionResult BuildResult(
            InspectionResponse white, InspectionUvResponse uv)
        {
            if (uv == null)
            {
                if (white.HasUvGeometry || white.Result != (int)InspectionResult.Failed)
                    throw new InvalidOperationException("UV inspection did not complete.");
                return new CameraInspectionResult
                {
                    Result = InspectionResult.Failed, WhitePassed = white.Result,
                    WhiteErrorCode = white.ErrorCode, WhiteErrorDesc = white.ErrorDesc,
                    MinDiskDistance = white.MinDiskDistance, MaxDiskDistance = white.MaxDiskDistance,
                    UvErrorDesc = "UV skipped: White inspection could not locate the tray."
                };
            }

            InspectionResult res;
            if (white.Result == (int)InspectionResult.Failed || !uv.Result )
                res = InspectionResult.Failed;
            else if (white.Result == (int)InspectionResult.Warning && uv.Result)
                res = InspectionResult.Warning;
            else
                res = InspectionResult.Passed;

            return new CameraInspectionResult
            {
                Result = res,
                UvResultBitmap = Converter.Base64ToBitmapSource(uv.ResImg),
                MinDiskDistance = white.MinDiskDistance,
                MaxDiskDistance = white.MaxDiskDistance,
                UvDiskCount = uv.CountUvDisk,
                WhitePassed = white.Result,
                WhiteErrorCode = white.ErrorCode,
                WhiteErrorDesc = white.ErrorDesc,
                UvPassed = uv.Result,
                UvErrorDesc = uv.ErrorDesc,
                UvErrorCode = uv.ErrorCode
            };
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 7. CẬP NHẬT UI
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Cập nhật UI ảnh gốc ngay sau khi chụp.
        /// </summary>
        private void UpdateCapturedFrameUI(Bitmap frame1, Bitmap frame2, InspectionLight light)
        {
            if (frame1 == null)
            {
                ShowAndLogError("Captured frame 1 null", "CAM1");
                return;
            }
            if (frame2 == null)
            {
                ShowAndLogError("Captured frame 2 null", "CAM2");
                return;
            }
            if (light == InspectionLight.White)
            {  
                lock (_cam1Lock)
                {
                   
                        
                    _cam1WhiteOrigin = Converter.BitmapToBitmapSource(frame1);
                    App.ImageViewer.AddImage(_cam1WhiteOrigin, "1-White-Origin", ThumbStatus.Origin, "Camera 1 - White - Original");
                }
                lock (_cam2Lock)
                {
                    
                        
                    _cam2WhiteOrigin = Converter.BitmapToBitmapSource(frame2);
                    App.ImageViewer.AddImage(_cam2WhiteOrigin, "2-White-Origin", ThumbStatus.Origin, "Camera 2 - White - Original");
                }
                _mainWindow.UpdateCam1WhiteOrigin(_cam1WhiteOrigin);
                _mainWindow.UpdateCam2WhiteOrigin(_cam2WhiteOrigin);
            }
            else
            {
                lock (_cam1Lock)
                {
                    _cam1UvOrigin = Converter.BitmapToBitmapSource(frame1);
                    App.ImageViewer.AddImage(_cam1UvOrigin, "1-UV-Origin", ThumbStatus.Origin, "Camera 1 - UV - Original");
                }
                lock (_cam2Lock)
                {
                    _cam2UvOrigin = Converter.BitmapToBitmapSource(frame2);
                    App.ImageViewer.AddImage(_cam2UvOrigin, "2-UV-Origin", ThumbStatus.Origin, "Camera 2 - UV - Original");
                }
                _mainWindow.UpdateCam1UvOrigin(_cam1UvOrigin);
                _mainWindow.UpdateCam2UvOrigin(_cam2UvOrigin);
            }
        }

        /// <summary>
        /// Cập nhật UI ảnh kết quả White ngay sau khi AI White xong.
        /// Dùng InspectionResponse trực tiếp vì chưa có CameraInspectionResult đầy đủ.
        /// </summary>
        private void UpdateUIWithWhiteResults(InspectionResponse white1, InspectionResponse white2)
        {
            if (white1?.ResImg != null)
            {
                var bitmap = Converter.Base64ToBitmapSource(white1.ResImg);
                lock (_cam1Lock) { _cam1WhiteResult = bitmap; }
                _mainWindow.UpdateCam1WhiteResult(_cam1WhiteResult);
                _mainWindow.UpdateCam1MinMaxDis(white1.MinDiskDistance, white1.MaxDiskDistance);

                ThumbStatus status;
                if (white1.Result == (int)InspectionResult.Failed)
                {
                    status = ThumbStatus.Ng;
                }
                else if (white1.Result == (int)InspectionResult.Warning)
                {
                    status = ThumbStatus.Warning;
                }
                else
                {
                    status = ThumbStatus.Ok;
                }
                App.ImageViewer.AddImage(_cam1WhiteResult, "1-White-Result", status, $"CAM1 White: {white1.ErrorDesc}");

            }

            if (white2?.ResImg != null)
            {
                var bitmap = Converter.Base64ToBitmapSource(white2.ResImg);
                lock (_cam2Lock) { _cam2WhiteResult = bitmap; }
                _mainWindow.UpdateCam2WhiteResult(_cam2WhiteResult);
                _mainWindow.UpdateCam2MinMaxDis(white2.MinDiskDistance, white2.MaxDiskDistance);

                ThumbStatus status;
                if (white2.Result == (int)InspectionResult.Failed)
                {
                    status = ThumbStatus.Ng;
                }
                else if (white2.Result == (int)InspectionResult.Warning)
                {
                    status = ThumbStatus.Warning;
                }
                else
                {
                    status = ThumbStatus.Ok;
                }

                App.ImageViewer.AddImage(_cam2WhiteResult, "2-White-Result", status, $"CAM2 White: {white2.ErrorDesc}");
            }
        }

        /// <summary>
        /// Cập nhật UI ảnh kết quả UV + tất cả thông tin tổng hợp sau khi AI UV xong.
        /// </summary>
        private void UpdateUIWithUvResults(CameraInspectionResult cam1, CameraInspectionResult cam2)
        {
            _mainWindow.UpdateInspectionStatusCam1(cam1.Result);
            _mainWindow.UpdateCam1ProcessedTime(cam1.Duration);
            _mainWindow.UpdateInspectionStatusCam2(cam2.Result);
            _mainWindow.UpdateCam2ProcessedTime(cam2.Duration);
            if (cam1.UvResultBitmap != null)
            {
                lock (_cam1Lock) { _cam1UvResult = cam1.UvResultBitmap; }
                _mainWindow.UpdateCam1UvResult(_cam1UvResult);
                _mainWindow.UpdateCam1DiskUv(cam1.UvDiskCount);
                var status = cam1.UvPassed ? ThumbStatus.Ok : ThumbStatus.Ng;
                App.ImageViewer.AddImage(_cam1UvResult, "1-UV-Result", status, $"CAM1 UV: {cam1.UvErrorDesc}");
            }

            if (cam2.UvResultBitmap != null)
            {
                lock (_cam2Lock) { _cam2UvResult = cam2.UvResultBitmap; }
                _mainWindow.UpdateCam2UvResult(_cam2UvResult);
                _mainWindow.UpdateCam2DiskUv(cam2.UvDiskCount);
                var status = cam2.UvPassed ? ThumbStatus.Ok : ThumbStatus.Ng;
                App.ImageViewer.AddImage(_cam2UvResult, "2-UV-Result", status, $"CAM2 UV: {cam2.UvErrorDesc}");
            }
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 8. LƯU ẢNH
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Lưu ảnh theo SaveMode. Chạy trong background thread, không block UI.
        /// </summary>
        private void ClearCycleImages()
        {
            lock (_cam1Lock)
            lock (_cam2Lock)
            {
                _cam1WhiteOrigin = _cam1WhiteResult = _cam1UvOrigin = _cam1UvResult = null;
                _cam2WhiteOrigin = _cam2WhiteResult = _cam2UvOrigin = _cam2UvResult = null;
            }
            _mainWindow.UpdateCam1WhiteResult(null);
            _mainWindow.UpdateCam2WhiteResult(null);
            _mainWindow.UpdateCam1UvResult(null);
            _mainWindow.UpdateCam2UvResult(null);
            _mainWindow.UpdateCam1DiskUv(0);
            _mainWindow.UpdateCam2DiskUv(0);
        }

        private void QueueImages(DateTime timestamp, InspectionResult status)
        {
            if (!_param.SaveEnable) return;
            var option = (SaveOption)_param.SaveOption;
            bool passed = status == InspectionResult.Passed;
            if ((option == SaveOption.OK && !passed) || (option == SaveOption.NG && passed)) return;
            var images = new Dictionary<string, BitmapSource>();
            var mode = (SaveType)_param.SaveMode;
            lock (_cam1Lock)
            lock (_cam2Lock)
            {
                if (mode != SaveType.RESULT)
                {
                    Add("Cam1_White_Original", _cam1WhiteOrigin); Add("Cam2_White_Original", _cam2WhiteOrigin);
                    Add("Cam1_UV_Original", _cam1UvOrigin); Add("Cam2_UV_Original", _cam2UvOrigin);
                }
                if (mode != SaveType.ORIGINAL)
                {
                    Add("Cam1_White_Result", _cam1WhiteResult); Add("Cam2_White_Result", _cam2WhiteResult);
                    Add("Cam1_UV_Result", _cam1UvResult); Add("Cam2_UV_Result", _cam2UvResult);
                }
            }
            _imageWriter.Enqueue(_param.SavePath, timestamp, images);
            void Add(string name, BitmapSource image)
            {
                if (image != null) images.Add(name, image);
            }
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 9. STATUS TIMER — kiểm tra trạng thái kết nối định kỳ
        // ═════════════════════════════════════════════════════════════════════════

        public void StartStatusTimer()
        {
            lock (_stateGate)
            {
                if (_statusTimer != null) return;
                _statusTimer = new System.Timers.Timer(StatusPollIntervalMs) { AutoReset = false };
                _statusTimer.Elapsed += OnStatusTimerElapsed;
                _statusTimer.Start();
            }
        }

        private void OnStatusTimerElapsed(object sender, EventArgs e)
        {
            try
            {
                if (!_isRunning) return;
                _mainWindow.SetStatusService(APICommunication.CheckAPIStatus(_param.ApiUrlAi, 1000),
                    PlcController.CheckPlcConnection(_param.ApiUrlCom, 1000),
                    _camera1?.IsOpen() == true, _camera2?.IsOpen() == true);
            }
            catch (Exception ex) { _logger.Warn(ex, "Status update failed."); }
            finally
            {
                lock (_stateGate)
                {
                    if (_isRunning) _statusTimer?.Start();
                }
            }
        }

        public void StopStatusTimer()
        {
            lock (_stateGate)
            {
                if (_statusTimer == null) return;
                _statusTimer.Stop();
                _statusTimer.Elapsed -= OnStatusTimerElapsed;
                _statusTimer.Dispose();
                _statusTimer = null;
            }
        }

        public Task StopAsync()
        {
            lock (_stateGate)
            {
                if (!_stopTask.IsCompleted) return _stopTask;
                _stopRequested = true;
                _isRunning = false;
                _inspectCts?.Cancel();
                StopStatusTimer();
                _stopTask = FinishStopAsync(_inspectionLoop, _startup);
                return _stopTask;
            }
        }

        private async Task FinishStopAsync(Task loop, Task startup)
        {
            await startup.ConfigureAwait(false);
            await loop.ConfigureAwait(false);
            await Task.Run(() => ShutdownAllLights()).ConfigureAwait(false);
            StopAllCameras();
            await _imageWriter.FlushAsync().ConfigureAwait(false);
        }

        public async Task ShutdownAsync()
        {
            await StopAsync().ConfigureAwait(false);
            CloseCamera();
            CloseAIService();
        }

        internal void CloseAIService()
        {
            AIServiceController.CloseProcessExisting();
            ServiceIsRunning = false;
        }

        internal void CloseCamera()
        {
            StopAllCameras(andClose: true);
        }

        internal void ShutdownLight()
        {
            ShutdownAllLights();
        }

        private void StopAllCameras(bool andClose = false)
        {
            foreach (var cam in new[] { _camera1, _camera2 })
            {
                if (cam == null || !cam.IsOpen()) continue;
                cam.Stop();
                if (andClose) cam.Close();
            }
        }

        private bool ShutdownAllLights()
        {
            // Attempt both commands even when the first fails.
            bool whiteOff = PlcController.ControlWhiteLight(_param.ApiUrlCom, false);
            bool uvOff = PlcController.ControlUvLight(_param.ApiUrlCom, false);
            if (!whiteOff || !uvOff)
                AppLogger.Instance.Error("PLC did not confirm that all lights are off.", "PLC");
            return whiteOff && uvOff;
        }

        // ═════════════════════════════════════════════════════════════════════════
        // 11. LICENSE
        // ═════════════════════════════════════════════════════════════════════════

        internal bool CheckLicense()
        {
            if (!File.Exists(LicensePath))
                return ShowActivationWindow();

            var (isValid, message) = LicenseManager.ValidateActivationKey(File.ReadAllText(LicensePath));

            if (isValid)
            {
                AppLogger.Instance.Info("License is valid.", "SYSTEM");
                _logger.Info("License is valid.");
                return true;
            }

            AppLogger.Instance.Error(message, "SYSTEM");
            _logger.Error("License is not valid. Please contact vendor.");
            _mainWindow.ShowError("License is not valid. Please contact vendor.");
            return ShowActivationWindow();
        }

        private bool ShowActivationWindow()
        {
            bool activated = false;
            _mainWindow.Dispatcher.Invoke(() =>
            {
                var win = new ActivationWindow { Topmost = true };
                if (win.ShowDialog() == true)
                {
                    AppLogger.Instance.Info("Activation successful.", "SYSTEM");
                    _mainWindow.ShowInfo("Activation key is valid. Continue using the program.");
                    activated = true;
                }
                else
                {
                    AppLogger.Instance.Error("Activation failed or cancelled.", "SYSTEM");
                    _mainWindow.ShowError("License is not valid. Please contact vendor.");
                }
            });
            return activated;
        }

        // ═════════════════════════════════════════════════════════════════════════
        // HELPERS
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Log lỗi vào cả NLog lẫn AppLogger và hiển thị lên UI cùng 1 lúc.
        /// </summary>
        private void ShowAndLogError(string message, string tag)
        {
            _logger.Error(message);
            AppLogger.Instance.Error(message, tag);
            _mainWindow.ShowError(message);
        }
    }
}
