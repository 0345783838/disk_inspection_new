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
        private readonly CameraFrameNormalizer _frameNormalizer;

        // ─── Timers ───────────────────────────────────────────────────────────────
        private System.Timers.Timer _statusTimer;

        // ─── State ────────────────────────────────────────────────────────────────
        private volatile bool _isRunning;
        private readonly object _stateGate = new object();
        private Task _inspectionLoop = Task.CompletedTask;
        private Task _stopTask = Task.CompletedTask;
        private Task _startup = Task.CompletedTask;
        private bool _stopRequested;
        private bool _hardwareMayBeActive;
        private readonly InspectionImageWriter _imageWriter = new InspectionImageWriter();
        private CancellationTokenSource _inspectCts;
        public bool ServiceIsRunning { get; private set; }

        // ─── Last captured frames (dùng để update UI và lưu ảnh) ─────────────────
        // Mỗi cặp (origin, result) được bảo vệ bởi 1 lock duy nhất
        private readonly object _cam1Lock = new object();
        private readonly object _cam2Lock = new object();

        private BitmapSource _cam1WhiteRaw, _cam1WhiteOrigin, _cam1WhiteResult;
        private BitmapSource _cam1UvRaw, _cam1UvOrigin, _cam1UvResult;
        private BitmapSource _cam2WhiteRaw, _cam2WhiteOrigin, _cam2WhiteResult;
        private BitmapSource _cam2UvRaw, _cam2UvOrigin, _cam2UvResult;

        // ─── Constants ───────────────────────────────────────────────────────────
        private const int PlcPollIntervalMs = 50;
        private const int StatusPollIntervalMs = 2000;
        private const string LicensePath = @"plugin\license.dat";

        // ─────────────────────────────────────────────────────────────────────────

        public MainController(MainWindow window)
        {
            _mainWindow = window;
            _frameNormalizer = CameraFrameNormalizer.FromAppSettings();
        }

        // The legacy settings/debug windows can also connect/control hardware.
        internal void MarkHardwareMayBeActive() { _hardwareMayBeActive = true; }

        // ═════════════════════════════════════════════════════════════════════════
        // 1. KHỞI ĐỘNG CHƯƠNG TRÌNH
        // ═════════════════════════════════════════════════════════════════════════

        /// <summary>
        /// Khởi động AI engine và chờ nó sẵn sàng trong khoảng thời gian timeout.
        /// </summary>
        public bool StartAIService(int timeoutMs, string loadingMessage)
        {
            ServiceIsRunning = false;
            _mainWindow.SetLoadingService(loadingMessage, timeoutMs + 1000);
            _logger.Info("Starting AI service...");
            AppLogger.Instance.Info("Loading Program...", "SYSTEM");

            AIServiceController.CloseProcessExisting();
            AIServiceController.Start();

            var startupTime = Stopwatch.StartNew();
            while (startupTime.ElapsedMilliseconds < timeoutMs)
            {
                Thread.Sleep((int)Math.Min(500, Math.Max(1, timeoutMs - startupTime.ElapsedMilliseconds)));
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
                _hardwareMayBeActive = true;
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
            await PlcTriggerReset.RunAsync(
                () => PlcController.ResetTrigger(_param.ApiUrlCom, 1000), token).ConfigureAwait(true);

            _mainWindow.UpdateInspectingMode();
            App.ImageViewer.ClearImages();
            App.ImageViewer.HideViewer();
            ClearCycleImages();
            Bitmap whiteRaw1 = null, whiteRaw2 = null, uvRaw1 = null, uvRaw2 = null;
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
                        (whiteRaw1, whiteRaw2) = await CaptureFromBothCamerasAsync(ct).ConfigureAwait(false);
                        if (whiteRaw1 == null || whiteRaw2 == null)
                            throw new InvalidOperationException("Camera did not return both White frames.");
                        white1 = _frameNormalizer.Normalize(whiteRaw1, 1);
                        white2 = _frameNormalizer.Normalize(whiteRaw2, 2);
                        UpdateCapturedFrameUI(whiteRaw1, whiteRaw2, white1, white2, InspectionLight.White);
                    },
                    async ct =>
                    {
                        (whiteResult1, whiteResult2) = await RunWhiteAIAsync(white1, white2, ct).ConfigureAwait(false);
                        ct.ThrowIfCancellationRequested();
                        UpdateUIWithWhiteResults(whiteResult1, whiteResult2);
                    },
                    async ct =>
                    {
                        (uvRaw1, uvRaw2) = await CaptureFromBothCamerasAsync(ct).ConfigureAwait(false);
                        if (uvRaw1 == null || uvRaw2 == null)
                            throw new InvalidOperationException("Camera did not return both UV frames.");
                        uv1 = _frameNormalizer.Normalize(uvRaw1, 1);
                        uv2 = _frameNormalizer.Normalize(uvRaw2, 2);
                        ValidateWhiteUvMapping(white1, uv1, 1);
                        ValidateWhiteUvMapping(white2, uv2, 2);
                        UpdateCapturedFrameUI(uvRaw1, uvRaw2, uv1, uv2, InspectionLight.Uv);
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
                whiteRaw1?.Dispose();
                whiteRaw2?.Dispose();
                uvRaw1?.Dispose();
                uvRaw2?.Dispose();
                white1?.Dispose();
                white2?.Dispose();
                uv1?.Dispose();
                uv2?.Dispose();
            }
        }

        private static void ValidateWhiteUvMapping(Bitmap white, Bitmap uv, int cameraId)
        {
            if (white == null || uv == null || white.Size != uv.Size)
                throw new InvalidOperationException(
                    $"Camera {cameraId} White/UV normalized frames do not share the same dimensions.");
        }

        private async Task<(Bitmap cam1, Bitmap cam2)> CaptureFromBothCamerasAsync(CancellationToken token)
        {
            token.ThrowIfCancellationRequested();

            var tasks = new[]
            {
                Task.Run(() => _camera1.TriggerAndGetFrame(timeoutMs:3000)),
                Task.Run(() => _camera2.TriggerAndGetFrame(timeoutMs:3000))
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
            var results = await Task.WhenAll(InspectWhiteFrameAsync(frame1, token),
                InspectWhiteFrameAsync(frame2, token)).ConfigureAwait(false);
            return (results[0], results[1]);
        }

        private async Task<InspectionResponse> InspectWhiteFrameAsync(Bitmap frame, CancellationToken token)
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

        /// <summary>
        /// Local RAW White frame -> the SAME normalization/API/UI as production.
        /// No camera, lights, PLC trigger/result, UV, counters or production image saving.
        /// Instead of commenting TriggerAndGetFrame, use Tools > Test local RAW White.
        /// </summary>
        public Task InspectLocalWhiteAsync(string path, int cameraId)
        {
            if (cameraId != 1 && cameraId != 2) throw new ArgumentOutOfRangeException(nameof(cameraId));
            if (!File.Exists(path)) throw new FileNotFoundException("Local image not found.", path);
            lock (_stateGate)
            {
                if (_isRunning || !_inspectionLoop.IsCompleted || !_stopTask.IsCompleted || !_startup.IsCompleted)
                    throw new InvalidOperationException("Stop inspection before testing a local image.");
                _inspectCts?.Dispose();
                _inspectCts = new CancellationTokenSource();
                var token = _inspectCts.Token;
                // Stop/Shutdown joins this completion even if the API fails; the caller still receives the error.
                var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                _inspectionLoop = completion.Task;
                return Task.Run(async () =>
                {
                    try { await RunLocalWhiteAsync(path, cameraId, token).ConfigureAwait(false); }
                    finally { completion.TrySetResult(true); }
                });
            }
        }

        private async Task RunLocalWhiteAsync(string path, int cameraId, CancellationToken token)
        {
            token.ThrowIfCancellationRequested();
            App.ImageViewer.ClearImages();
            App.ImageViewer.HideViewer();
            ClearCycleImages(updateUi: false);
            _mainWindow.PrepareLocalWhiteUI(cameraId);
            var elapsed = Stopwatch.StartNew();
            // This is the local-image replacement for camera.TriggerAndGetFrame().
            using (var raw = new Bitmap(path))
            using (var normalized = _frameNormalizer.Normalize(raw, cameraId))
            {
                var rawSource = Converter.BitmapToBitmapSource(raw);
                var normalizedSource = Converter.BitmapToBitmapSource(normalized);
                // Origin always uses the already rotated canonical frame. Keep
                // RAW only in the capture snapshot/save path, not as an Origin.
                App.ImageViewer.AddImage(normalizedSource, cameraId + "-White-Origin", ThumbStatus.Origin, "LOCAL White - Origin (rotated)");
                if (cameraId == 1)
                {
                    lock (_cam1Lock) { _cam1WhiteRaw = rawSource; _cam1WhiteOrigin = normalizedSource; }
                    _mainWindow.UpdateCam1WhiteOrigin(normalizedSource);
                }
                else
                {
                    lock (_cam2Lock) { _cam2WhiteRaw = rawSource; _cam2WhiteOrigin = normalizedSource; }
                    _mainWindow.UpdateCam2WhiteOrigin(normalizedSource);
                }
                var result = await InspectWhiteFrameAsync(normalized, token).ConfigureAwait(false);
                token.ThrowIfCancellationRequested();
                UpdateUIWithWhiteResults(cameraId == 1 ? result : null, cameraId == 2 ? result : null);
                elapsed.Stop();
                // Per-camera status is WHITE ONLY; aggregate result remains UNKNOWN (UV not tested).
                if (cameraId == 1)
                {
                    _mainWindow.UpdateInspectionStatusCam1((InspectionResult)result.Result);
                    _mainWindow.UpdateCam1ProcessedTime(elapsed.Elapsed);
                }
                else
                {
                    _mainWindow.UpdateInspectionStatusCam2((InspectionResult)result.Result);
                    _mainWindow.UpdateCam2ProcessedTime(elapsed.Elapsed);
                }
                AppLogger.Instance.Info($"LOCAL White CAM{cameraId}: {path}; {result.ErrorDesc}; {elapsed.ElapsedMilliseconds} ms. UV/PLC NOT tested.", "DEBUG");
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
        /// Origin trên main UI/gallery dùng canonical frame đã xoay đúng một lần.
        /// Raw frame chỉ giữ để lưu và truy vết, không thêm làm ảnh Origin.
        /// </summary>
        private void UpdateCapturedFrameUI(
            Bitmap raw1, Bitmap raw2, Bitmap normalized1, Bitmap normalized2, InspectionLight light)
        {
            if (raw1 == null || normalized1 == null)
            {
                ShowAndLogError("Captured frame 1 null", "CAM1");
                return;
            }
            if (raw2 == null || normalized2 == null)
            {
                ShowAndLogError("Captured frame 2 null", "CAM2");
                return;
            }
            if (light == InspectionLight.White)
            {
                lock (_cam1Lock)
                {
                    _cam1WhiteRaw = Converter.BitmapToBitmapSource(raw1);
                    _cam1WhiteOrigin = Converter.BitmapToBitmapSource(normalized1);
                    App.ImageViewer.AddImage(_cam1WhiteOrigin, "1-White-Origin", ThumbStatus.Origin, "Camera 1 - White - Origin (rotated)");
                }
                lock (_cam2Lock)
                {
                    _cam2WhiteRaw = Converter.BitmapToBitmapSource(raw2);
                    _cam2WhiteOrigin = Converter.BitmapToBitmapSource(normalized2);
                    App.ImageViewer.AddImage(_cam2WhiteOrigin, "2-White-Origin", ThumbStatus.Origin, "Camera 2 - White - Origin (rotated)");
                }
                _mainWindow.UpdateCam1WhiteOrigin(_cam1WhiteOrigin);
                _mainWindow.UpdateCam2WhiteOrigin(_cam2WhiteOrigin);
            }
            else
            {
                lock (_cam1Lock)
                {
                    _cam1UvRaw = Converter.BitmapToBitmapSource(raw1);
                    _cam1UvOrigin = Converter.BitmapToBitmapSource(normalized1);
                    App.ImageViewer.AddImage(_cam1UvOrigin, "1-UV-Origin", ThumbStatus.Origin, "Camera 1 - UV - Origin (rotated)");
                }
                lock (_cam2Lock)
                {
                    _cam2UvRaw = Converter.BitmapToBitmapSource(raw2);
                    _cam2UvOrigin = Converter.BitmapToBitmapSource(normalized2);
                    App.ImageViewer.AddImage(_cam2UvOrigin, "2-UV-Origin", ThumbStatus.Origin, "Camera 2 - UV - Origin (rotated)");
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
        private void ClearCycleImages(bool updateUi = true)
        {
            lock (_cam1Lock)
            lock (_cam2Lock)
            {
                _cam1WhiteRaw = _cam1UvRaw = null;
                _cam1WhiteOrigin = _cam1WhiteResult = _cam1UvOrigin = _cam1UvResult = null;
                _cam2WhiteRaw = _cam2UvRaw = null;
                _cam2WhiteOrigin = _cam2WhiteResult = _cam2UvOrigin = _cam2UvResult = null;
            }
            if (!updateUi) return;
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
            var origins = new Dictionary<string, BitmapSource>();
            var results = new Dictionary<string, BitmapSource>();
            var mode = (SaveType)_param.SaveMode;
            lock (_cam1Lock)
            lock (_cam2Lock)
            {
                if (mode != SaveType.RESULT)
                {
                    Add(origins, "Cam1_White_Raw", _cam1WhiteRaw); Add(origins, "Cam2_White_Raw", _cam2WhiteRaw);
                    Add(origins, "Cam1_UV_Raw", _cam1UvRaw); Add(origins, "Cam2_UV_Raw", _cam2UvRaw);
                }
                if (mode != SaveType.ORIGINAL)
                {
                    Add(results, "Cam1_White_Result", _cam1WhiteResult); Add(results, "Cam2_White_Result", _cam2WhiteResult);
                    Add(results, "Cam1_UV_Result", _cam1UvResult); Add(results, "Cam2_UV_Result", _cam2UvResult);
                }
            }
            var originOptions = new ImageSaveOptions(_param.OriginJpegQuality);
            var resultOptions = new ImageSaveOptions(_param.ResultJpegQuality);
            _imageWriter.Enqueue(_param.SavePath, timestamp, origins, originOptions);
            _imageWriter.Enqueue(_param.SavePath, timestamp, results, resultOptions);
            void Add(Dictionary<string, BitmapSource> images, string name, BitmapSource image)
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
            // A local-only session has never connected hardware; do not contact PLC on close.
            if (_hardwareMayBeActive)
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
