using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using DiskInspection.Controllers;
using DiskInspection.Controllers.Camera;
using DiskInspection.Controllers.PLC;
using DiskInspection.Models;
using DiskInspection.Utils;

namespace DiskInspection.Controllers
{
    internal sealed class AppLogger
    {
        public static AppLogger Instance { get; } = new AppLogger();
        public void Error(string message, string source) { }
    }
}

internal static class FlowTests
{
    private static int _passed;
    [STAThread]
    private static int Main()
    {
        try { Run().GetAwaiter().GetResult(); Console.WriteLine($"PASS: {_passed} flow tests"); return 0; }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
    private static async Task Run()
    {
        await Check("trigger reset continues immediately on first success", () => TriggerReset(1));
        await Check("trigger reset retries once then continues", () => TriggerReset(2));
        await Check("trigger reset accepts success on third attempt", () => TriggerReset(3));
        await Check("trigger reset stops after three failures", () => TriggerReset(0));
        await Check("trigger reset cancellation prevents more commands", TriggerResetCancellation);
        await Check("lighting order and configured delays", Normal);
        await Check("capture failure cleans up and skips UV", () => Failure(capture: true));
        await Check("AI failure joins lighting work and skips UV", () => Failure(capture: false));
        await Check("failed light-off never advances to UV", () => LightFailure(false));
        await Check("failed light-on also attempts cleanup", () => LightFailure(true));
        await Check("cancel during light wait switches light off", CancelDuringWait);
        await Check("cancel waits for in-flight work", CancelDuringInference);
        await Check("queued images stay with their cycle", ImageSnapshots);
        await Check("image extension matches encoded bytes", ImageFormat);
        await Check("lossless origin and compressed result keep full dimensions and separate quality", ImageSaveProfiles);
        await Check("all quality settings preserve odd and single-pixel dimensions", ImageFullResolution);
        await Check("missing array settings use safe defaults", ConfigArrayDefaults);
        await Check("camera normalization preserves orientation, ROI and White/UV mapping", CameraNormalization);
    }
    private static async Task Check(string name, Func<Task> test)
    {
        await test(); _passed++; Console.WriteLine("PASS " + name);
    }
    private static async Task TriggerReset(int successAttempt)
    {
        var events = new List<string>();
        int attempts = 0;
        var reset = PlcTriggerReset.RunAsync(() =>
        {
            events.Add("reset:" + ++attempts);
            return attempts == successAttempt;
        }, CancellationToken.None, (ms, token) =>
        {
            events.Add("delay:" + ms);
            return Task.CompletedTask;
        });
        if (successAttempt == 0)
        {
            bool failed = false;
            try { await reset; }
            catch (InvalidOperationException ex)
            {
                failed = ex.Message == "Cannot reset PLC trigger after 3 attempts.";
            }
            Assert(failed, "Reset failure must block the inspection cycle.");
        }
        else
        {
            await reset;
            events.Add("continue");
        }
        var expected = new List<string>();
        int expectedAttempts = successAttempt == 0 ? 3 : successAttempt;
        for (int i = 1; i <= expectedAttempts; i++)
        {
            expected.Add("reset:" + i);
            if (i < expectedAttempts) expected.Add("delay:100");
        }
        if (successAttempt != 0) expected.Add("continue");
        Assert(events.SequenceEqual(expected), "Unexpected reset commands, delays or continuation.");
    }
    private static async Task TriggerResetCancellation()
    {
        using (var cts = new CancellationTokenSource())
        {
            int attempts = 0;
            var reset = PlcTriggerReset.RunAsync(() => { attempts++; return false; }, cts.Token,
                (ms, token) => { cts.Cancel(); return Task.Delay(ms, token); });
            bool cancelled = false;
            try { await reset; }
            catch (OperationCanceledException) { cancelled = true; }
            Assert(cancelled && attempts == 1, "Stop during retry delay sent another reset.");

            await Fails(PlcTriggerReset.RunAsync(() => { attempts++; return true; }, cts.Token));
            Assert(attempts == 1, "An already cancelled cycle sent a reset.");
        }
    }
    private static void Assert(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }
    private static async Task Fails(Task task)
    {
        try { await task; }
        catch (InvalidOperationException) { return; }
        catch (OperationCanceledException) { return; }
        throw new Exception("Expected the cycle to fail or cancel.");
    }
    private static Task Done(CancellationToken token) => Task.CompletedTask;
    private static Task Cycle(InspectionSequence sequence, CancellationToken token,
        Func<CancellationToken, Task> capture = null, Func<CancellationToken, Task> inspect = null) =>
        sequence.RunAsync(capture ?? Done, inspect ?? Done, Done, Done, 2000, 5000, 2000, 5000, token);
    private static InspectionSequence Record(List<string> events, Func<InspectionLight, bool, bool> result = null) =>
        new InspectionSequence((light, on) =>
        {
            events.Add(light + (on ? "+" : "-"));
            return Task.FromResult(result?.Invoke(light, on) ?? true);
        }, (ms, token) => Task.CompletedTask);
    private static async Task Normal()
    {
        var events = new List<string>();
        var sequence = new InspectionSequence((light, on) =>
        {
            events.Add(light + (on ? "+" : "-")); return Task.FromResult(true);
        }, (ms, token) => { events.Add("delay:" + ms); return Task.CompletedTask; });
        await sequence.RunAsync(t => { events.Add("capture-white"); return Task.CompletedTask; },
            t => { events.Add("inspect-white"); return Task.CompletedTask; },
            t => { events.Add("capture-uv"); return Task.CompletedTask; },
            t => { events.Add("inspect-uv"); return Task.CompletedTask; }, 2000, 5000, 2000, 5000, CancellationToken.None);
        Assert(string.Join(",", events) == "White+,delay:2000,capture-white,inspect-white,delay:5000,White-,Uv+,delay:2000,capture-uv,inspect-uv,delay:5000,Uv-", "Unexpected sequence or timing.");
    }
    private static async Task Failure(bool capture)
    {
        var events = new List<string>();
        Func<CancellationToken, Task> fail = token => throw new InvalidOperationException("Frame/API failure");
        await Fails(Cycle(Record(events), CancellationToken.None, capture ? fail : null, capture ? null : fail));
        Assert(events.SequenceEqual(new[] { "White+", "White-" }), "Failure left lighting work behind or advanced to UV.");
    }
    private static async Task LightFailure(bool onFailure)
    {
        var events = new List<string>();
        await Fails(Cycle(Record(events, (light, on) => onFailure ? !on : on), CancellationToken.None));
        Assert(events.SequenceEqual(onFailure ? new[] { "White+", "White-" } : new[] { "White+", "White-", "White-" }), "Invalid cleanup after failed light command.");
    }
    private static async Task CancelDuringWait()
    {
        using (var cts = new CancellationTokenSource())
        {
            var events = new List<string>();
            var sequence = new InspectionSequence((light, on) =>
            {
                events.Add(light + (on ? "+" : "-")); return Task.FromResult(true);
            }, (ms, token) => { cts.Cancel(); return Task.FromCanceled(token); });
            await Fails(Cycle(sequence, cts.Token));
            Assert(events.SequenceEqual(new[] { "White+", "White-" }), "Cancel skipped cleanup.");
        }
    }
    private static async Task CancelDuringInference()
    {
        using (var cts = new CancellationTokenSource())
        {
            var pending = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            var events = new ConcurrentQueue<string>();
            int delayCount = 0;
            var sequence = new InspectionSequence((light, on) =>
            {
                events.Enqueue(light + (on ? "+" : "-")); return Task.FromResult(true);
            }, (ms, token) => ++delayCount == 1 ? Task.CompletedTask : Task.Delay(Timeout.Infinite, token));
            var cycle = Cycle(sequence, cts.Token, inspect: token => pending.Task);
            cts.Cancel();
            Assert(!cycle.IsCompleted, "Cycle completed while inference still owned its frames.");
            pending.SetResult(true);
            await Fails(cycle);
            Assert(events.SequenceEqual(new[] { "White+", "White-" }), "Cancelled inference advanced to UV or skipped cleanup.");
        }
    }
    private static BitmapSource Pixel(byte red)
    {
        var bitmap = BitmapSource.Create(1, 1, 96, 96, PixelFormats.Rgb24, null, new[] { red, (byte)0, (byte)0 }, 3);
        bitmap.Freeze(); return bitmap;
    }
    private static async Task ImageSnapshots()
    {
        var release = new ManualResetEventSlim(); var entered = new ManualResetEventSlim();
        var saved = new ConcurrentQueue<BitmapSource>();
        var savedOptions = new ConcurrentQueue<ImageSaveOptions>();
        var a = Pixel(10); var b = Pixel(20);
        var writer = new InspectionImageWriter((bitmap, path, time, name, options) =>
        {
            entered.Set();
            if (!release.Wait(5000)) throw new Exception("Save gate timed out.");
            saved.Enqueue(bitmap);
            savedOptions.Enqueue(options);
            return name != "first"; // An earlier save failure must not suppress later writes.
        });
        var images = new Dictionary<string, BitmapSource> { ["first"] = a, ["second"] = a };
        var originOptions = new ImageSaveOptions(0);
        var resultOptions = new ImageSaveOptions(50);
        writer.Enqueue("unused", DateTime.Now, images, originOptions);
        Assert(entered.Wait(5000), "Writer did not start.");
        images["first"] = b; images["second"] = b;
        writer.Enqueue("unused", DateTime.Now, new Dictionary<string, BitmapSource> { ["result"] = b }, resultOptions);
        release.Set(); await writer.FlushAsync();
        Assert(saved.SequenceEqual(new[] { a, a, b }), "Images were skipped or mixed across cycles.");
        Assert(savedOptions.SequenceEqual(new[] { originOptions, originOptions, resultOptions }),
            "Queued images used another batch's save settings.");
    }
    private static Task ImageFormat()
    {
        var folder = Path.Combine(Path.GetTempPath(), "DiskInspectionFlowTests", Guid.NewGuid().ToString("N"));
        Assert(ImageSaver.SaveImage(Pixel(255), folder, DateTime.Now, "frame"), "Could not save image.");
        var path = Directory.GetFiles(folder, "*", SearchOption.AllDirectories).Single();
        var bytes = File.ReadAllBytes(path);
        Assert(Path.GetExtension(path) == ".jpg" && bytes[0] == 0xff && bytes[1] == 0xd8, "Extension and encoded format disagree.");
        return Task.CompletedTask;
    }
    private static BitmapSource Pattern(int width, int height)
    {
        var bytes = new byte[width * height * 3];
        new Random(42).NextBytes(bytes);
        var image = BitmapSource.Create(width, height, 96, 96, PixelFormats.Rgb24, null, bytes, width * 3);
        image.Freeze();
        return image;
    }
    private static BitmapSource ReadImage(string path)
    {
        using (var stream = File.OpenRead(path))
        {
            var image = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad).Frames[0];
            image.Freeze();
            return image;
        }
    }
    private static byte[] RgbPixels(BitmapSource image)
    {
        var rgb = new FormatConvertedBitmap(image, PixelFormats.Rgb24, null, 0);
        var pixels = new byte[rgb.PixelWidth * rgb.PixelHeight * 3];
        rgb.CopyPixels(pixels, rgb.PixelWidth * 3, 0);
        return pixels;
    }
    private static async Task ImageSaveProfiles()
    {
        var folder = Path.Combine(Path.GetTempPath(), "DiskInspectionFlowTests", Guid.NewGuid().ToString("N"));
        try
        {
            var source = Pattern(128, 96);
            var before = RgbPixels(source);
            var timestamp = new DateTime(2026, 10, 31, 12, 30, 0);
            var writer = new InspectionImageWriter();
            writer.Enqueue(folder, timestamp, new Dictionary<string, BitmapSource>
            {
                ["Cam1_White_Raw"] = source, ["Cam2_White_Raw"] = source
            }, new ImageSaveOptions(0));
            writer.Enqueue(folder, timestamp, new Dictionary<string, BitmapSource>
            {
                ["Cam1_White_Result"] = source
            }, new ImageSaveOptions(75));
            await writer.FlushAsync();
            var paths = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);
            Assert(paths.Length == 3, "Origin/result files were lost or overwritten.");
            foreach (var path in paths.Where(p => p.EndsWith(".png")))
            {
                var bytes = File.ReadAllBytes(path);
                Assert(bytes.Take(8).SequenceEqual(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }), "PNG signature is invalid.");
                var decoded = ReadImage(path);
                Assert(decoded.PixelWidth == 128 && decoded.PixelHeight == 96 && RgbPixels(decoded).SequenceEqual(before),
                    "100% PNG changed source dimensions or pixels.");
            }
            Assert(paths.Count(p => p.EndsWith(".png")) == 2, "Raw images from both cameras must use the origin profile.");
            var result = paths.Single(p => p.EndsWith("_Result.jpg"));
            var decodedResult = ReadImage(result);
            Assert(decodedResult.PixelWidth == 128 && decodedResult.PixelHeight == 96, "JPEG result changed the original dimensions.");
            Assert(source.PixelWidth == 128 && source.PixelHeight == 96 && RgbPixels(source).SequenceEqual(before),
                "Saving changed the source used by inspection or UI.");

            Assert(ImageSaver.SaveImage(source, folder, timestamp, "jpeg_high", new ImageSaveOptions(95)), "JPEG95 failed.");
            Assert(ImageSaver.SaveImage(source, folder, timestamp, "jpeg_low", new ImageSaveOptions(50)), "JPEG50 failed.");
            paths = Directory.GetFiles(folder, "*.jpg", SearchOption.AllDirectories);
            Assert(new FileInfo(paths.Single(p => p.EndsWith("jpeg_high.jpg"))).Length >
                new FileInfo(paths.Single(p => p.EndsWith("jpeg_low.jpg"))).Length,
                "Selected JPEG quality did not affect actual compression.");
        }
        finally { if (Directory.Exists(folder)) Directory.Delete(folder, true); }
    }
    private static Task ImageFullResolution()
    {
        var folder = Path.Combine(Path.GetTempPath(), "DiskInspectionFlowTests", Guid.NewGuid().ToString("N"));
        try
        {
            foreach (int quality in new[] { 0, 100, 95, 90, 75, 50 })
            foreach (var source in new[] { Pattern(123, 79), Pixel(42) })
            {
                var title = $"{source.PixelWidth}_{quality}";
                Assert(ImageSaver.SaveImage(source, folder, DateTime.Now, title, new ImageSaveOptions(quality)), "Image save failed.");
                var extension = quality == 0 ? ".png" : ".jpg";
                var decoded = ReadImage(Directory.GetFiles(folder, "*_" + title + extension, SearchOption.AllDirectories).Single());
                Assert(decoded.PixelWidth == source.PixelWidth && decoded.PixelHeight == source.PixelHeight,
                    "Compression setting changed the original image dimensions.");
            }
            Assert(new ImageSaveOptions(-1).JpegQuality == 90, "Invalid persisted quality needs a safe default.");
        }
        finally { if (Directory.Exists(folder)) Directory.Delete(folder, true); }
        return Task.CompletedTask;
    }
    private static Task ConfigArrayDefaults()
    {
        var folder = Path.Combine(Path.GetTempPath(), "DiskInspectionFlowTests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        var path = Path.Combine(folder, "config.env");
        File.WriteAllText(path, "CALIPER_THICKNESS_LIST =\r\n");
        var reader = new EnvReader(path);
        Assert(reader.GetIntArray("CALIPER_THICKNESS_LIST", new[] { 3, 5, 7 }).SequenceEqual(new[] { 3, 5, 7 }),
            "Missing caliper thickness did not use defaults.");

        var config = new EnvironmentConfig(0.1f, 0.1f, 0.5f, 0.5f, 4, 25, 0.95f,
            new List<int>(), 25, 9999, 40, 250, new List<int>(), new List<int>(), 20);
        Assert(config.CaliperThicknessList.SequenceEqual(new[] { 3, 5, 7 }),
            "Environment config retained an empty caliper thickness list.");
        Assert(config.UvLowerThreshold.Count == 3 && config.UvUpperThreshold.Count == 3,
            "Environment config retained invalid UV threshold arrays.");
        return Task.CompletedTask;
    }

    private static Task CameraNormalization()
    {
        using (var source = new System.Drawing.Bitmap(3, 2))
        {
            source.SetPixel(0, 0, System.Drawing.Color.Red);
            source.SetPixel(2, 0, System.Drawing.Color.Green);
            source.SetPixel(0, 1, System.Drawing.Color.Blue);
            source.SetPixel(2, 1, System.Drawing.Color.Yellow);

            var normalizer = new CameraFrameNormalizer(
                new CameraFrameNormalizationOptions(CameraFrameRotation.Clockwise90),
                new CameraFrameNormalizationOptions(CameraFrameRotation.CounterClockwise90));
            using (var cam1White = normalizer.Normalize(source, 1))
            using (var cam1Uv = normalizer.Normalize(source, 1))
            using (var cam2 = normalizer.Normalize(source, 2))
            using (var cam2Uv = normalizer.Normalize(source, 2))
            {
                Assert(cam1White.Size == new System.Drawing.Size(2, 3), "Cam1 rotation has the wrong shape.");
                Assert(cam2.Size == new System.Drawing.Size(2, 3), "Cam2 rotation has the wrong shape.");
                Assert(cam1White.GetPixel(1, 0).ToArgb() == System.Drawing.Color.Red.ToArgb(),
                    "Cam1 did not rotate clockwise.");
                Assert(cam2.GetPixel(0, 2).ToArgb() == System.Drawing.Color.Red.ToArgb(),
                    "Cam2 did not rotate counter-clockwise.");
                Assert(cam1White.Size == cam1Uv.Size &&
                    cam1White.GetPixel(0, 0).ToArgb() == cam1Uv.GetPixel(0, 0).ToArgb(),
                    "White and UV do not share the same camera transform.");
                Assert(cam2.Size == cam2Uv.Size &&
                    cam2.GetPixel(0, 2).ToArgb() == cam2Uv.GetPixel(0, 2).ToArgb(),
                    "Cam2 White/UV Origin must share the same rotated frame.");
                Assert(source.Size == new System.Drawing.Size(3, 2) &&
                    source.GetPixel(0, 0).ToArgb() == System.Drawing.Color.Red.ToArgb(),
                    "Origin rotation mutated the original RAW capture.");
            }

            var croppedNormalizer = new CameraFrameNormalizer(
                new CameraFrameNormalizationOptions(CameraFrameRotation.Clockwise90, "0,1,2,2"),
                new CameraFrameNormalizationOptions(CameraFrameRotation.None));
            using (var full = normalizer.Normalize(source, 1))
            using (var cropped = croppedNormalizer.Normalize(source, 1))
            {
                Assert(cropped.Size == new System.Drawing.Size(2, 2), "Configured ROI was not applied.");
                Assert(cropped.GetPixel(0, 0).ToArgb() == full.GetPixel(0, 1).ToArgb(),
                    "ROI coordinates were not evaluated after rotation.");
            }
        }
        return Task.CompletedTask;
    }
}
