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
namespace DiskInspection.Utils { } // ImageSaver's existing namespace import.

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
        await Check("lighting order and configured delays", Normal);
        await Check("capture failure cleans up and skips UV", () => Failure(capture: true));
        await Check("AI failure joins lighting work and skips UV", () => Failure(capture: false));
        await Check("failed light-off never advances to UV", () => LightFailure(false));
        await Check("failed light-on also attempts cleanup", () => LightFailure(true));
        await Check("cancel during light wait switches light off", CancelDuringWait);
        await Check("cancel waits for in-flight work", CancelDuringInference);
        await Check("queued images stay with their cycle", ImageSnapshots);
        await Check("image extension matches encoded bytes", ImageFormat);
        await Check("missing array settings use safe defaults", ConfigArrayDefaults);
    }
    private static async Task Check(string name, Func<Task> test)
    {
        await test(); _passed++; Console.WriteLine("PASS " + name);
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
        var a = Pixel(10); var b = Pixel(20);
        var writer = new InspectionImageWriter((bitmap, path, time, name) =>
        {
            entered.Set();
            if (!release.Wait(5000)) throw new Exception("Save gate timed out.");
            saved.Enqueue(bitmap);
            return name != "first"; // An earlier save failure must not suppress later writes.
        });
        var images = new Dictionary<string, BitmapSource> { ["first"] = a, ["second"] = a };
        writer.Enqueue("unused", DateTime.Now, images);
        Assert(entered.Wait(5000), "Writer did not start.");
        images["first"] = b; images["second"] = b;
        release.Set(); await writer.FlushAsync();
        Assert(saved.Count == 2 && saved.All(item => ReferenceEquals(item, a)), "Images were skipped or mixed across cycles.");
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
}
