using DiskInspection.Controllers.Camera;
using DiskInspection.Models;
using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

// Opt-in integration test against a running loopback AI service; NO UI/camera/PLC calls.
internal static class LocalWhiteApiSmoke
{
    [STAThread]
    private static int Main(string[] args)
    {
        try { Run(args).GetAwaiter().GetResult(); return 0; }
        catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }

    private static async Task Run(string[] args)
    {
        if (args.Length != 3 && args.Length != 4) throw new ArgumentException("Expected Cam1 raw path, Cam2 raw path, output directory, optional loopback AI URL.");
        var url = args.Length == 4 ? args[3] : "http://127.0.0.1:8386/ai";
        Uri uri;
        if (!Uri.TryCreate(url, UriKind.Absolute, out uri) ||
            (uri.Host != "127.0.0.1" && uri.Host != "localhost") || uri.AbsolutePath.TrimEnd('/') != "/ai")
            throw new ArgumentException("Only a loopback /ai URL is permitted.");
        var normalizer = CameraFrameNormalizer.FromAppSettings();
        var api = typeof(InspectionResponse).Assembly.GetType("DiskInspection.Controllers.APIs.APICommunication", true);
        var inspect = api.GetMethod("InspectWhiteLightAsync", BindingFlags.Public | BindingFlags.Static);
        Directory.CreateDirectory(args[2]);
        for (int cameraId = 1; cameraId <= 2; cameraId++)
        {
            using (var raw = new Bitmap(args[cameraId - 1]))
            using (var normalized = normalizer.Normalize(raw, cameraId))
            using (var image = new Image<Bgr, byte>(normalized))
            {
                // Calls the actual APP image encoder, RestSharp transport and response deserializer.
                var pending = (Task<InspectionResponse>)inspect.Invoke(null, new object[] {
                    url, image.Mat, CancellationToken.None, 10000 });
                var result = await pending;
                if (result == null || result.Result < 0 || result.Result > 2 || string.IsNullOrWhiteSpace(result.ResImg))
                    throw new InvalidOperationException("Invalid White response.");
                if (!result.HasUvGeometry && result.Result != 1)
                    throw new InvalidOperationException("Missing White geometry.");
                File.WriteAllBytes(Path.Combine(args[2], "Cam" + cameraId + "_APP_White_result.png"),
                    Convert.FromBase64String(result.ResImg));
                Console.WriteLine($"PASS API transport CAM{cameraId}: {raw.Width}x{raw.Height} -> {normalized.Width}x{normalized.Height}, result={result.Result}, code={result.ErrorCode}, geometry={result.HasUvGeometry}; {result.ErrorDesc}");
            }
        }
    }
}
