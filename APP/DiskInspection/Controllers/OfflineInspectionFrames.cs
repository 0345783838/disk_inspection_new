using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;

namespace DiskInspection.Controllers
{
    internal sealed class OfflineInspectionFrames
    {
        private static readonly string[] Suffixes =
        {
            "_Cam1_White_Raw", "_Cam2_White_Raw", "_Cam1_UV_Raw", "_Cam2_UV_Raw"
        };

        private static readonly HashSet<string> ImageExtensions = new HashSet<string>(
            new[] { ".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff" },
            StringComparer.OrdinalIgnoreCase);

        private readonly string[] _paths;

        private OfflineInspectionFrames(string captureName, string[] paths)
        {
            CaptureName = captureName;
            _paths = paths;
        }

        internal string CaptureName { get; }

        internal static OfflineInspectionFrames FromFolder(string folder)
        {
            var captures = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
            foreach (string path in Directory.EnumerateFiles(folder))
            {
                if (!ImageExtensions.Contains(Path.GetExtension(path))) continue;
                string name = Path.GetFileNameWithoutExtension(path);
                int slot = Array.FindIndex(Suffixes, suffix => name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase));
                if (slot < 0) continue;

                string prefix = name.Substring(0, name.Length - Suffixes[slot].Length);
                if (!captures.TryGetValue(prefix, out var paths))
                {
                    paths = new string[Suffixes.Length];
                    captures.Add(prefix, paths);
                }
                if (paths[slot] != null)
                    throw new InvalidDataException($"Có nhiều ảnh RAW cho {prefix + Suffixes[slot]}.");
                paths[slot] = path;
            }

            if (captures.Count == 0)
                throw new InvalidDataException("Thư mục không có ảnh RAW theo mẫu *_Cam1_White_Raw.png hoặc *_Cam2_UV_Raw.png.");
            if (captures.Count != 1)
                throw new InvalidDataException("Thư mục có ảnh RAW của nhiều lượt. Hãy chọn thư mục của một lượt kiểm tra.");

            var capture = captures.Single();
            var missing = Suffixes.Where((suffix, slot) => capture.Value[slot] == null).ToArray();
            if (missing.Length > 0)
                throw new InvalidDataException("Thiếu ảnh RAW: " + string.Join(", ", missing.Select(suffix => capture.Key + suffix)) + ".");

            return new OfflineInspectionFrames(capture.Key, capture.Value);
        }

        internal (Bitmap cam1, Bitmap cam2) Load(InspectionLight light, CancellationToken token)
        {
            int first = light == InspectionLight.White ? 0 : 2;
            token.ThrowIfCancellationRequested();
            Bitmap cam1 = Read(_paths[first]);
            try
            {
                token.ThrowIfCancellationRequested();
                return (cam1, Read(_paths[first + 1]));
            }
            catch
            {
                cam1.Dispose();
                throw;
            }
        }

        private static Bitmap Read(string path)
        {
            try { return new Bitmap(path); }
            catch (ArgumentException ex)
            {
                throw new InvalidDataException($"Không đọc được ảnh RAW {Path.GetFileName(path)}.", ex);
            }
        }
    }
}
