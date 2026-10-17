using System;
using System.Collections.Specialized;
using System.Configuration;
using System.Drawing;
using System.Globalization;

namespace DiskInspection.Controllers.Camera
{
    public enum CameraFrameRotation
    {
        None,
        Clockwise90,
        CounterClockwise90,
        Rotate180
    }

    public sealed class CameraFrameNormalizationOptions
    {
        public CameraFrameRotation Rotation { get; }
        public string Roi { get; }

        public CameraFrameNormalizationOptions(CameraFrameRotation rotation, string roi = null)
        {
            Rotation = rotation;
            Roi = roi ?? string.Empty;
        }
    }

    /// <summary>
    /// Converts camera frames to the canonical coordinate system consumed by AI.
    /// ROI is x,y,width,height after rotation; an empty ROI keeps the full frame.
    /// </summary>
    public sealed class CameraFrameNormalizer
    {
        private readonly CameraFrameNormalizationOptions _cam1;
        private readonly CameraFrameNormalizationOptions _cam2;

        public CameraFrameNormalizer(CameraFrameNormalizationOptions cam1, CameraFrameNormalizationOptions cam2)
        {
            _cam1 = cam1 ?? throw new ArgumentNullException(nameof(cam1));
            _cam2 = cam2 ?? throw new ArgumentNullException(nameof(cam2));
        }

        public static CameraFrameNormalizer FromAppSettings()
        {
            NameValueCollection settings = ConfigurationManager.AppSettings;
            return new CameraFrameNormalizer(
                ReadOptions(settings, 1, CameraFrameRotation.Clockwise90),
                ReadOptions(settings, 2, CameraFrameRotation.CounterClockwise90));
        }

        public Bitmap Normalize(Bitmap source, int cameraId)
        {
            if (source == null) throw new ArgumentNullException(nameof(source));
            CameraFrameNormalizationOptions options = GetOptions(cameraId);
            Bitmap rotated = new Bitmap(source);
            try
            {
                rotated.RotateFlip(ToRotateFlipType(options.Rotation));
                Rectangle roi = ParseRoi(options.Roi, rotated.Size, cameraId);
                if (roi.X == 0 && roi.Y == 0 && roi.Width == rotated.Width && roi.Height == rotated.Height)
                    return rotated;

                Bitmap cropped = rotated.Clone(roi, rotated.PixelFormat);
                rotated.Dispose();
                return cropped;
            }
            catch
            {
                rotated.Dispose();
                throw;
            }
        }

        private CameraFrameNormalizationOptions GetOptions(int cameraId)
        {
            if (cameraId == 1) return _cam1;
            if (cameraId == 2) return _cam2;
            throw new ArgumentOutOfRangeException(nameof(cameraId), "Camera id must be 1 or 2.");
        }

        private static CameraFrameNormalizationOptions ReadOptions(
            NameValueCollection settings, int cameraId, CameraFrameRotation defaultRotation)
        {
            string prefix = "Cam" + cameraId.ToString(CultureInfo.InvariantCulture);
            string rotationValue = settings[prefix + "FrameRotation"];
            CameraFrameRotation rotation = defaultRotation;
            if (!string.IsNullOrWhiteSpace(rotationValue) &&
                !Enum.TryParse(rotationValue.Trim(), true, out rotation))
                throw new ConfigurationErrorsException(prefix + "FrameRotation is invalid: " + rotationValue);

            return new CameraFrameNormalizationOptions(rotation, settings[prefix + "NormalizedRoi"]);
        }

        private static RotateFlipType ToRotateFlipType(CameraFrameRotation rotation)
        {
            switch (rotation)
            {
                case CameraFrameRotation.None: return RotateFlipType.RotateNoneFlipNone;
                case CameraFrameRotation.Clockwise90: return RotateFlipType.Rotate90FlipNone;
                case CameraFrameRotation.CounterClockwise90: return RotateFlipType.Rotate270FlipNone;
                case CameraFrameRotation.Rotate180: return RotateFlipType.Rotate180FlipNone;
                default: throw new ArgumentOutOfRangeException(nameof(rotation));
            }
        }

        private static Rectangle ParseRoi(string value, Size imageSize, int cameraId)
        {
            if (string.IsNullOrWhiteSpace(value))
                return new Rectangle(Point.Empty, imageSize);

            string[] parts = value.Split(',');
            int x, y, width, height;
            if (parts.Length != 4 ||
                !int.TryParse(parts[0].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out x) ||
                !int.TryParse(parts[1].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out y) ||
                !int.TryParse(parts[2].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out width) ||
                !int.TryParse(parts[3].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out height))
                throw new ConfigurationErrorsException(
                    "Cam" + cameraId + "NormalizedRoi must be empty or x,y,width,height.");

            var roi = new Rectangle(x, y, width, height);
            var bounds = new Rectangle(Point.Empty, imageSize);
            if (width <= 0 || height <= 0 || !bounds.Contains(roi))
                throw new ConfigurationErrorsException(
                    "Cam" + cameraId + "NormalizedRoi is outside the rotated frame " +
                    imageSize.Width + "x" + imageSize.Height + ": " + value);
            return roi;
        }
    }
}
