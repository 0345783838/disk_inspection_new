using System;
using DiskInspection.Models;
using System.IO;
using System.Windows.Media.Imaging;

namespace DiskInspection.Controllers
{
    public static class ImageSaver
    {
        public static bool SaveImage(
            BitmapSource bitmap,
            string savePath,
            DateTime saveTime,
            string title)
        {
            return SaveImage(bitmap, savePath, saveTime, title, ImageSaveOptions.Default);
        }

        public static bool SaveImage(
            BitmapSource bitmap,
            string savePath,
            DateTime saveTime,
            string title,
            ImageSaveOptions options)
        {
            if (bitmap == null || string.IsNullOrEmpty(savePath))
                return false;

            try
            {
                options = options ?? ImageSaveOptions.Default;
                var dateTime = saveTime.ToString("dd_MM_yyyy");
                var stringHour = saveTime.ToString("HH_mm_ss_fffffff");
                string saveDir = Path.Combine(savePath, dateTime);
                saveDir = Path.Combine(saveDir, stringHour);
                string extension = options.IsLossless ? ".png" : ".jpg";
                string imageFile = $"{saveTime.ToString("ddMMyyyy_HHmmss_fffffff")}_{title}{extension}";
                if (!string.IsNullOrEmpty(saveDir) && !Directory.Exists(saveDir))
                {
                    Directory.CreateDirectory(saveDir);
                }

                var filePath = Path.Combine(saveDir, imageFile);
                using (FileStream fs = new FileStream(filePath, FileMode.Create))
                {
                    BitmapEncoder encoder = options.IsLossless
                        ? (BitmapEncoder)new PngBitmapEncoder()
                        : new JpegBitmapEncoder { QualityLevel = options.JpegQuality };

                    // Encode at the original pixel dimensions for every quality setting.
                    encoder.Frames.Add(BitmapFrame.Create(bitmap));
                    encoder.Save(fs);
                }
                return true;
            }
            catch (Exception ex)
            {
                // log nếu cần
                Console.WriteLine(ex.ToString());
            }
            return false;
        }
    }
}
