using DiskInspection.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
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
            if (bitmap == null || string.IsNullOrEmpty(savePath))
                return false;

            // Clone + Freeze để dùng được ở thread khác
            BitmapSource cloned = bitmap.Clone();
            cloned.Freeze();

            try
            {
                var dateTime = saveTime.ToString("dd_MM_yyyy");
                var stringHour = saveTime.ToString("HH_mm_ss_fffffff");
                string saveDir = Path.Combine(savePath, dateTime);
                saveDir = Path.Combine(saveDir, stringHour);
                string imageFile = $"{saveTime.ToString("ddMMyyyy_HHmmss_fffffff")}_{title}.jpg";
                if (!string.IsNullOrEmpty(saveDir) && !Directory.Exists(saveDir))
                {
                    Directory.CreateDirectory(saveDir);
                }

                var filePath = Path.Combine(saveDir, imageFile);
                using (FileStream fs = new FileStream(filePath, FileMode.Create))
                {
                    //BitmapEncoder encoder = new PngBitmapEncoder();
                    // Nếu cần JPEG:
                    BitmapEncoder encoder = new JpegBitmapEncoder { QualityLevel = 90 };

                    encoder.Frames.Add(BitmapFrame.Create(cloned));
                    encoder.Save(fs);
                }
                return true;
            }
            catch (Exception ex)
            {
                // log nếu cần
                Console.WriteLine(ex.ToString());
            }
            finally
            {
                cloned = null;
            }
            return false;
        }
    }
}
