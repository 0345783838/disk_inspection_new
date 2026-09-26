using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Interop;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Runtime.InteropServices;

namespace DiskInspection.Utils
{
    public class Converter
    {

        public static Bitmap Base64ToBitmap(string base64)
        {
            // Nếu base64 có prefix kiểu: data:image/png;base64,...
            if (base64.Contains(","))
                base64 = base64.Substring(base64.IndexOf(",") + 1);

            byte[] imageBytes = Convert.FromBase64String(base64);

            using (var ms = new MemoryStream(imageBytes))
            {
                using (var source = new Bitmap(ms))
                    return new Bitmap(source);
            }
        }
        public static BitmapSource Base64ToBitmapSource(string base64)
        {
            try
            {
                if (base64.Contains(","))
                    base64 = base64.Substring(base64.IndexOf(",") + 1);

                byte[] imageBytes = Convert.FromBase64String(base64);

                using (var ms = new MemoryStream(imageBytes))
                {
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad; // rất quan trọng
                    bitmap.StreamSource = ms;
                    bitmap.EndInit();
                    bitmap.Freeze(); // cho phép dùng khác thread

                    return bitmap;
                }
            }
            catch
            {
                return null;
            }
        }
        public static BitmapSource BitmapToBitmapSource(Bitmap bitmap)
        {
            if (bitmap == null) return null;

            IntPtr hBitmap = bitmap.GetHbitmap();

            try
            {
                var bitmapSource = Imaging.CreateBitmapSourceFromHBitmap(
                    hBitmap,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());

                bitmapSource.Freeze();

                return bitmapSource;
            }
            finally
            {
                DeleteObject(hBitmap);
            }
        }
        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);
    }
}
