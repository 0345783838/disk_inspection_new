using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace DiskInspection.Models
{
    public class ThumbItem
    {
        public BitmapSource Image { get; set; }
        public string Text { get; set; }
        public ThumbStatus StatusColor { get; set; }
        public string ErrorDescription { get; set; }

        public ThumbItem(BitmapSource image, string text, ThumbStatus statusColor, string errorDescription)
        {
            Image = image;
            Text = text;
            StatusColor = statusColor;
            //if (statusColor == ThumbStatus.Ok)
            //    StatusColor = Brushes.DarkGreen;
            //else if (statusColor == ThumbStatus.Ng)
            //    StatusColor = new SolidColorBrush(System.Windows.Media.Color.FromRgb(249, 68, 73));
            //else
            //    StatusColor = Brushes.DarkGray;
            ErrorDescription = errorDescription;
        }
    }
}
