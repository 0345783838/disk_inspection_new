using DiskInspection.Models;
using System;
using System.Globalization;
using System.IO;
using System.Windows.Data;
using System.Windows.Media;

namespace DiskInspection.Utils
{
    public class FileNameConverterMain : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is string path)
            {
                var parentFolder = IO.GetParentFolderFromFilePath(path);
                var fileName = IO.GetFileName(path);
                return $"{parentFolder}/{fileName}"; // Trả về tên file từ đường dẫn
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value; // Không cần sử dụng ở đây, vì chỉ hiển thị tên file
        }
    }
    public class MyNullToBoolConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
            => value != null;   // Có ảnh -> true

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class StatusConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == 0) return "DarkGreen";
                else if (status == 1) return "Red";
                else if (status == 2) return "DarkOrange";
                else if (status == 3) return "Black";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class MainStatusConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == (int)(StatusState.Ok)) return "CONNECTED";
                else if (status == (int)(StatusState.Ng)) return "DISCONNECTED";
                else if (status == (int)(StatusState.Stopped)) return "STOPPED";
                return "UNKNOWN";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class MainStatusColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == (int)(StatusState.Ok)) return "DarkGreen";
                else if (status == (int)(StatusState.Ng))  return "#F94449";
                else if (status == (int)(StatusState.Inspecting)) return "#E6B400";
                else if (status == (int)(StatusState.Stopped)) return "#C30010";
                else if (status == (int)(StatusState.Warning)) return "#FF6600";
                return "Gray";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class TextColorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == (int)(StatusState.Ok)) return "White";
                else if (status == (int)(StatusState.Ng)) return "White";
                else if (status == (int)(StatusState.Inspecting)) return "White";
                else if (status == (int)(StatusState.Stopped)) return "White";
                else if (status == (int)(StatusState.Warning)) return "White";
                return "White";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }

    public class MainResCamStatusConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == (int)(StatusState.Ok)) return "OK";
                else if (status == (int)(StatusState.Ng)) return "NG";
                else if (status == (int)(StatusState.Inspecting)) return "...";
                else if (status == (int)(StatusState.Warning)) return "⚠️";
                else if (status == (int)(StatusState.Stopped)) return "X";
                return "N/A";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class MainInspectionStatusConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is int status)
            {
                if (status == (int)(StatusState.Ok)) return "PASSED";
                else if (status == (int)(StatusState.Ng)) return "NOT GOOD";
                else if (status == (int)(StatusState.Inspecting)) return "INSPECTING";
                else if (status == (int)(StatusState.Stopped)) return "STOPPED";
                else if (status == (int)(StatusState.Warning)) return "WARNING";
                return "N/A";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class NullToBoolConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value != null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
    public class StartErrorBackgroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ThumbStatus status)
            {
                if (status == ThumbStatus.Ok) return "#88E788";
                else if (status == ThumbStatus.Origin) return "#787276";
                else if (status == ThumbStatus.Warning) return "#F96209";
                else return "#ee6b6e";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class StopErrorBackgroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ThumbStatus status)
            {
                if (status == ThumbStatus.Ok) return "#cce7c9";
                else if (status == ThumbStatus.Origin) return "#d9dddc";
                else if (status == ThumbStatus.Warning) return "#ed9d0B";
                else return "#ffcbd1";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class ErrorIconConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ThumbStatus status)
            {
                if (status == ThumbStatus.Ok) return "/Resources/Icons/check.png";
                else if (status == ThumbStatus.Origin) return "/Resources/Icons/camera_ic.png";
                else if (status == ThumbStatus.Warning) return "/Resources/Icons/warning.png";
                else return "/Resources/Icons/bad.png";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
    public class ErrorForegroundConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
        {
            if (value is ThumbStatus status)
            {
                if (status == ThumbStatus.Ok) return "#0F5132";
                else if (status == ThumbStatus.Origin) return "#1F2937";
                else if (status == ThumbStatus.Ng) return "#7A0000";
                else if (status == ThumbStatus.Warning) return "Yellow";
                else return "#FF6600";
            }
            return value;
        }

        public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
        {
            return value;
        }
    }
}
