using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace DiskInspection.Resources.Styles
{
    /// <summary>
    /// Interaction logic for CustomButton2.xaml
    /// </summary>
    public partial class CustomButton2 : UserControl
    {
        public CustomButton2()
        {
            InitializeComponent();
        }
        public string Title
        {
            get => (string)GetValue(TitleProperty);
            set => SetValue(TitleProperty, value);
        }
        public static readonly DependencyProperty TitleProperty =
            DependencyProperty.Register(nameof(Title), typeof(string), typeof(CustomButton2), new PropertyMetadata("Title"));
        public string MyFontSize
        {
            get => (string)GetValue(MyFontSizeProperty);
            set => SetValue(FontSizeProperty, value);
        }
        public static readonly DependencyProperty MyFontSizeProperty =
            DependencyProperty.Register(nameof(MyFontSize), typeof(string), typeof(CustomButton2), new PropertyMetadata("14"));

        public ImageSource LeftIcon
        {
            get => (ImageSource)GetValue(LeftIconProperty);
            set => SetValue(LeftIconProperty, value);
        }
        public static readonly DependencyProperty LeftIconProperty =
            DependencyProperty.Register(nameof(LeftIcon), typeof(ImageSource), typeof(CustomButton2), new PropertyMetadata(null));

        //public ImageSource RightIcon
        //{
        //    get => (ImageSource)GetValue(RightIconProperty);
        //    set => SetValue(RightIconProperty, value);
        //}
        //public static readonly DependencyProperty RightIconProperty =
        //    DependencyProperty.Register(nameof(RightIcon), typeof(ImageSource), typeof(CustomButton), new PropertyMetadata(null));

        public Color StartColor
        {
            get => (Color)GetValue(StartColorProperty);
            set => SetValue(StartColorProperty, value);
        }
        public static readonly DependencyProperty StartColorProperty =
            DependencyProperty.Register(nameof(StartColor), typeof(Color), typeof(CustomButton2),
                new PropertyMetadata(Colors.Blue, OnGradientColorChanged));

        public Color EndColor
        {
            get => (Color)GetValue(EndColorProperty);
            set => SetValue(EndColorProperty, value);

        }
        public static readonly DependencyProperty EndColorProperty =
            DependencyProperty.Register(nameof(EndColor), typeof(Color), typeof(CustomButton2),
                new PropertyMetadata(Colors.LightBlue, OnGradientColorChanged));

        public Brush BackgroundBrush
        {
            get => (Brush)GetValue(BackgroundBrushProperty);
            set => SetValue(BackgroundBrushProperty, value);
        }
        public static readonly DependencyProperty BackgroundBrushProperty =
            DependencyProperty.Register(nameof(BackgroundBrush), typeof(Brush), typeof(CustomButton2),
                new PropertyMetadata(Brushes.White));


        public int IconWidth
        {
            get => (int)GetValue(IconWidthProperty);
            set => SetValue(IconWidthProperty, value);
        }
        public static readonly DependencyProperty IconWidthProperty =
            DependencyProperty.Register(nameof(IconWidth), typeof(int), typeof(CustomButton2),
                new PropertyMetadata(0));

        public int IconHeight
        {
            get => (int)GetValue(IconHeightProperty);
            set => SetValue(IconHeightProperty, value);
        }
        public static readonly DependencyProperty IconHeightProperty =
            DependencyProperty.Register(nameof(IconHeight), typeof(int), typeof(CustomButton2),
                new PropertyMetadata(0));


        private static void OnGradientColorChanged(DependencyObject d, DependencyPropertyChangedEventArgs e)
        {
            if (d is CustomButton2 button)
            {
                button.UpdateBackgroundBrush();
            }
        }
        private void UpdateBackgroundBrush()
        {
            var gradient = new LinearGradientBrush
            {
                StartPoint = new Point(0, 0),
                EndPoint = new Point(1, 1),
                GradientStops = new GradientStopCollection
        {
            new GradientStop(StartColor, 0.0),
            new GradientStop(EndColor, 1.0)
        }
            };
            BackgroundBrush = gradient;
        }
    }
}
