using DiskInspection.Models;
using Emgu.CV.Plot;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;

namespace DiskInspection.Views
{
    public partial class ImageViewerWindow : Window, INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([System.Runtime.CompilerServices.CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        private Point _start;
        private Point _origin;

        private const double EDGE_ZONE = 150;

        public ObservableCollection<ThumbItem> ImageList { get; set; } = new ObservableCollection<ThumbItem>();

        private ThumbItem _selectedThumb;
        public ThumbItem SelectedThumb
        {
            get { return _selectedThumb; }
            set
            {
                if (_selectedThumb != value)
                {
                    _selectedThumb = value;
                    OnPropertyChanged();
                    ThumbSelectionChanged();
                }
            }
        }

        private void ThumbSelectionChanged()
        {
            if (SelectedThumb == null)
            {
                imbImage.Source = null;

                HideButton(btnBack);
                HideButton(btnNext);
            }

            else
            {
                imbImage.Source = SelectedThumb.Image;
                ResetView();

                Dispatcher.BeginInvoke(new Action(() =>
                {
                    lbThumbList.Focus();
                    lbThumbList.SelectedItem = SelectedThumb;
                    lbThumbList.ScrollIntoView(lbThumbList.SelectedItem);
                }), System.Windows.Threading.DispatcherPriority.Background);

            }
        }

        public ImageViewerWindow()
        {
            InitializeComponent();
            DataContext = this;
        }
        private void InvokeUI(Action action)
        {
            if (Dispatcher.CheckAccess())
                action();
            else
                Dispatcher.Invoke(action);
        }

        // ================= PUBLIC API =================

        public void ClearImages()
        {
            InvokeUI(() =>
            {
                ImageList.Clear();
                SelectedThumb = null;
            });
        }


        public void ShowViewer()
        {
            if (!IsVisible)
                Show();

            Activate();
            WindowState = WindowState.Normal;
        }
        public void HideViewer()
        {
            InvokeUI(() => { if (IsVisible) Hide(); });
        }
        public void ShowByImage(BitmapSource img)
        {
            if (img == null) return;

            InvokeUI(() =>
            {
                int index = ImageList
                    .Select((item, i) => new { item, i })
                    .FirstOrDefault(x => x.item.Image == img)?.i ?? -1;

                if (index >= 0)
                {
                    SelectedThumb = ImageList[index];
                    ShowViewer();
                }
            });
        }
        public void ShowFirstErrorImage()
        {
            InvokeUI(() =>
            {
                var firstError = ImageList.FirstOrDefault(x => x.StatusColor == ThumbStatus.Ng);
                if (firstError != null)
                {
                    SelectedThumb = firstError;
                    ShowViewer();
                }
            });
        }

        public void AddImage(BitmapSource img, string title, ThumbStatus thumbStatus, string errorDescription, bool autoShow = false)
        {
            InvokeUI(() =>
            {
                ImageList.Add(new ThumbItem(img, title, thumbStatus, errorDescription));

                if (autoShow)
                {
                    SelectedThumb = ImageList[ImageList.Count - 1];
                }
            });
        }

        // ================= IMAGE PAN + ZOOM =================

        private void imbImage_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            imbImage.CaptureMouse();
            _start = e.GetPosition(this);
            _origin = new Point(translateTransform.X, translateTransform.Y);
            Cursor = Cursors.Hand;
        }

        private void imbImage_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            imbImage.ReleaseMouseCapture();
            Cursor = Cursors.Arrow;
        }

        private void imbImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (!imbImage.IsMouseCaptured) return;

            Point p = e.GetPosition(this);
            translateTransform.X = _origin.X + (p.X - _start.X);
            translateTransform.Y = _origin.Y + (p.Y - _start.Y);
        }

        private void imbImage_MouseWheel(object sender, MouseWheelEventArgs e)
        {
            double zoom = e.Delta > 0 ? 1.1 : 0.9;

            scaleTransform.ScaleX *= zoom;
            scaleTransform.ScaleY *= zoom;
        }

        private void imbImage_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.ClickCount == 2)
                ResetView();
        }

        private async void ResetView()
        {
            var duration = TimeSpan.FromMilliseconds(150);

            scaleTransform.BeginAnimation(ScaleTransform.ScaleXProperty, new DoubleAnimation(1, duration));
            scaleTransform.BeginAnimation(ScaleTransform.ScaleYProperty, new DoubleAnimation(1, duration));
            translateTransform.BeginAnimation(TranslateTransform.XProperty, new DoubleAnimation(0, duration));
            translateTransform.BeginAnimation(TranslateTransform.YProperty, new DoubleAnimation(0, duration));

            await Task.Delay(duration);

            // Xóa animation, trả quyền điều khiển về code
            scaleTransform.BeginAnimation(ScaleTransform.ScaleXProperty, null);
            scaleTransform.BeginAnimation(ScaleTransform.ScaleYProperty, null);
            translateTransform.BeginAnimation(TranslateTransform.XProperty, null);
            translateTransform.BeginAnimation(TranslateTransform.YProperty, null);

            scaleTransform.ScaleX = 1;
            scaleTransform.ScaleY = 1;
            translateTransform.X = 0;
            translateTransform.Y = 0;
        }

        // ================= BUTTON =================

        private void btnBack_Click(object sender, RoutedEventArgs e)
        {
            var curIndex = lbThumbList.SelectedIndex;
            if (curIndex > 0)
                SelectedThumb = ImageList[curIndex - 1];
        }

        private void btnNext_Click(object sender, RoutedEventArgs e)
        {
            var curIndex = lbThumbList.SelectedIndex;
            if (curIndex < ImageList.Count - 1)
                SelectedThumb = ImageList[curIndex + 1];
        }

        private void ShowButton(Button btn)
        {
            btn.Visibility = Visibility.Visible;
            btn.BeginAnimation(OpacityProperty,
                new DoubleAnimation(1, TimeSpan.FromMilliseconds(120)));
        }

        private void HideButton(Button btn)
        {
            var anim = new DoubleAnimation(0, TimeSpan.FromMilliseconds(120));
            anim.Completed += (s, e) => btn.Visibility = Visibility.Collapsed;
            btn.BeginAnimation(OpacityProperty, anim);
        }

        private void grView_MouseMove(object sender, MouseEventArgs e)
        {
            if (ImageList.Count <= 1)
            {
                HideButton(btnBack);
                HideButton(btnNext);
                return;
            }

            Point p = e.GetPosition(bdView);
            double w = bdView.ActualWidth;

            if (p.X < EDGE_ZONE && lbThumbList.SelectedIndex > 0)
                ShowButton(btnBack);
            else
                HideButton(btnBack);

            if (p.X > w - EDGE_ZONE && lbThumbList.SelectedIndex < ImageList.Count - 1)
                ShowButton(btnNext);
            else
                HideButton(btnNext);
        }

        private void grView_MouseLeave(object sender, MouseEventArgs e)
        {
            HideButton(btnBack);
            HideButton(btnNext);
        }
        protected override void OnClosing(CancelEventArgs e)
        {
            if (Application.Current.ShutdownMode == ShutdownMode.OnExplicitShutdown)
            {
                base.OnClosing(e);
                return;
            }

            e.Cancel = true;
            Hide();
        }
    }
}
