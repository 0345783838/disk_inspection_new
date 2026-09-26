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
using System.Windows.Shapes;

namespace DiskInspection.Views.UtilitiesWindows
{
    /// <summary>
    /// Interaction logic for WaitingWindow.xaml
    /// </summary>
    public partial class WaitingWindow : Window
    {
        private bool _KillMe = false;
        public string _Content = "Processing...";

        private System.Timers.Timer _Timer = new System.Timers.Timer(1000);
        private int _CountSecond = 0;
        private int _TimeOut = 180;
        public string LabelContent
        {
            get
            {
                return _Content;
            }
            set
            {
                _Content = value;
                _CountSecond = 0;
                this.Dispatcher.Invoke(() => {
                    lbStatus.Content = _Content;
                });
            }
        }
        private void OntimedEvent(object sender, System.Timers.ElapsedEventArgs e)
        {
            _CountSecond++;
            if (_CountSecond > _TimeOut)
            {
                _Timer.Enabled = false;
                this.Dispatcher.Invoke(() => {
                    lbStatus.Content = _Content;
                });
                KillMe = true;
            }
        }
        public WaitingWindow(string Content = "Processing...", int Timeout = 180)
        {
            InitializeComponent();
            this.LabelContent = Content;
            _TimeOut = Timeout;
            _Timer.Elapsed += OntimedEvent;
            _Timer.Enabled = true;
        }
        public WaitingWindow(string Content = "Processing...")
        {
            InitializeComponent();
            this.LabelContent = Content;
            _Timer.Elapsed += OntimedEvent;
            _Timer.Enabled = true;
        }
        public bool KillMe
        {
            get { return _KillMe; }
            set
            {
                _KillMe = value;
                if (_KillMe == true)
                {
                    this.Dispatcher.Invoke(() => {
                        _Timer.Enabled = false;
                        this.Close();
                    });
                }
            }
        }

        private void btOK_Click(object sender, RoutedEventArgs e)
        {
            this.Dispatcher.Invoke(() => {
                _Timer.Enabled = false;
                this.Close();
            });
        }
    }
}
