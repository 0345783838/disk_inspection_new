using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Threading.Tasks;
using DiskInspection.Views.UtilitiesWindows;

namespace DiskInspection.Services
{
    public static class ErrorService
    {
        private static bool _isShowing;
        private static readonly object _lock = new object();

        public static void ShowError(string content)
        {
            lock (_lock)
            {
                if (_isShowing)
                    return;

                _isShowing = true;
            }

            Application.Current.Dispatcher.Invoke(() =>
            {
                try
                {
                    var dialog = new ErrorWindow(content);
                    dialog.ShowDialog();
                }
                finally
                {
                    lock (_lock)
                    {
                        _isShowing = false;
                    }
                }
            });
        }
    }
}
