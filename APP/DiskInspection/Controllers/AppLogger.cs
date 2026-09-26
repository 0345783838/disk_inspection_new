using DiskInspection.Models;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Threading;

namespace DiskInspection.Controllers
{
    public class AppLogger
    {
        private static readonly Lazy<AppLogger> _instance =
            new Lazy<AppLogger>(() => new AppLogger());

        public static AppLogger Instance
        {
            get { return _instance.Value; }
        }

        public ObservableCollection<LogItem> Logs { get; private set; }

        private const int MAX_LOG = 300;

        private AppLogger()
        {
            Logs = new ObservableCollection<LogItem>();
        }

        public void Log(LogLevel level, string message, string source)
        {
            if (Application.Current == null)
                return;

            Application.Current.Dispatcher.Invoke(new Action(() =>
            {
                if (Logs.Count >= MAX_LOG)
                {
                    Logs.RemoveAt(0);
                }

                Logs.Add(new LogItem(level, source, message));
            }));
        }
        public void LogSafe(LogLevel level, string message, string source)
        {
            var dispatcher = Application.Current != null
                ? Application.Current.Dispatcher
                : Dispatcher.CurrentDispatcher;

            dispatcher.Invoke(() =>
            {
                if (Logs.Count >= MAX_LOG)
                    Logs.RemoveAt(0);

                Logs.Add(new LogItem(level, source, message));
            });
        }

        // Helper methods (C# 7.3 vẫn OK)
        public void Info(string message, string source)
        {
            Log(LogLevel.Info, message, source);
        }

        public void Success(string message, string source)
        {
            Log(LogLevel.Success, message, source);
        }

        public void Warning(string message, string source)
        {
            Log(LogLevel.Warning, message, source);
        }

        public void Error(string message, string source)
        {
            Log(LogLevel.Error, message, source);
        }
    }
}
