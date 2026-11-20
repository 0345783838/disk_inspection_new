using DiskInspection.Models;
using System;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Threading;
using NLog;
using LogLevel = DiskInspection.Models.LogLevel;

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

        public void Log(LogLevel level, string message, string source, Exception exception = null)
        {
            var fileLevel = level == LogLevel.Error ? NLog.LogLevel.Error
                : level == LogLevel.Warning ? NLog.LogLevel.Warn : NLog.LogLevel.Info;
            LogManager.GetLogger(source).Log(new LogEventInfo(fileLevel, source, message) { Exception = exception });

            var dispatcher = Application.Current?.Dispatcher;
            if (dispatcher == null || dispatcher.HasShutdownStarted) return;
            ScopeContext.TryGetProperty("cycle", out var cycle);
            var item = new LogItem(level, cycle == null ? source : $"{source} {cycle}", message);
            dispatcher.BeginInvoke(new Action(() =>
            {
                if (Logs.Count >= MAX_LOG)
                    Logs.RemoveAt(0);
                Logs.Add(item);
            }), DispatcherPriority.Background);
        }

        public void LogSafe(LogLevel level, string message, string source) => Log(level, message, source);

        public void Debug(string message, string source)
        {
            LogManager.GetLogger(source).Debug(message);
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

        public void Error(string message, string source, Exception exception = null)
        {
            Log(LogLevel.Error, message, source, exception);
        }
    }
}
