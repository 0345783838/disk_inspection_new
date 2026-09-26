using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace DiskInspection.Controllers
{
    public class ImageCleanupService : IDisposable
    {
        private readonly Properties.Settings _param = Properties.Settings.Default;

        private CancellationTokenSource _cts;
        private Task _workerTask;

        private readonly TimeSpan _interval = TimeSpan.FromHours(6);

        public void Start()
        {
            if (_cts != null)
                return;

            _cts = new CancellationTokenSource();
            _workerTask = Task.Run(() => RunAsync(_cts.Token));
        }

        public void Stop()
        {
            try
            {
                if (_cts == null)
                    return;

                _cts.Cancel();

                try
                {
                    _workerTask?.Wait(3000);
                }
                catch
                {
                    // ignore
                }
            }
            finally
            {
                _cts?.Dispose();
                _cts = null;
                _workerTask = null;
            }
        }

        private async Task RunAsync(CancellationToken token)
        {
            // Chạy 1 lần ngay khi start app
            await CleanupSafeAsync(token);

            while (!token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(_interval, token);
                    await CleanupSafeAsync(token);
                }
                catch (TaskCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Image cleanup loop error:");
                    Console.WriteLine(ex);
                }
            }
        }

        private Task CleanupSafeAsync(CancellationToken token)
        {
            return Task.Run(() =>
            {
                try
                {
                    CleanupOldImages(token);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("CleanupOldImages error:");
                    Console.WriteLine(ex);
                }
            }, token);
        }

        private void CleanupOldImages(CancellationToken token)
        {
            string rootPath = _param.SavePath;
            int deleteDays = _param.DeleteDays;

            if (string.IsNullOrWhiteSpace(rootPath))
                return;

            if (!Directory.Exists(rootPath))
                return;

            // DeleteDays <= 0 thì hiểu là không tự xóa
            if (deleteDays <= 0)
                return;

            DateTime minDateToKeep = DateTime.Today.AddDays(-deleteDays + 1);

            foreach (string dateDir in Directory.GetDirectories(rootPath))
            {
                token.ThrowIfCancellationRequested();

                string folderName = Path.GetFileName(dateDir);

                DateTime folderDate;
                bool parsed = DateTime.TryParseExact(
                    folderName,
                    "dd_MM_yyyy",
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None,
                    out folderDate);

                if (!parsed)
                {
                    // Không đúng format dd_MM_yyyy thì bỏ qua, tránh xóa nhầm folder khác
                    continue;
                }

                if (folderDate < minDateToKeep)
                {
                    TryDeleteDirectory(dateDir);
                }
            }
        }

        private void TryDeleteDirectory(string dir)
        {
            try
            {
                Directory.Delete(dir, true);
                AppLogger.Instance.Info($"Deleted old image folder: {dir}", "SYSYEM");
            }
            catch (Exception ex)
            {
                AppLogger.Instance.Error($"Cannot delete folder: {ex}", "SYSYEM");
            }
        }

        public void Dispose()
        {
            Stop();
        }
    }
}
