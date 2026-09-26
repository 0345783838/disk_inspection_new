using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace DiskInspection.Controllers
{
    /// <summary>Serializes writes of frozen snapshots captured by the owning inspection cycle.</summary>
    internal sealed class InspectionImageWriter
    {
        private readonly object _gate = new object();
        private Task _pending = Task.CompletedTask;
        private readonly Func<BitmapSource, string, DateTime, string, bool> _save;

        public InspectionImageWriter(Func<BitmapSource, string, DateTime, string, bool> save = null)
        {
            _save = save ?? ImageSaver.SaveImage;
        }

        public void Enqueue(string path, DateTime timestamp, IReadOnlyDictionary<string, BitmapSource> images)
        {
            var snapshot = new Dictionary<string, BitmapSource>();
            foreach (var image in images) snapshot.Add(image.Key, image.Value);
            lock (_gate)
            {
                _pending = _pending.ContinueWith(_ =>
                {
                    try
                    {
                        bool success = true;
                        foreach (var image in snapshot)
                        {
                            try { success = _save(image.Value, path, timestamp, image.Key) && success; }
                            catch (Exception ex)
                            {
                                success = false;
                                AppLogger.Instance.Error($"Cannot save {image.Key}: {ex.Message}", "SAVE");
                            }
                        }
                        if (!success)
                            AppLogger.Instance.Error("Some inspection images could not be saved.", "SAVE");
                    }
                    catch (Exception ex)
                    {
                        AppLogger.Instance.Error($"Image saving failed: {ex.Message}", "SAVE");
                    }
                }, TaskScheduler.Default);
            }
        }

        public Task FlushAsync()
        {
            lock (_gate) return _pending;
        }
    }
}
