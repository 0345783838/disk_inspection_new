using DiskInspection.Controllers;
using DiskInspection.Views;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;

namespace DiskInspection
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        private ImageCleanupService _imageCleanupService;
        public static ImageViewerWindow ImageViewer { get; private set; }

        protected override void OnStartup(StartupEventArgs e)
        {
            // plugin/config/license/user.ini paths belong to this application,
            // including when a shortcut supplies a different working directory.
            System.IO.Directory.SetCurrentDirectory(AppDomain.CurrentDomain.BaseDirectory);
            base.OnStartup(e);

            ImageViewer = new ImageViewerWindow();
            ImageViewer.Hide();   // chạy nền

            _imageCleanupService = new ImageCleanupService();
            _imageCleanupService.Start();
        }
        protected override void OnExit(ExitEventArgs e)
        {
            _imageCleanupService?.Stop();
            base.OnExit(e);
        }
    }
}
