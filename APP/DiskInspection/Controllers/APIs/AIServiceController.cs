using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Controllers
{
    public class AIServiceController
    {
        private static NLog.Logger logger = NLog.LogManager.GetCurrentClassLogger();
        public static void Start()
        {
            try
            {
                System.Diagnostics.Process p = new System.Diagnostics.Process();
                System.Diagnostics.ProcessStartInfo info = new System.Diagnostics.ProcessStartInfo();
                info.FileName = "cmd.exe";
                info.Arguments = "/c main.exe";
                info.UseShellExecute = false;
                info.CreateNoWindow = true;
                //info.RedirectStandardOutput = true;
                //info.RedirectStandardError = true;
                info.WorkingDirectory = "plugin";
                info.Arguments = "/c main.exe";
                p.StartInfo = info;
                p.Start();
            }
            catch (Exception ex)
            {
                logger.Debug(ex.Message);
            }
        }
        public static bool CheckProcessExisting()
        {
            Process[] processes = Process.GetProcessesByName("main");
            if (processes.Length > 0)
            {
                return true;
            }
            return false;
        }
        public static void CloseProcessExisting()
        {
            Process[] processes = Process.GetProcessesByName("main");
            foreach (var process in processes)
            {
                process.Kill();
                process.WaitForExit();
                process.Dispose();
            }
        }
    }
}
