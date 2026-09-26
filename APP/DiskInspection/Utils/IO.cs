using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Utils
{
    class IO
    {
        static private NLog.Logger logger = NLog.LogManager.GetLogger("debug");
        public static void CreateFolderIfNotExists(string path)
        {
            try
            {
                if (!Directory.Exists(path))
                    Directory.CreateDirectory(path);
            }
            catch (Exception e)
            {
                logger.Error(e.Message);
            }

        }
        public static string GetFileName(string path)
        {
            return Path.GetFileName(path);
        }

        public static string GetParentFolderFromFilePath(string path)
        {
            return Path.GetFileName(Directory.GetParent(path).ToString());
        }
        public static string GetParentChildFolderFromFolderPath(string path)
        {
            var child = Path.GetFileName(path);
            var parent = Path.GetFileName(Directory.GetParent(path).ToString());
            return $"{parent}\\{child}";
        }
        public static string GetParentFolderFromFolderPath(string path)
        {
            var parent = Path.GetFileName(Directory.GetParent(path).ToString());
            return $"{parent}";
        }
        public static string GetFileNameWithoutExtension(string path)
        {
            return System.IO.Path.GetFileNameWithoutExtension(path);
        }
    }
}
