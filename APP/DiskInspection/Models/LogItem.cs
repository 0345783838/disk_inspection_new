using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    public class LogItem
    {
        public DateTime Time { get; private set; }
        public LogLevel Level { get; private set; }
        public string Source { get; private set; }
        public string Message { get; private set; }

        public string TimeText
        {
            get { return Time.ToString("HH:mm:ss.fff"); }
        }

        public LogItem(LogLevel level, string source, string message)
        {
            Time = DateTime.Now;
            Level = level;
            Source = source;
            Message = message;
        }
    }
}
