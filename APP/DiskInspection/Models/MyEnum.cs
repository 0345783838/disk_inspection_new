using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;

namespace DiskInspection.Models
{
    enum StatusState
    {
        Ok = 0,
        Ng = 1,
        Inspecting = 3,
        Unknown = 2,
        Stopped = 4,
        Warning = 5
    }
    enum TriggerState
    {
        Ok = 1,
        Error = -1
    }
    public enum LogLevel
    {
        Info,
        Success,
        Warning,
        Error
    }
    public enum ThumbStatus
    {
        Origin,
        Ok,
        Ng,
        Warning
    }
    public enum SaveType
    {
        ORIGINAL_RESULT = 0,
        RESULT = 1,
        ORIGINAL = 2
    }
    public enum SaveOption
    {
        OK = 0,
        NG = 1,
        BOTH = 2
    }
    public enum InspectionResult { Passed, Failed, Warning }
    public static class CameraName
    {
        public static readonly string CAM_1 = "Camera 1";
        public static readonly string CAM_2 = "Camera 2";
    }
    public static class ErrorCode
    {
        public static readonly string PASS = "PASS";
        public static readonly string ERROR_001 = "ERROR_001";
        public static readonly string ERROR_002 = "ERROR_002";
        public static readonly string ERROR_003 = "ERROR_003";
        public static readonly string WARNING_001 = "WARNING_001";
        
    }
}
