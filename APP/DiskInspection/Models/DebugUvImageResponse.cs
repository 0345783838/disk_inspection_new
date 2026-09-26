using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    public class DebugUvImageResponse
    {
        public bool Result { get; set; }
        public string ThresholdImg { get; set; }
        public string FinalImg { get; set; }
    }
}
