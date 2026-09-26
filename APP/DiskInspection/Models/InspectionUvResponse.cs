using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    public class InspectionUvResponse
    {
        [Newtonsoft.Json.JsonProperty(Required = Newtonsoft.Json.Required.Always)]
        public bool Result { get; set; }
        public string ErrorCode { get; set; }
        public string ErrorDesc { get; set; }
        public string ResImg { get; set; }
        public int CountUvDisk { get; set; }
    }
}
