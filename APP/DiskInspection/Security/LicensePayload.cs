using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Security
{
    public class LicensePayload
    {
        public string MachineId { get; set; }
        public string Challenge { get; set; }
        public string LicenseType { get; set; }
        public string Expire { get; set; }
    }

    public class LicenseObject
    {
        public LicensePayload data { get; set; }
        public string sig { get; set; }
    }
}
