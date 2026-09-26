using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Management;

namespace DiskInspection.Security
{
    public static class MachineInfo
    {
        public static string GetMachineId()
        {
            string cpu = GetWMI("Win32_Processor", "ProcessorId");
            string board = GetWMI("Win32_BaseBoard", "SerialNumber");
            string disk = GetWMI("Win32_DiskDrive", "SerialNumber");

            string raw = cpu + board + disk;

            using (SHA256 sha = SHA256.Create())
            {
                byte[] hash = sha.ComputeHash(Encoding.UTF8.GetBytes(raw));

                return BitConverter.ToString(hash).Replace("-", "").Substring(0, 16);
            }
        }

        static string GetWMI(string cls, string prop)
        {
            try
            {
                var searcher = new ManagementObjectSearcher($"SELECT {prop} FROM {cls}");

                foreach (ManagementObject obj in searcher.Get())
                {
                    return obj[prop]?.ToString() ?? "";
                }
            }
            catch { }

            return "";
        }
    }
}
