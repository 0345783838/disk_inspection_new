using Newtonsoft.Json;
using DiskInspection.Models;
using DiskInspection.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Security
{
    public static class LicenseManager
    {
        //public static bool ValidateActivationKey(string activationKey)
        //{
        //    string json = string.Empty;
        //    try
        //    {
        //        json = Encoding.UTF8.GetString(
        //        Convert.FromBase64String(activationKey));
        //    }
        //    catch
        //    {
        //        return false;
        //    }

        //    var license = JsonConvert.DeserializeObject<LicenseObject>(json);

        //    string payloadJson = JsonConvert.SerializeObject(license.data);

        //    bool valid = RsaVerifier.VerifySignature(payloadJson, license.sig);

        //    if (!valid)
        //        return false;

        //    if (license.data.MachineId != MachineInfo.GetMachineId())
        //        return false;

        //    return true;
        //}
        public static (bool, string) ValidateActivationKey(string activationKey)
        {
            string json = string.Empty;

            try
            {
                json = Encoding.UTF8.GetString(
                    Convert.FromBase64String(activationKey));
            }
            catch
            {
                return (false, "Invalid activation key"); //false;
            }

            LicenseObject license;

            try
            {
                license = JsonConvert.DeserializeObject<LicenseObject>(json);
            }
            catch
            {
                return (false, "Invalid activation key"); 
            }

            // verify RSA signature
            string payloadJson = JsonConvert.SerializeObject(license.data);

            bool valid = RsaVerifier.VerifySignature(payloadJson, license.sig);

            if (!valid)
                return (false, "Invalid activation key"); 

            // check machine id
            if (license.data.MachineId != MachineInfo.GetMachineId())
                return (false, "Invalid activation key");

            // check expire
            if (license.data.LicenseType == "time")
            {
                if (string.IsNullOrEmpty(license.data.Expire))
                    return (false, "Invalid activation key");

                if (!DateTime.TryParse(license.data.Expire, out DateTime expire))
                    return (false, "Invalid activation key");

                if (DateTime.Now > expire)
                {
                    return (false, "Activation key has expired, contact vendor to get a new one.");
                }      
            }

            return (true, "");
        }
        public static void SaveLicense(string key)
        {
            string dir = @"plugin";

            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            File.WriteAllText(Path.Combine(dir, "license.dat"), key);
        }
    }
}
