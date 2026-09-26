using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Security
{
    public static class RsaVerifier
    {
        public static bool VerifySignature(string payloadJson, string signatureBase64)
        {
            byte[] dataBytes = Encoding.UTF8.GetBytes(payloadJson);
            byte[] sigBytes = Convert.FromBase64String(signatureBase64);

            using (var rsa = new RSACryptoServiceProvider())
            {
                rsa.FromXmlString(LicenseKeys.PUBLIC_KEY);

                return rsa.VerifyData(
                    dataBytes,
                    CryptoConfig.MapNameToOID("SHA256"),
                    sigBytes
                );
            }
        }
    }
}
