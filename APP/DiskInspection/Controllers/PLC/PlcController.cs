using DiskInspection.Models;
using Newtonsoft.Json;
using RestSharp;
using System;
using System.Net;

namespace DiskInspection.Controllers.PLC
{
    internal static class PlcController
    {
        private static readonly NLog.Logger Logger = NLog.LogManager.GetCurrentClassLogger();
        private static Properties.Settings Settings => Properties.Settings.Default;
        private sealed class CommandResponse
        {
            [JsonProperty(Required = Required.Always)]
            public bool Success { get; set; }
        }
        private sealed class TriggerResponse
        {
            [JsonProperty(Required = Required.Always)]
            public int Success { get; set; }
            [JsonProperty(Required = Required.Always)]
            public bool Status { get; set; }
        }

        private static T Get<T>(string url, string endpoint, int timeout, Action<RestRequest> configure = null) where T : class
        {
            try
            {
                using (var client = new RestClient(new RestClientOptions(url) { Timeout = TimeSpan.FromMilliseconds(timeout) }))
                {
                    var request = new RestRequest(endpoint, Method.Get);
                    configure?.Invoke(request);
                    var response = client.Execute(request);
                    if (response.StatusCode != HttpStatusCode.OK || response.ErrorException != null) return null;
                    return JsonConvert.DeserializeObject<T>(response.Content ?? "null");
                }
            }
            catch (Exception ex) { Logger.Error(ex, $"PLC request failed: {endpoint}"); return null; }
        }

        private static bool Command(string url, string endpoint, int timeout, Action<RestRequest> configure = null) =>
            Get<CommandResponse>(url, endpoint, timeout, configure)?.Success == true;

        public static bool ConnectPlc(string url, string ip, int port, int timeout = 1500) =>
            Command(url, Settings.EndpointConnectPlc, timeout, request =>
            {
                request.AddQueryParameter("ip", ip);
                request.AddQueryParameter("port", port);
            });

        internal static (TriggerState, bool) CheckTrigger(string url, int timeout = 1500)
        {
            var result = Get<TriggerResponse>(url, Settings.EndpointReadTrigger, timeout);
            return result?.Success == (int)TriggerState.Ok
                ? (TriggerState.Ok, result.Status) : (TriggerState.Error, false);
        }

        public static bool DisConnectPlc(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointDisconnectPlc, timeout);

        public static bool ResetTrigger(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointResetTrigger, timeout);

        public static bool CheckPlcConnection(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointCheckConnection, timeout);

        public static bool OnErrorAbnormal(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointOnErrorAbnormal, timeout);

        public static bool OnErrorMixing(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointOnErrorMixing, timeout);

        public static bool OnOkSignal(string url, int timeout = 1500) =>
            Command(url, Settings.EndpointOnOkSignal, timeout);

        public static bool ControlWhiteLight(string url, bool status, int timeout = 1500) =>
            Command(url, Settings.EndpointControlWhiteLight, timeout, request => request.AddQueryParameter("status", status));

        public static bool ControlUvLight(string url, bool status, int timeout = 1500) =>
            Command(url, Settings.EndpointControlUvLight, timeout, request => request.AddQueryParameter("status", status));
    }
}
