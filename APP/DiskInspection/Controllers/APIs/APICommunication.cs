using DiskInspection.Models;
using Emgu.CV;
using Emgu.CV.Structure;
using Newtonsoft.Json;
using RestSharp;
using System;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace DiskInspection.Controllers.APIs
{
    internal static class APICommunication
    {
        private static readonly NLog.Logger Logger = NLog.LogManager.GetCurrentClassLogger();
        private static Properties.Settings Settings => Properties.Settings.Default;

        public static DebugImageResponse DebugImages(string url, Mat image, EnvironmentConfig config, int timeout = 10000)
        {
            var payload = new
            {
                segment_threshold = config.SegmentThreshold, segment_iou = config.SegmentIou,
                detect_threshold = config.DetectThreshold, detect_iou = config.DetectIou,
                caliper_min_edge_distance = config.CaliperMinEdgeDistance,
                caliper_max_edge_distance = config.CaliperMaxEdgeDistance,
                caliper_length_rate = config.CaliperLengthRate,
                caliper_thickness_list = config.CaliperThicknessList,
                disk_num = config.DiskNumber, disk_max_distance = config.DiskMaxDistance,
                disk_min_distance = config.DiskMinDistance, disk_min_area = config.DiskMinArea
            };
            return DebugRequest<DebugImageResponse>(url, Settings.EndPointDebug, image, "params_json", payload, timeout);
        }

        public static DebugUvImageResponse DebugUvImages(string url, Mat image, string crop_box, string uv_box_1,
            string uv_box_2, string mid_1, string mid_2, EnvironmentConfig config, int timeout = 10000)
        {
            var payload = new
            {
                uv_disk_lower_threshold = config.UvLowerThreshold, uv_disk_upper_threshold = config.UvUpperThreshold,
                uv_disk_min_area = config.UvMinArea, crop_box, uv_box_1, uv_box_2, mid_1, mid_2
            };
            return DebugRequest<DebugUvImageResponse>(url, Settings.EndPointDebugUv, image, "params_json", payload, timeout);
        }

        public static Task<InspectionResponse> InspectWhiteLightAsync(string url, Mat image,
            CancellationToken token, int timeout = 10000) =>
            SendImageAsync<InspectionResponse>(url, Settings.EndpointInspectWhiteLight, image, null, null, timeout, token);

        public static Task<InspectionUvResponse> InspectUvLightAsync(string url, Mat image, string crop_box,
            string uv_box_1, string uv_box_2, string mid_1, string mid_2, CancellationToken token, int timeout = 10000) =>
            SendImageAsync<InspectionUvResponse>(url, Settings.EndpointInspectUvLight, image, "uv_box",
                new { crop_box, uv_box_1, uv_box_2, mid_1, mid_2 }, timeout, token);

        private static T DebugRequest<T>(string url, string endpoint, Mat image, string parameter,
            object payload, int timeout) where T : class
        {
            try { return SendImageAsync<T>(url, endpoint, image, parameter, payload, timeout, CancellationToken.None).GetAwaiter().GetResult(); }
            catch (Exception ex) { Logger.Error(ex, $"Debug request failed: {endpoint}"); return null; }
        }

        private static async Task<T> SendImageAsync<T>(string url, string endpoint, Mat image,
            string parameter, object payload, int timeout, CancellationToken token) where T : class
        {
            token.ThrowIfCancellationRequested();
            if (image == null || image.IsEmpty) throw new InvalidOperationException("Cannot inspect an empty image.");
            var request = new RestRequest(endpoint, Method.Post) { AlwaysMultipartFormData = true };
            using (var bgr = image.ToImage<Bgr, byte>())
                request.AddFile("image", bgr.ToJpegData(), "image.jpg");
            if (payload != null)
                request.AddParameter(parameter, JsonConvert.SerializeObject(payload), ParameterType.GetOrPost);
            using (var client = new RestClient(new RestClientOptions(url) { Timeout = TimeSpan.FromMilliseconds(timeout) }))
            {
                var response = await client.ExecuteAsync(request, token).ConfigureAwait(false);
                token.ThrowIfCancellationRequested();
                if (response.StatusCode != HttpStatusCode.OK || response.ErrorException != null)
                    throw new InvalidOperationException($"AI request {endpoint} failed ({response.ResponseStatus}, HTTP {(int)response.StatusCode}).", response.ErrorException);
                var result = JsonConvert.DeserializeObject<T>(response.Content ?? "null");
                if (result == null) throw new InvalidOperationException($"AI request {endpoint} returned an empty result.");
                return result;
            }
        }

        public static bool CheckAPIStatus(string url, int timeout = 1000)
        {
            try
            {
                using (var client = new RestClient(new RestClientOptions(url) { Timeout = TimeSpan.FromMilliseconds(timeout) }))
                    return client.Execute(new RestRequest(Settings.EndPointCheckStatus, Method.Get)).StatusCode == HttpStatusCode.OK;
            }
            catch (Exception ex) { Logger.Debug(ex, "Service health check failed."); return false; }
        }
    }
}
