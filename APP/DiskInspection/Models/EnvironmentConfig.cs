using OpenTK.Graphics.ES11;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    public class EnvironmentConfig
    {
        public float DetectThreshold { get; set; }
        public float DetectIou { get; set; }
        public float SegmentThreshold { get; set; }
        public float SegmentIou { get; set; }

        public float CaliperMinEdgeDistance { get; set; }
        public float CaliperMaxEdgeDistance { get; set; }
        public float CaliperLengthRate { get; set; }
        public List<int> CaliperThicknessList { get; set; }

        public int DiskNumber { get; set; }
        public float DiskMaxDistance { get; set; }
        public float DiskMinDistance { get; set; }
        public float DiskMinArea { get; set; }

        public List<int> UvLowerThreshold { get; set; }
        public List<int> UvUpperThreshold { get; set; }
        public float UvMinArea { get; set; }

        public EnvironmentConfig() { }
        public EnvironmentConfig(float detectThreshold, float detectIoU, float segmentThreshold, float segmentIoU, float caliperMinEdgeDistance, 
            float caliperMaxEdgeDistance, float caliperLengthRate, List<int> caliperThicknessList, int diskNumber, float diskMaxDistance, 
            float diskMinDistance, float diskMinArea, List<int> uvLowerThreshold, List<int> uvUpperThreshold, float uvMinArea)
        {
            DetectThreshold = detectThreshold;
            DetectIou = detectIoU;
            SegmentThreshold = segmentThreshold;
            SegmentIou = segmentIoU;
            CaliperMinEdgeDistance = caliperMinEdgeDistance;
            CaliperMaxEdgeDistance = caliperMaxEdgeDistance;
            CaliperLengthRate = caliperLengthRate;
            CaliperThicknessList = caliperThicknessList;
            DiskNumber = diskNumber;
            DiskMaxDistance = diskMaxDistance;
            DiskMinDistance = diskMinDistance;
            DiskMinArea = diskMinArea;
            UvLowerThreshold = uvLowerThreshold;
            UvUpperThreshold = uvUpperThreshold;
            UvMinArea = uvMinArea;
        }
    }
}
