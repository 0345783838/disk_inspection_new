using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    enum FileStatus
    {
        OK = 0,
        NG = 1,
        NOT_DONE = 3,
        WARNING = 2
    }
    public class ImageDebugInfo :INotifyPropertyChanged
    {
        public event PropertyChangedEventHandler PropertyChanged;
        protected void OnPropertyChanged([System.Runtime.CompilerServices.CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public int ID { get; set; }
        public string FilePath { get; set; }
        private List<ImageList> _images;
        public List<ImageList> Images
        {
            get => _images;
            set
            {
                if (_images != value)
                {
                    _images = value;
                    OnPropertyChanged();
                }
            }
        }

        private int _status;
        public int Status
        {
            get => _status;
            set
            {
                if (_status != value)
                {
                    _status = value;
                    OnPropertyChanged();
                }
            }
        }
        public ImageDebugInfo(int id, string filePath)
        {
            ID = id;
            FilePath = filePath;
            Status = (int)FileStatus.NOT_DONE;
            Images = new List<ImageList>();
        }
    }
}
