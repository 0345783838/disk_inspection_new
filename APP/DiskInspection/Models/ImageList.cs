using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Models
{
    public class ImageList
    {
        public int ID { get; set; }
        public string Title { get; set; }
        public Bitmap Image { get; set; }

        public ImageList(int iD, string title, Bitmap image)
        {
            ID = iD;
            Title = title;
            Image = image;
        }
    }
}
