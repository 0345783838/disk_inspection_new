namespace DiskInspection.Models
{
    /// <summary>Immutable settings captured when a batch is queued. Quality 0 means lossless PNG.</summary>
    public sealed class ImageSaveOptions
    {
        public static ImageSaveOptions Default { get; } = new ImageSaveOptions(90);

        public int JpegQuality { get; }
        public bool IsLossless => JpegQuality == 0;

        public ImageSaveOptions(int jpegQuality)
        {
            JpegQuality = jpegQuality >= 0 && jpegQuality <= 100 ? jpegQuality : 90;
        }
    }
}
