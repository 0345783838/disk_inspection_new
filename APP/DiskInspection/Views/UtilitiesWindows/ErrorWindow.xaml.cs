using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace DiskInspection.Views.UtilitiesWindows
{
    /// <summary>
    /// Interaction logic for ErrorWindow.xaml
    /// </summary>
    public partial class ErrorWindow : Window
    {
        public ErrorWindow(string message)
        {
            InitializeComponent();
            tbContent.Text = message;
        }

        private void bdOK_MouseDown(object sender, MouseButtonEventArgs e)
        {
            this.Close();
        }
    }
}
