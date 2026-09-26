using DiskInspection.Security;
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

namespace DiskInspection.Views.ActivationWindows
{
    /// <summary>
    /// Interaction logic for ActivationWindow.xaml
    /// </summary>
    public partial class ActivationWindow : Window
    {
        private string _challenge;

        public ActivationWindow()
        {
            InitializeComponent();

            tbMachineId.Text = MachineInfo.GetMachineId();

            _challenge = Guid.NewGuid().ToString("N");

            tbChallenge.Text = _challenge;
        }

        private void Activate_Click(object sender, RoutedEventArgs e)
        {
            string key = tbActivationKey.Text.Trim();

            (bool isValid, string message) = LicenseManager.ValidateActivationKey(key);
            if (isValid)
            {
                LicenseManager.SaveLicense(key);
                DialogResult = true;
                Close();
            }
            else
            {
                DialogResult = false;
            }
        }
    }
}
