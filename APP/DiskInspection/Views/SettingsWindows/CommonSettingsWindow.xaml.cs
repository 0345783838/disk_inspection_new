using DiskInspection.Controllers;
using DiskInspection.Controllers.APIs;
using DiskInspection.Controllers.Camera;
using DiskInspection.Controllers.PLC;
using DiskInspection.Models;
using DiskInspection.Views.UtilitiesWindows;
using Emgu.CV;
using LiveCharts.Wpf;
using Microsoft.WindowsAPICodePack.Dialogs;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace DiskInspection.Views.SettingsWindows
{
    /// <summary>
    /// Interaction logic for CommonSettingsWindow.xaml
    /// </summary>
    public partial class CommonSettingsWindow : Window
    {
        private Properties.Settings _param = Properties.Settings.Default;
        private CameraManager _cameraManager;

        public CommonSettingsWindow()
        {
            InitializeComponent();
            Init();
            _cameraManager = CameraManager.GetInstance();
        }
        private void Init()
        {
            // Get cameras list
            List<CamInfo> camInfoList = LincolnCamera.GetListCamInfo();

            for (int i = 0; i < camInfoList.Count; i++)
            {
                cbbCam1Sn.Items.Add(camInfoList[i].SN);
                cbbCam2Sn.Items.Add(camInfoList[i].SN);
            }

            // Hardware Settings
            cbbCam1Sn.Text = _param.Cam1Sn;
            tbCam1Exposure.Text = _param.Cam1Exposure.ToString();
            cbbCam2Sn.Text = _param.Cam2Sn;
            tbCam2Exposure.Text = _param.Cam2Exposure.ToString();
            tbPlcIp.Text = _param.PlcIp;
            tbPlcPort.Text = _param.PlcPort.ToString();

            tbWhiteLightOnDelay.Text = _param.WaitWhiteLightOn.ToString();
            tbWhiteLightOffDelay.Text = _param.WaitWhiteLightOff.ToString();
            tbUvLightOnDelay.Text = _param.WaitUvLightOn.ToString();
            tbUvLightOffDelay.Text = _param.WaitUvLightOff.ToString();

            // Saving Settings
            if (_param.SaveEnable == true)
            {
                cbSaveEnable.IsChecked = true;
                if (_param.SaveMode == (int)SaveType.ORIGINAL_RESULT)
                    rbSaveOptionResultOrigin.IsChecked = true;
                else if (_param.SaveMode == (int)SaveType.RESULT)
                    rbSaveOptionResult.IsChecked = true;
                else if (_param.SaveMode == (int)SaveType.ORIGINAL)
                    rbSaveOptionOrigin.IsChecked = true;
                tbSavePath.Text = _param.SavePath;

                if (_param.SaveOption == (int)SaveOption.OK)
                    rbSaveOK.IsChecked = true;
                else if (_param.SaveOption == (int)SaveOption.NG)
                    rbSaveNG.IsChecked = true;
                else if (_param.SaveOption == (int)SaveOption.BOTH)
                    rbSaveBoth.IsChecked = true;

                udtbDeleteDay.Value = _param.DeleteDays;
            }
            else
            {
                tbSavePath.Text = string.Empty;
                udtbDeleteDay.Value = 0;
                rbSaveOptionResultOrigin.IsChecked = false;
                rbSaveOptionResult.IsChecked = false;
                rbSaveOptionOrigin.IsChecked = false;
                rbSaveOK.IsChecked = false;
                rbSaveNG.IsChecked = false;
                rbSaveBoth.IsChecked = false;
            }
        }

        private void NumberValidationTextBox(object sender, TextCompositionEventArgs e)
        {
            Regex regex = new Regex("[^0-9]+");
            e.Handled = regex.IsMatch(e.Text);
        }

        private void btnBrowser_Click(object sender, RoutedEventArgs e)
        {
            var savePaths = string.Empty;
            var dialog = new CommonOpenFileDialog
            {
                IsFolderPicker = true,
                Title = "Chọn thư mục lưu ảnh",
                Multiselect = false
            };
            WindowInteropHelper helper = new WindowInteropHelper(this);
            if (dialog.ShowDialog(helper.Handle) == CommonFileDialogResult.Ok)
            {
                savePaths = dialog.FileName;
                tbSavePath.Text = savePaths;
                tbSavePath.Focus();
                tbSavePath.CaretIndex = tbSavePath.Text.Length;
            }

            if (savePaths == string.Empty)
                return;
        }

        private void btnSave_Click(object sender, RoutedEventArgs e)
        {
            if (tbCam1Exposure.Text == string.Empty || tbCam2Exposure.Text == string.Empty || int.Parse(tbCam1Exposure.Text) <= 0 || int.Parse(tbCam2Exposure.Text) <= 0)
            {
                var error = new ErrorWindow("Please input exposure time!\rHãy nhập thời gian Exposure!");
                error.ShowDialog();
                return;
            }
            if (!IPAddress.TryParse(tbPlcIp.Text, out _))
            {
                var error = new ErrorWindow("Please input correct PLC IP!\rHãy nhập IP PLC chính xác!");
                error.ShowDialog();
                return;
            }
            if (cbbCam1Sn.Text == string.Empty || cbbCam2Sn.Text == string.Empty)
            {
                var error = new ErrorWindow("Please select camera serial number!\rHãy chọn Serial Number cho camera!");
                error.ShowDialog();
                return;
            }
            // Check connection
            if (!_cameraManager.CheckCamera1Connection(cbbCam1Sn.Text))
            {
                var error = new ErrorWindow($"Camera {cbbCam1Sn.Text} is not connected!\rKhông có kết nối camera {cbbCam1Sn.Text}!");
                error.ShowDialog();
                return;
            }
            if (!_cameraManager.CheckCamera2Connection(cbbCam2Sn.Text))
            {
                var error = new ErrorWindow($"Camera {cbbCam2Sn.Text} is not connected!\rKhông có kết nối camera {cbbCam2Sn.Text}!");
                error.ShowDialog();
                return;
            }
            if (!PlcController.ConnectPlc(_param.ApiUrlCom, tbPlcIp.Text, int.Parse(tbPlcPort.Text)))
            {
                var error = new ErrorWindow("No connection to PLC!\rKhông có kết nối PLC!");
                error.ShowDialog();
                return;
            }
            if (tbWhiteLightOnDelay.Text == string.Empty || tbWhiteLightOffDelay.Text == string.Empty || tbUvLightOnDelay.Text == string.Empty || tbUvLightOffDelay.Text == string.Empty)
            {
                var error = new ErrorWindow("Please input light delay time!\rHãy nhập thời gian delay đèn!");
                error.ShowDialog();
                return;
            }
            else
            {
                PlcController.DisConnectPlc(_param.ApiUrlCom);
            }
            // Save Settings
            _param.Cam1Sn = cbbCam1Sn.Text;
            _param.Cam1Exposure = int.Parse(tbCam1Exposure.Text);
            _param.Cam2Sn = cbbCam2Sn.Text;
            _param.Cam2Exposure = int.Parse(tbCam2Exposure.Text);
            _param.PlcIp = tbPlcIp.Text;
            _param.PlcPort = int.Parse(tbPlcPort.Text);
            _param.WaitWhiteLightOn = int.Parse(tbWhiteLightOnDelay.Text);
            _param.WaitWhiteLightOff = int.Parse(tbWhiteLightOffDelay.Text);
            _param.WaitUvLightOn = int.Parse(tbUvLightOnDelay.Text);
            _param.WaitUvLightOff = int.Parse(tbUvLightOffDelay.Text);

            // Saving Settings
            if (cbSaveEnable.IsChecked == true)
            {
                if (tbSavePath.Text == string.Empty)
                {
                    var error = new ErrorWindow("Please select save path!\rHãy chọn thư mục lưu ảnh!");
                    error.ShowDialog();
                    return;
                }
                if (rbSaveOptionOrigin.IsChecked == false && rbSaveOptionResult.IsChecked == false && rbSaveOptionResultOrigin.IsChecked == false)
                {
                    var error = new ErrorWindow("Please select save option!\rHãy chọn mode lưu ảnh!");
                    error.ShowDialog();
                    return;
                }

                // Save Settings
                int saveMode = 1;
                if (rbSaveOptionOrigin.IsChecked == true)
                    saveMode = (int)SaveType.ORIGINAL;
                else if (rbSaveOptionResult.IsChecked == true)
                    saveMode = (int)SaveType.RESULT;
                else if (rbSaveOptionResultOrigin.IsChecked == true)
                    saveMode = (int)SaveType.ORIGINAL_RESULT;
                _param.SaveEnable = true;
                _param.SavePath = tbSavePath.Text;
                _param.DeleteDays = int.Parse(udtbDeleteDay.Text);
                _param.SaveMode = saveMode;
                
                // Save Options
                int saveOption = 1;
                if (rbSaveOK.IsChecked == true)
                    saveOption = (int)SaveOption.OK;
                else if (rbSaveNG.IsChecked == true)
                    saveOption = (int)SaveOption.NG;
                else if (rbSaveBoth.IsChecked == true)
                    saveOption = (int)SaveOption.BOTH;
                _param.SaveOption = saveOption;
            }
            else
            {
                _param.SaveEnable = false;
                _param.SavePath = string.Empty;
                _param.SaveMode = 1;
                _param.SaveOption = 1;
            }
            _param.Save();
        }

        private void btnCancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
        private void btnCheckCamera1_Click(object sender, RoutedEventArgs e)
        {
            if (cbbCam1Sn.Text == string.Empty)
            {
                var error = new ErrorWindow("Please select camera serial number!\rHãy chọn mã Serial cho camera!");
                error.ShowDialog();
                return;
            }

            var waiting = new WaitingWindow("Checking camera connection...\rĐang kiểm tra kết nối camera...");
            var cam1Sn = cbbCam1Sn.Text;
            bool resConnection = false;
            new Task(() =>
            {
                resConnection = _cameraManager.CheckCamera1Connection(cam1Sn);
                waiting.KillMe = true;
            }).Start();

            waiting.WindowStartupLocation = WindowStartupLocation.CenterScreen;
            waiting.ShowDialog();
            if (resConnection)
            { 
                var info = new InformationWindow("Camera connection is OK!\rKết nối camera OK!");
                info.ShowDialog();
            }
            else
            {
                var error = new ErrorWindow("No camera connection!\rKhông có kết nối camera!");
                error.ShowDialog();
            }
        }
        private void btnCheckCamera2_Click(object sender, RoutedEventArgs e)
        {
            if (cbbCam2Sn.Text == string.Empty)
            {
                var error = new ErrorWindow("Please select camera serial number!\rHãy chọn mã Serial cho camera!");
                error.ShowDialog();
                return;
            }

            var waiting = new WaitingWindow("Checking camera connection...\rĐang kiểm tra kết nối camera...");
            var cam2Sn = cbbCam2Sn.Text;
            bool resConnection = false;
            new Task(() =>
            {
                resConnection = _cameraManager.CheckCamera2Connection(cam2Sn);
                waiting.KillMe = true;
            }).Start();

            waiting.WindowStartupLocation = WindowStartupLocation.CenterScreen;
            waiting.ShowDialog();
            if (resConnection)
            {
                var info = new InformationWindow("Camera connection is OK!\rKết nối camera OK!");
                info.ShowDialog();
            }
            else
            {
                var error = new ErrorWindow("No camera connection!\rKhông có kết nối camera!");
                error.ShowDialog();
            }
        }

        private void btnCheckPlc_Click(object sender, RoutedEventArgs e)
        {
            if (!IPAddress.TryParse(tbPlcIp.Text, out _))
            {
                var error = new ErrorWindow("Invalid IP address!\rIP không hợp lệ!");
                error.ShowDialog();
                return;
            }
            if (tbPlcPort.Text == string.Empty)
            {
                var error = new ErrorWindow("Please enter the port number!\rHãy nhập số port!");
                error.ShowDialog();
                return;
            }

            var waiting = new WaitingWindow("Checking PLC connection...\rĐang kiểm tra kết nối PLC...");
            bool result = false;
            var plcIp = tbPlcIp.Text;
            var plcPort = int.Parse(tbPlcPort.Text);
            new Task(() =>
            {
                result = PlcController.ConnectPlc(_param.ApiUrlCom, plcIp, plcPort);
                waiting.KillMe = true;
            }).Start();

            waiting.WindowStartupLocation = WindowStartupLocation.CenterScreen;
            waiting.ShowDialog();

            if (result)
            {
                new Task(() =>
                {
                    PlcController.DisConnectPlc(_param.ApiUrlCom);
                }).Start();          
                var info = new InformationWindow("PLC connection is OK!\rKết nối PLC OK!");
                info.ShowDialog();
            }
            else
            {
                var error = new ErrorWindow("No PLC connection!\rKhông có kết nối PLC!");
                error.ShowDialog();
            }
        }

        private void tbPlcIp_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            Regex regex = new Regex(@"^-?(?:\d+)?(?:\.\d*)?$");
            e.Handled = !regex.IsMatch(e.Text);
        }
    }
}
