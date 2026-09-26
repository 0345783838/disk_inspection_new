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
using System.IO;

namespace DiskInspection.Views.LoginWindows
{
    /// <summary>
    /// Interaction logic for RegisterWindow.xaml
    /// </summary>
    public partial class RegisterWindow : Window
    {
        public UserType UserType { get; set; }
        public string UserName { get; set; }
        private UserElement[] Users { get; set; }
        private string UserPath = "user.ini";
        public RegisterWindow()
        {
            InitializeComponent();
            LoadUI();
        }
        private void LoadUI()
        {
            EnableLogin();
            if (!File.Exists(UserPath))
            {
                InitUser();
            }
            Users = LoadListStrUser();
            cbUserType.Items.Add("Worker");
            cbUserType.Items.Add("Engineer");
            cbUserType.SelectedIndex = 0;
        }
        private void EnableLogin()
        {
            if (txtUser.Text != string.Empty && txtPassWord.Password != string.Empty && txtAdminCode.Password != string.Empty && txtRePass.Password != string.Empty)
            {
                btRegister.IsEnabled = true;
            }
            else
            {
                btRegister.IsEnabled = false;
            }
        }

        private UserElement[] LoadListStrUser()
        {
            string str = File.ReadAllText(UserPath);
            byte[] bytes = Convert.FromBase64String(str);
            string userStr = ASCIIEncoding.ASCII.GetString(bytes);
            string[] arUserStr = userStr.Split(',');
            return GetListUser(arUserStr);
        }
        private UserElement[] GetListUser(string[] arUserStr)
        {
            UserElement[] user = new UserElement[arUserStr.Length];
            for (int i = 0; i < arUserStr.Length; i++)
            {
                user[i] = new UserElement();
                string[] info = arUserStr[i].Split('_');
                user[i].User = info[0];
                user[i].PassWord = info[2];
                switch (info[1])
                {
                    case "Admin":
                        user[i].Type = UserType.Admin;
                        break;
                    case "Engineer":
                        user[i].Type = UserType.Engineer;
                        break;
                    case "Worker":
                        user[i].Type = UserType.Worker;
                        break;
                    case "DontKnow":
                        user[i].Type = UserType.DontKnow;
                        break;
                    default:
                        break;
                }
            }
            return user;
        }
        private void InitUser()
        {
            string adminInfo = "admin_Admin_disk";
            byte[] bytes = ASCIIEncoding.ASCII.GetBytes(adminInfo);
            string base64 = Convert.ToBase64String(bytes);
            File.WriteAllText(UserPath, base64);

        }
        private void Window_Initialized(object sender, EventArgs e)
        {

        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                CheckAccount();
            }
        }
        private void CheckAccount()
        {
            bool login_ok = false;
            string user = txtUser.Text;
            string password = txtPassWord.Password;
            string rePassword = txtRePass.Password;
            string adminCode = txtAdminCode.Password;
            UserType userType = UserType.DontKnow;
            string userTypeStr = cbUserType.SelectedItem.ToString();
            switch (userTypeStr)
            {
                case "Admin":
                    userType = UserType.Admin;
                    break;
                case "Engineer":
                    userType = UserType.Engineer;
                    break;
                case "Worker":
                    userType = UserType.Worker;
                    break;
                case "DontKnow":
                    userType = UserType.DontKnow;
                    break;
                default:
                    break;
            }
            if (string.IsNullOrEmpty(user))
            {
                MessageBox.Show("Please insert user name...", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (string.IsNullOrEmpty(password))
            {
                MessageBox.Show("Please insert password...", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (string.IsNullOrEmpty(rePassword))
            {
                MessageBox.Show("Please insert confirm password...", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (string.IsNullOrEmpty(adminCode))
            {
                MessageBox.Show("Please insert admin code...", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (password != rePassword)
            {
                MessageBox.Show("Password and confirm password do not match!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (adminCode != "disk123456")
            {
                MessageBox.Show("Admin code is incorrect!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            for (int i = 0; i < Users.Length; i++)
            {
                if (user.ToUpper() == Users[i].User.ToUpper())
                {
                    login_ok = true;
                }
            }
            if (login_ok)
            {
                MessageBox.Show("This user is existed!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            else
            {
                string account = "";
                for (int i = 0; i < this.Users.Length; i++)
                {
                    account += $"{this.Users[i].User}_{this.Users[i].Type.ToString()}_{this.Users[i].PassWord},";
                }
                account += $"{user}_{userType}_{password}";
                byte[] bytes = ASCIIEncoding.ASCII.GetBytes(account);
                string base64 = Convert.ToBase64String(bytes);
                File.WriteAllText(UserPath, base64);
                MessageBox.Show("Successful account registration!", "INFO", MessageBoxButton.OK, MessageBoxImage.Information);
                this.Close();
            }
        }
        private void txtAdminCode_PasswordChanged(object sender, RoutedEventArgs e)
        {
            EnableLogin();
        }

        private void txtPassWord_PasswordChanged(object sender, RoutedEventArgs e)
        {
            EnableLogin();
        }

        private void txtUser_TextChanged(object sender, TextChangedEventArgs e)
        {
            EnableLogin();
        }

        private void btCancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }
        private void btRegister_Click(object sender, RoutedEventArgs e)
        {
            CheckAccount();
        }
    }
}
