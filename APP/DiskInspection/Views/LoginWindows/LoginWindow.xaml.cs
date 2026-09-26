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
using DiskInspection.Views.LoginWindows;

namespace DiskInspection.Views.LoginWindows
{
    /// <summary>
    /// Interaction logic for LoginWindow.xaml
    /// </summary>
    public partial class LoginWindow : Window
    {
        public UserType UserType { get; set; }
        public string UserName { get; set; }
        private UserElement[] Users { get; set; }
        private string UserPath = "user.ini";
        public LoginWindow()
        {
            InitializeComponent();
            UserType = UserType.DontKnow;
            txtUser.Text = "Admin";
            txtPassWord.Focus();
        }
        private void LoadUI()
        {
            EnableLogin();
            if (!File.Exists(UserPath))
            {
                InitUser();
            }
            Users = LoadListStrUser();
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
        private void CheckLoginAccount()
        {
            bool login_ok = false;
            string user = txtUser.Text;
            string password = txtPassWord.Password;
            for (int i = 0; i < Users.Length; i++)
            {
                if (user.ToUpper() == Users[i].User.ToUpper())
                {
                    if (Users[i].PassWord.ToUpper() == password.ToUpper())
                    {
                        UserType = Users[i].Type;
                        UserName = Users[i].User;
                        login_ok = true;
                        break;

                    }
                }
            }
            if (login_ok)
            {
                this.Close();
            }
            else
            {
                MessageBox.Show("User or password incorrect!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        private void btCancel_Click(object sender, RoutedEventArgs e)
        {
            this.Close();
        }

        private void txtUser_TextChanged(object sender, TextChangedEventArgs e)
        {
            EnableLogin();
        }

        private void txtPassWord_PasswordChanged(object sender, RoutedEventArgs e)
        {
            EnableLogin();
        }
        private void EnableLogin()
        {
            if (txtUser.Text != string.Empty && txtPassWord.Password != string.Empty)
            {
                btLogin.IsEnabled = true;
            }
            else
            {
                btLogin.IsEnabled = false;
            }
        }

        private void btLogin_Click(object sender, RoutedEventArgs e)
        {

            CheckLoginAccount();
        }

        private void Window_Initialized(object sender, EventArgs e)
        {
            LoadUI();
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (btLogin.IsEnabled)
            {
                if (e.Key == Key.Enter)
                {
                    CheckLoginAccount();
                }
            }
        }

        private void lbRegister_MouseClick(object sender, RoutedEventArgs e)
        {
            RegisterWindow window = new RegisterWindow();
            window.ShowDialog();
            LoadUI();
        }

        private void lbChangePass_MouseClick(object sender, RoutedEventArgs e)
        {
            ChangePassWindow window = new ChangePassWindow(this);
            window.ShowDialog();
        }

        internal void ReloadUsers()
        {
            Users = LoadListStrUser();
        }
    }
    public class UserElement
    {
        public string User { get; set; }
        public string PassWord { get; set; }
        public UserType Type { get; set; }
    }
    public enum UserType
    {
        Admin,
        Engineer,
        Worker,
        DontKnow,
    }
}
