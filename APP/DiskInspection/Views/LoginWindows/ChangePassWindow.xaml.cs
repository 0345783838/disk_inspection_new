using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
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

namespace DiskInspection.Views.LoginWindows
{
    /// <summary>
    /// Interaction logic for ChangePassWindow.xaml
    /// </summary>
    public partial class ChangePassWindow : Window
    {
        private string UserPath = "user.ini";
        public ObservableCollection<UserElement> Users { get; set; } = new ObservableCollection<UserElement>();
        public UserElement SelectedUser { get; set; }
        LoginWindow _loginWindow;
        public ChangePassWindow(LoginWindow window)
        {
            InitializeComponent();
            Init();
            DataContext = this;
            _loginWindow = window;
        }

        private void Init()
        {
            LoadListStrUser();
        }
        private void LoadListStrUser()
        {
            string str = File.ReadAllText(UserPath);
            byte[] bytes = Convert.FromBase64String(str);
            string userStr = ASCIIEncoding.ASCII.GetString(bytes);
            string[] arUserStr = userStr.Split(',');
            GetListUser(arUserStr);
        }
        private void GetListUser(string[] arUserStr)
        {
            for (int i = 0; i < arUserStr.Length; i++)
            {
                var user = new UserElement();
                string[] info = arUserStr[i].Split('_');
                user.User = info[0];
                user.PassWord = info[2];
                switch (info[1])
                {
                    case "Admin":
                        user.Type = UserType.Admin;
                        break;
                    case "Engineer":
                        user.Type = UserType.Engineer;
                        break;
                    case "Worker":
                        user.Type = UserType.Worker;
                        break;
                    case "DontKnow":
                        user.Type = UserType.DontKnow;
                        break;
                    default:
                        break;
                }
                Users.Add(user);
            }
        }
        private void EnableChange()
        {
            if (txtNewPass.Password != string.Empty && txtCurPassWord.Password != string.Empty && txtAdminCode.Password != string.Empty && txtConfirmPass.Password != string.Empty && cbbUser.SelectedIndex != -1)
            {
                btChangePass.IsEnabled = true;
            }
            else
            {
                btChangePass.IsEnabled = false;
            }
        }

        private void txtPassWord_PasswordChanged(object sender, RoutedEventArgs e)
        {
            EnableChange();
        }

        private void btChangePass_Click(object sender, RoutedEventArgs e)
        {
            CheckChangingCondition();
        }

        private void CheckChangingCondition()
        {
            // Check if current password is correct
            var currentUser = Users.Where(x => x.User == SelectedUser.User).FirstOrDefault();
            if (currentUser.PassWord != txtCurPassWord.Password)
            {
                MessageBox.Show("Current password is incorrect!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (txtNewPass.Password != txtConfirmPass.Password)
            {
                MessageBox.Show("New password is not matched with confirm password!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (txtNewPass.Password == txtCurPassWord.Password)
            {
                MessageBox.Show("New password is same with current password!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }
            if (txtAdminCode.Password != "spismt")
            {
                MessageBox.Show("Admin code is incorrect!", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // Change password
            string account = "";
            currentUser.PassWord = txtNewPass.Password;
            for (int i = 0; i < Users.Count; i++)
            {
                account += $"{Users[i].User}_{Users[i].Type.ToString()}_{Users[i].PassWord},";
            }
            if (account.EndsWith(","))
            {
                account = account.Substring(0, account.Length - 1);
            }
            byte[] bytes = ASCIIEncoding.ASCII.GetBytes(account);
            string base64 = Convert.ToBase64String(bytes);
            File.WriteAllText(UserPath, base64);

            MessageBox.Show("Change password successfully!", "Success", MessageBoxButton.OK, MessageBoxImage.Information);
            this.Close();
        }

        private void btCancel_Click(object sender, RoutedEventArgs e)
        {

        }

        private void cbbUser_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            EnableChange();
        }

        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                btChangePass_Click(null, null);
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            _loginWindow.ReloadUsers();
        }
    }
}
