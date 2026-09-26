using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiskInspection.Utils
{
    public class EnvReader
    {
        private readonly string _envPath;
        private readonly List<string> _lines;
        private readonly Dictionary<string, string> _values;

        public IDictionary<string, string> Values
        {
            get { return _values; }
        }

        public EnvReader(string envPath)
        {
            _envPath = envPath;
            _lines = new List<string>();
            _values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

            Load();
        }

        public EnvReader() : this(".env")
        {
        }

        // ================= LOAD =================
        private void Load()
        {
            _lines.Clear();
            _values.Clear();

            if (!File.Exists(_envPath))
                return;

            _lines.AddRange(File.ReadAllLines(_envPath));

            foreach (string line in _lines)
            {
                string trimmed = line.Trim();

                if (string.IsNullOrEmpty(trimmed))
                    continue;

                if (trimmed.StartsWith("#"))
                    continue;

                int idx = trimmed.IndexOf('=');
                if (idx <= 0)
                    continue;

                string key = trimmed.Substring(0, idx).Trim();
                string value = trimmed.Substring(idx + 1).Trim();

                _values[key] = value;
            }
        }

        // ================= GET =================
        public string Get(string key, string defaultValue)
        {
            string value;
            if (_values.TryGetValue(key, out value))
                return value;

            return defaultValue;
        }

        public string Get(string key)
        {
            return Get(key, null);
        }

        public int GetInt(string key, int defaultValue)
        {
            string v = Get(key);
            int result;
            return int.TryParse(v, out result) ? result : defaultValue;
        }

        public float GetFloat(string key, float defaultValue)
        {
            string v = Get(key);
            float result;
            return float.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out result) ? result : defaultValue;
        }

        public bool GetBool(string key, bool defaultValue)
        {
            string v = Get(key);
            bool result;
            return bool.TryParse(v, out result) ? result : defaultValue;
        }
        public string[] GetArray(string key, char separator)
        {
            string value = Get(key);
            if (string.IsNullOrEmpty(value))
                return new string[0];

            return value
                .Split(new char[] { separator }, StringSplitOptions.RemoveEmptyEntries)
                .Select(v => v.Trim())
                .ToArray();
        }

        public string[] GetArray(string key)
        {
            return GetArray(key, ',');
        }
        public List<int> GetIntArray(string key, char separator)
        {
            string[] parts = GetArray(key, separator);
            List<int> result = new List<int>();

            foreach (string p in parts)
            {
                int v;
                if (int.TryParse(p, out v))
                    result.Add(v);
            }

            return result;
        }

        public List<int> GetIntArray(string key)
        {
            return GetIntArray(key, ',');
        }
        public float[] GetFloatArray(string key, char separator)
        {
            string[] parts = GetArray(key, separator);
            List<float> result = new List<float>();

            foreach (string p in parts)
            {
                float v;
                if (float.TryParse(p, NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                    result.Add(v);
            }

            return result.ToArray();
        }

        public float[] GetFloatArray(string key)
        {
            return GetFloatArray(key, ',');
        }
        public bool[] GetBoolArray(string key, char separator)
        {
            string[] parts = GetArray(key, separator);
            List<bool> result = new List<bool>();

            foreach (string p in parts)
            {
                bool v;
                if (bool.TryParse(p, out v))
                    result.Add(v);
            }

            return result.ToArray();
        }

        public bool[] GetBoolArray(string key)
        {
            return GetBoolArray(key, ',');
        }
        // ================= SET =================
        public void Set(string key, string value)
        {
            _values[key] = value;

            bool updated = false;

            for (int i = 0; i < _lines.Count; i++)
            {
                string line = _lines[i].TrimStart();
                if (line.Split(new[] { '=' }, 2)[0].Trim().Equals(key, StringComparison.OrdinalIgnoreCase))
                {
                    _lines[i] = key + " = " + value;
                    updated = true;
                    break;
                }
            }

            if (!updated)
                _lines.Add(key + "=" + value);
        }

        public void Remove(string key)
        {
            if (_values.ContainsKey(key))
                _values.Remove(key);

            _lines.RemoveAll(l => l.Split(new[] { '=' }, 2)[0].Trim().Equals(key, StringComparison.OrdinalIgnoreCase));
        }

        // ================= SAVE =================
        public void Save()
        {
            File.WriteAllLines(_envPath, _lines.ToArray());
        }
        public EnvReader Clone()
        {
            return new EnvReader(_envPath);
        }

        // ================= APPLY =================
        public void ApplyToEnvironment()
        {
            foreach (KeyValuePair<string, string> kv in _values)
            {
                Environment.SetEnvironmentVariable(kv.Key, kv.Value);
            }
        }
    }
}
