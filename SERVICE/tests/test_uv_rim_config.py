"""Config parsing and behavioral wiring for the experimental UV-only counter."""
from dataclasses import fields
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.tools.uv_rim_counter import RimSettings, load_rim_settings, count_uv_rims, uv_masks


class UvRimConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'config.env'
        clean_env = {key: value for key, value in os.environ.items() if not key.startswith('UV_RIM_')}
        environment = patch.dict(os.environ, clean_env, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def write_config(self, values):
        self.path.write_text('\n'.join(f'UV_RIM_{key.upper()} = {value}' for key, value in values.items()), encoding='utf-8')

    def test_all_parameters_load_with_correct_types(self):
        values = dict(hue_low=99, hue_high=132, saturation_low=10, value_low=25,
                      strong_value=80, weak_contrast=3.5, strong_contrast=10.5,
                      min_coverage=.55, min_span=.75, min_strong_coverage=.4,
                      min_separation=7, core_saturation_max=40, core_value_min=130)
        self.write_config(values)
        actual = load_rim_settings(self.path)
        self.assertEqual(actual, RimSettings(**values))
        self.assertEqual(set(values), {field.name for field in fields(RimSettings)})

    def test_old_config_missing_keys_retains_original_defaults(self):
        self.path.write_text('UV_DISK_LOWER_THRESHOLD=107,35,40\n', encoding='utf-8')
        self.assertEqual(load_rim_settings(self.path), RimSettings())

    def test_explicit_config_is_independent_of_working_directory(self):
        self.write_config({'strong_value': 120})
        self.assertEqual(load_rim_settings(self.path).strong_value, 120)

    def test_service_uses_its_active_config_instead_of_source_file(self):
        from decouple import Config, RepositoryEnv
        self.write_config({'strong_value': 125, 'hue_high': 140})
        actual = load_rim_settings(config=Config(RepositoryEnv(str(self.path))))
        self.assertEqual(actual.strong_value, 125)
        self.assertEqual(actual.hue_high, 140)
        with self.assertRaisesRegex(ValueError, 'either config_path or config'):
            load_rim_settings(self.path, config=Config(RepositoryEnv(str(self.path))))

    def test_config_is_reloaded_after_edit(self):
        self.write_config({'strong_value': 120})
        self.assertEqual(load_rim_settings(self.path).strong_value, 120)
        self.write_config({'strong_value': 230})
        self.assertEqual(load_rim_settings(self.path).strong_value, 230)

    def test_environment_override_is_typed(self):
        self.write_config({'strong_value': 120})
        with patch.dict(os.environ, {'UV_RIM_STRONG_VALUE': '210'}):
            self.assertEqual(load_rim_settings(self.path).strong_value, 210)

    def test_bad_type_reports_parameter_name(self):
        self.write_config({'strong_value': 'abc'})
        with self.assertRaisesRegex(ValueError, 'UV_RIM_STRONG_VALUE'):
            load_rim_settings(self.path)

    def test_invalid_ranges_fail_explicitly(self):
        for values, key in [({'min_coverage': 60}, 'MIN_COVERAGE'),
                            ({'min_span': 0}, 'MIN_SPAN'),
                            ({'strong_contrast': 'nan'}, 'STRONG_CONTRAST'),
                            ({'weak_contrast': -1}, 'WEAK_CONTRAST'),
                            ({'hue_high': 180}, 'HUE_HIGH'),
                            ({'hue_low': 150, 'hue_high': 100}, 'HUE_LOW'),
                            ({'min_separation': 0}, 'MIN_SEPARATION'),
                            ({'core_value_min': 256}, 'CORE_VALUE_MIN')]:
            with self.subTest(values=values):
                self.write_config(values)
                with self.assertRaisesRegex(ValueError, 'UV_RIM_'+key):
                    load_rim_settings(self.path)

    def test_missing_file_does_not_silently_use_defaults(self):
        with self.assertRaises(FileNotFoundError):
            load_rim_settings(self.path)

    def test_default_counter_reads_config_but_explicit_settings_override_it(self):
        hsv = np.zeros((100, 320, 3), np.uint8)
        hsv[:, 80:83] = (113, 75, 180)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        with patch('src.tools.uv_rim_counter.load_rim_settings', return_value=RimSettings(strong_value=240)) as loader:
            actual, _ = count_uv_rims(image, 1)
            self.assertEqual(actual['count'], 0)
            loader.assert_called_once()
            explicit, _ = count_uv_rims(image, 1, RimSettings())
            self.assertEqual(explicit['count'], 1)

    def test_white_core_threshold_is_configurable(self):
        hsv = np.zeros((100, 320, 3), np.uint8)
        hsv[:, 60:67] = (113, 75, 150)
        hsv[:, 62:65] = (0, 0, 180)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        before, _, _ = uv_masks(image, RimSettings())
        after, _, _ = uv_masks(image, RimSettings(core_value_min=220))
        self.assertTrue(before[:, 63].all())
        self.assertFalse(after[:, 63].any())


if __name__ == '__main__':
    unittest.main()
