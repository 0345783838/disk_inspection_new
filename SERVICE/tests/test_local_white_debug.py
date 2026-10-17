"""Source-contract guards for the WPF local-image entry point, alongside C# build/flow tests."""
from pathlib import Path
import unittest


class LocalWhiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Path(__file__).resolve().parents[2] / 'APP' / 'DiskInspection'
        cls.controller = (app / 'Controllers' / 'MainController.cs').read_text(encoding='utf-8-sig')
        cls.window = (app / 'Views' / 'MainWindow.xaml.cs').read_text(encoding='utf-8-sig')
        cls.local = cls.controller.split('private async Task RunLocalWhiteAsync(', 1)[1].split('private async Task<(CameraInspectionResult', 1)[0]

    def test_same_normalizer_and_white_api_without_hardware(self):
        self.assertIn('_frameNormalizer.Normalize(raw, cameraId)', self.local)
        self.assertIn('InspectWhiteFrameAsync(normalized, token)', self.local)
        for forbidden in ('PlcController.', 'RunUVAIAsync(', 'TriggerAndGetFrame(', 'QueueImages(', 'UpdateStatistics(', 'UpdateInspectionStatus('):
            # Exclude the explanatory camera comment.
            code = '\n'.join(line for line in self.local.splitlines() if not line.strip().startswith('//'))
            self.assertNotIn(forbidden, code)

    def test_local_joins_shutdown_and_rejects_running_cycle(self):
        entry = self.controller.split('public Task InspectLocalWhiteAsync(', 1)[1].split('private async Task RunLocalWhiteAsync(', 1)[0]
        self.assertIn('_isRunning || !_inspectionLoop.IsCompleted', entry)
        self.assertIn('_inspectionLoop = completion.Task', entry)
        self.assertIn('finally { completion.TrySetResult(true); }', entry)
        self.assertIn('if (_hardwareMayBeActive)', self.controller)

    def test_cannot_present_uv_or_aggregate_as_passed(self):
        reset = self.window.split('internal void PrepareLocalWhiteUI(', 1)[1].split('private void Window_Loaded', 1)[0]
        self.assertIn('InspectionStatus = InspectionStatusCam1 = InspectionStatusCam2 = (int)StatusState.Unknown', reset)
        self.assertIn('lbUvDiskCountCam1.Content = lbUvDiskCountCam2.Content = "Not tested"', reset)
        self.assertIn('UV / PLC NOT TESTED', reset)
        self.assertIn('ClearCycleImages(updateUi: false)', self.local)


if __name__ == '__main__':
    unittest.main()
