import unittest
import numpy as np
from src.tools.slot_geometry import complete_slot_rows, filter_off_grid_extras
from src.inference.yolo_detect.onnx_detector import OnnxDetector


def tray():
    i = np.arange(25,dtype=np.float32)
    rows = []
    for r in range(3):
        x = (100+60*i+6*r)/(1+.003*i)
        y = (100+700*r)/(1+.003*i)
        rows.append(np.column_stack((x-30,y-50,x+30,y+50)))
    return rows


class SlotTests(unittest.TestCase):
    def test_off_grid_endpoint_extra_is_removed_without_changing_real_points(self):
        for target in range(3):
            truth = tray()
            rows = [row.copy() for row in truth]
            extra = rows[target][-1].copy()
            extra[[0, 2]] += 40
            rows[target] = np.vstack((rows[target], extra))
            actual, removed = filter_off_grid_extras(rows)
            self.assertEqual(len(removed), 1)
            self.assertEqual(removed[0]['row'], target + 1)
            for a, b in zip(actual, truth):
                np.testing.assert_array_equal(a, b)

    def test_extra_filter_rejects_ambiguous_or_insufficient_anchors(self):
        for mode in ('duplicate', 'two_overfull', 'short_reference', 'too_many', 'missing_and_extra'):
            rows = tray()
            extra = rows[1][-1].copy()
            extra[[0, 2]] += 40
            rows[1] = np.vstack((rows[1], extra))
            if mode == 'duplicate':
                rows[1][-1] = rows[1][12]
            elif mode == 'two_overfull':
                rows[0] = np.vstack((rows[0], rows[0][-1]))
            elif mode == 'short_reference':
                rows[0] = rows[0][1:]
            elif mode == 'too_many':
                rows[1] = np.vstack((rows[1], np.repeat(extra[None], 5, axis=0)))
            else:
                rows[1] = np.vstack((np.delete(rows[1], 12, axis=0), extra))
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                filter_off_grid_extras(rows)

    def test_extra_filter_leaves_complete_and_short_rows_unchanged(self):
        rows = tray()
        rows[0] = rows[0][1:]
        actual, removed = filter_off_grid_extras(rows)
        self.assertEqual(removed, [])
        for a, b in zip(actual, rows):
            np.testing.assert_array_equal(a, b)

    def test_complete_detections_are_unchanged(self):
        rows = tray()
        actual,meta = complete_slot_rows(rows,(2500,2000,3))
        self.assertEqual(meta,[])
        for a,b in zip(actual,rows):
            np.testing.assert_array_equal(a,b)

    def test_interior_and_endpoint_missing_slots(self):
        for r in range(3):
            for missing in ([12],[0],[24],[0,1,2,23,24],[4,8,12,16,20],list(range(5))):
                with self.subTest(row=r,missing=missing):
                    truth=tray(); rows=[row.copy() for row in truth]
                    rows[r]=np.delete(rows[r],missing,axis=0)
                    actual,meta=complete_slot_rows(rows,(2500,2000,3))
                    self.assertEqual([len(row) for row in actual],[25,25,25])
                    self.assertEqual([m['slot'] for m in meta],[i+1 for i in missing])
                    self.assertTrue(all(m['detector_confidence'] is None for m in meta))
                    np.testing.assert_allclose(actual[r][missing],truth[r][missing],atol=1.1)

    def test_rejects_unsafe_completion(self):
        for mode in ('six_missing','two_short','duplicate_reference','ambiguous_phase','outside_image'):
            rows=tray(); shape=(2500,2000,3)
            if mode=='six_missing': rows[0]=rows[0][6:]
            elif mode=='two_short': rows[0]=rows[0][1:]; rows[1]=rows[1][1:]
            elif mode=='duplicate_reference': rows[0]=rows[0][1:]; rows[1][12]=rows[1][11]
            elif mode=='ambiguous_phase': rows[0]=rows[0][1:]; rows[0][:,[0,2]]+=25
            else: rows[0]=rows[0][:-1]; shape=(2500,1400,3)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                complete_slot_rows(rows,shape)


class DetectorRoiTests(unittest.TestCase):
    def test_roi_only_changes_detector_input_and_offsets_boxes(self):
        model=OnnxDetector.__new__(OnnxDetector)
        model.input_roi=(3,4,8,9); model.canonical_size=(20,18)
        image=np.arange(18*20*3,dtype=np.uint16).reshape(18,20,3)
        view,offset=model.detector_image(image)
        np.testing.assert_array_equal(view,image[4:13,3:11])
        np.testing.assert_array_equal(model.offset_boxes([[1,2,5,6]],offset),[[4,6,8,10]])
        self.assertEqual(image.shape,(18,20,3))

    def test_wrong_frame_or_roi_rejected(self):
        model=OnnxDetector.__new__(OnnxDetector)
        model.input_roi=(0,0,20,20); model.canonical_size=(20,18)
        with self.assertRaises(ValueError): model.detector_image(np.zeros((18,20,3)))
        model.input_roi=None
        with self.assertRaises(ValueError): model.detector_image(np.zeros((20,18,3)))

