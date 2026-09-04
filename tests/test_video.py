import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from modules import private_logger
from modules import video_models
from modules import video_runtime
from modules import video_worker


class _ImageSize:
    def __init__(self, width, height):
        self.size = (width, height)


class TestVideoRuntime(unittest.TestCase):
    def test_rounded_size_preserves_aspect_and_multiple(self):
        width, height = video_runtime.rounded_size(_ImageSize(1600, 900), 832 * 480)
        self.assertEqual(0, width % 32)
        self.assertEqual(0, height % 32)
        self.assertLessEqual(width * height, 832 * 480 + 32 * max(width, height))
        self.assertAlmostEqual(16 / 9, width / height, delta=0.1)

    def test_h3_frames_follow_vae_grid(self):
        self.assertEqual(124, video_runtime.h3_num_frames(5))
        self.assertLessEqual(video_runtime.h3_num_frames(15), 362)
        self.assertEqual(0, (video_runtime.h3_num_frames(10) - 5) % 17)

    def test_low_vram_wan_profile_is_reduced(self):
        low = video_runtime.wan_profile("8 GB")
        high = video_runtime.wan_profile("16 GB+")
        self.assertLess(low["max_area"], high["max_area"])
        self.assertLess(low["num_frames"], high["num_frames"])
        self.assertTrue(low["group_offload"])
        self.assertEqual(832 * 480, video_runtime.requested_max_area("Medium (480p)"))


class TestVideoModels(unittest.TestCase):
    def test_auto_profile_selection(self):
        with mock.patch.object(video_models, "hardware_info", return_value={"vram_gb": 8}):
            self.assertEqual("8 GB", video_models.resolve_hardware_profile())
        with mock.patch.object(video_models, "hardware_info", return_value={"vram_gb": 12}):
            self.assertEqual("12 GB", video_models.resolve_hardware_profile())
        with mock.patch.object(video_models, "hardware_info", return_value={"vram_gb": 16}):
            self.assertEqual("16 GB+", video_models.resolve_hardware_profile())

    def test_h3_requires_authorization(self):
        info = {
            "cuda": True,
            "device_name": "GPU",
            "vram_gb": 16,
            "ram_gb": 128,
            "free_disk_gb": 500,
        }
        with mock.patch.object(video_models, "hardware_info", return_value=info), \
                mock.patch.object(video_models, "h3_authorized", return_value=False):
            allowed, message = video_models.preflight("h3")
        self.assertFalse(allowed)
        self.assertIn("authorization", message)

    def test_h3_rejects_insufficient_host_memory(self):
        info = {
            "cuda": True,
            "device_name": "GPU",
            "vram_gb": 16,
            "ram_gb": 32,
            "free_disk_gb": 500,
        }
        with mock.patch.object(video_models, "hardware_info", return_value=info), \
                mock.patch.object(video_models, "h3_authorized", return_value=True):
            allowed, message = video_models.preflight("h3")
        self.assertFalse(allowed)
        self.assertIn("system RAM", message)

    def test_model_readiness_checks_required_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(video_models.modules.config, "path_video_models", temp_dir):
            model = video_models.get_model("wan")
            for required in model.required_paths:
                path = model.path / required
                if "." in Path(required).name:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("{}", encoding="utf-8")
                else:
                    path.mkdir(parents=True, exist_ok=True)
            (model.path / ".fooocus-complete").write_text(model.repo_id, encoding="utf-8")
            self.assertTrue(video_models.model_is_ready("wan"))


class TestVideoWorkerAndLogging(unittest.TestCase):
    def test_task_cancel_terminates_runtime(self):
        task = video_worker.VideoTask(image="image.png", prompt="move")
        task.process = mock.Mock()
        task.process.poll.return_value = None
        task.cancel()
        self.assertTrue(task.cancel_requested)
        task.process.terminate.assert_called_once()

    def test_task_request_serialization(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
                mock.patch.object(video_worker.modules.config, "temp_path", temp_dir), \
                mock.patch.object(video_worker.modules.config, "path_outputs", temp_dir), \
                mock.patch.object(video_worker.args_manager.args, "disable_image_log", False):
            task = video_worker.VideoTask(
                image=np.zeros((64, 64, 3), dtype=np.uint8),
                prompt="camera pans right",
                resolution="Low (512px)",
                seed=7,
            )
            input_path, request_path, output_path = video_worker._create_request(task, "12 GB")
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual("camera pans right", request["prompt"])
            self.assertEqual(7, request["seed"])
            self.assertEqual("12 GB", request["profile"])
            self.assertEqual("Low (512px)", request["resolution"])
            self.assertTrue(input_path.is_file())
            video_worker._remove_files(input_path, request_path, output_path)

    def test_runtime_progress_message(self):
        task = video_worker.VideoTask(image="image.png", prompt="move")
        result = {}
        video_worker._handle_runtime_message(
            task,
            json.dumps({"event": "progress", "percent": 42, "message": "Working"}),
            result,
        )
        self.assertEqual(["progress", (42, "Working")], task.yields[0])

    def test_video_metadata_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "sample.mp4"
            video_path.write_bytes(b"video")
            with mock.patch.object(private_logger.args_manager.args, "disable_image_log", True):
                private_logger.log_video(str(video_path), {"Seed": 123})
            metadata = json.loads(video_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(123, metadata["Seed"])


if __name__ == "__main__":
    unittest.main()
