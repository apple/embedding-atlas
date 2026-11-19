import base64
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.umap import report_html  # noqa: E402


class EncodeTest(unittest.TestCase):
    def test_encode_run_points_layout(self):
        embedding = np.array([[0.0, 10.0, 99.0], [1.0, 20.0, 99.0]], dtype=np.float32)
        encoded = report_html.encode_run_points(embedding)
        codes = np.frombuffer(base64.b64decode(encoded["q"]), dtype="<u2")
        self.assertEqual(list(codes), [0, 65535, 0, 65535])
        self.assertEqual(encoded["r"], [0.0, 1.0, 10.0, 20.0])


def make_sweep(directory, n=400):
    (directory / "datasets/toy").mkdir(parents=True)
    (directory / "embeddings/toy").mkdir(parents=True)
    (np.arange(n) % 4).astype(np.uint8).tofile(directory / "datasets/toy/labels.bin")
    np.random.RandomState(0).normal(size=(n, 2)).astype("<f4").tofile(
        directory / "embeddings/toy/run.bin"
    )
    run = {field: None for field in report_html.RUN_FIELDS}
    run.update(
        dataset="toy",
        status="ok",
        reason="uses <b> & </script>",
        file="embeddings/toy/run.bin",
        n_components=2,
    )
    return {
        "main_seed": 42,
        "datasets": [{"name": "toy", "n": n, "labels_file": "datasets/toy/labels.bin"}],
        "axes": [],
        "variants": [],
        "runs": [run, {**run, "status": "error", "file": None}],
    }


class GeneratedHtmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp())
        cls.manifest = make_sweep(cls.directory)
        cls.output = cls.directory / "report.html"
        report_html.write_report_html(
            cls.manifest, points=100, sweep_dir=cls.directory, output=cls.output
        )
        cls.html = cls.output.read_text()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory)

    def data(self):
        match = re.search(
            r'<script type="application/json" id="rp-data">(.*?)</script>',
            self.html,
            re.S,
        )
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def test_is_self_contained(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertNotRegex(self.html, r"<script[^>]+src=")
        self.assertNotRegex(self.html, r"<link[^>]+href=")
        self.assertNotIn(report_html.LIB_PLACEHOLDER, self.html)
        self.assertNotIn(report_html.DATA_PLACEHOLDER, self.html)

    def test_embeds_the_data(self):
        data = self.data()
        self.assertEqual(len(data["runs"]), 2)
        self.assertEqual(data["datasets"][0]["count"], 100)
        self.assertEqual([("q" in run) for run in data["runs"]], [True, False])
        self.assertEqual(data["runs"][0]["reason"], "uses <b> & </script>")

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_page_scripts_are_valid_javascript(self):
        scripts = re.findall(r"<script>(.*?)</script>", self.html, re.S)
        self.assertEqual(len(scripts), 2)
        for index, script in enumerate(scripts):
            path = self.directory / f"script{index}.js"
            path.write_text(script)
            result = subprocess.run(
                ["node", "--check", str(path)], capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
