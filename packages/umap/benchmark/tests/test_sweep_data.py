import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.umap import sweep_data  # noqa: E402
from benchmark.umap.sweep_config import DATASETS  # noqa: E402


class DatasetIndexTest(unittest.TestCase):
    def test_index_lists_every_sweep_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source in {spec.source for spec in DATASETS}:
                folder = root / source
                folder.mkdir()
                np.zeros((3, 4), dtype="<f4").tofile(folder / "data.bin")
                np.zeros(3, dtype=np.uint8).tofile(folder / "labels.bin")
                (folder / "meta.json").write_text(json.dumps({"n_points": 3, "dim": 4}))
            index = json.loads(sweep_data.write_index(root).read_text())
        names = [d["name"] for d in index["datasets"]]
        self.assertEqual(names, [spec.name for spec in DATASETS])
        first = index["datasets"][0]
        self.assertEqual(
            (
                first["folder"],
                first["n"],
                first["dim"],
                first["labels"],
                first["bytes"],
            ),
            (DATASETS[0].source, 3, 4, True, 48),
        )


if __name__ == "__main__":
    unittest.main()
