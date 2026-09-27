"""Small numerical checks with a deterministic E5 stub; no model downloads."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from multabench.baselines.preprocessing import text_embeddings as embeddings
from multabench.e5.constants import format_e5_passage, TF_IDF


class TargetGuidedFusionTest(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({"Title": ["first", "second", "third"],
                               "Review Text": ["good", "bad", "okay"], "Age": [20, 30, 40]})
        self.target = np.linspace(-1, 1, 384, dtype=np.float32)
        self.model = Mock()
        self.calls = []

    def encode(self, texts, col_name, **kwargs):
        self.calls.append((texts, col_name))
        if col_name is None:
            self.assertEqual(texts, ["Recommended IND"])
            return self.target[None, :]
        return np.arange(len(texts) * 384, dtype=np.float32).reshape(len(texts), 384) / 1000

    def test_fusion_dimensions_values_and_inference(self):
        for columns in ({"Title"}, {"Title", "Review Text"}):
            with self.subTest(columns=columns), \
                    patch.object(embeddings, 'get_vanilla_e5', return_value=(self.model, Mock())), \
                    patch.object(embeddings, 'encode_texts_with_e5', side_effect=self.encode), \
                    patch.object(embeddings, 'PCA', side_effect=AssertionError('PCA must not run')):
                self.calls.clear()
                x = self.x[[*sorted(columns), "Age"]]
                encoders = embeddings.fit_text_encoders(
                    x=x, text_features=columns, device="cpu", y=object(),
                    target_guided_fusion=True, target_column_name="Recommended IND",
                )
                self.assertEqual(len(self.calls), 1)  # Metadata encoded once, without accessing y.
                self.model.requires_grad_.assert_called_with(False)
                for frame in (x, x.iloc[:1]):
                    result = embeddings.transform_text_features(frame, encoders, "cpu")
                    self.assertEqual(result.shape, (len(frame), 768 * len(columns) + 1))
                    pd.testing.assert_series_equal(result["Age"], frame["Age"])
                    for col in columns:
                        original = result[[f"{col}_e5_{i}" for i in range(384)]].to_numpy()
                        product = result[[f"{col}_target_product_{i}" for i in range(384)]].to_numpy()
                        expected = self.encode(frame[col].tolist(), col)
                        np.testing.assert_array_equal(original, expected)
                        np.testing.assert_array_equal(product, expected * self.target)
                self.assertEqual(sum(col is None for _, col in self.calls), 1)

    def test_off_keeps_identity_and_pca_paths(self):
        with patch.object(embeddings, 'get_vanilla_e5', return_value=(self.model, Mock())), \
                patch.object(embeddings, 'encode_texts_with_e5', side_effect=self.encode):
            for no_pca, dimension in ((True, 384), (False, 2)):
                encoders = embeddings.fit_text_encoders(
                    x=self.x, text_features={"Title"}, device="cpu",
                    no_pca=no_pca, pca_components=2,
                )
                result = embeddings.transform_text_features(self.x[["Title", "Age"]], encoders, "cpu")
                self.assertEqual(result.shape, (3, dimension + 1))
                self.assertIn("Title_txt_pca_0", result)
            self.assertFalse(any(col is None for _, col in self.calls))
            self.model.requires_grad_.assert_not_called()

    def test_invalid_modes_and_metadata(self):
        for kwargs in ({"tune_e5": True}, {"e5_model_name": TF_IDF}, {"target_column_name": None}):
            args = dict(x=self.x, text_features={"Title"}, device="cpu",
                        target_guided_fusion=True, target_column_name="Recommended IND")
            args.update(kwargs)
            with self.assertRaises(ValueError):
                embeddings.fit_text_encoders(**args)

    def test_target_passage_is_raw_metadata(self):
        self.assertEqual(format_e5_passage(None, "Recommended IND"), "passage: Recommended IND")
        self.assertEqual(format_e5_passage("Title", "Good"), "passage: Title: Good")


if __name__ == '__main__':
    unittest.main()
