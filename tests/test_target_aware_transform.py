"""Deterministic geometry and pipeline tests, without downloads or training."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from multabench.baselines.preprocessing import text_embeddings as embeddings
from multabench.baselines.lgbm import LightGBM
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import TF_IDF


class TargetAwareTest(unittest.TestCase):
    def test_geometry_normalization_and_no_mutation(self):
        x = np.array([[3., 4.], [-3., 4.], [0., 2.]])
        target = np.array([2., 0.])
        before = x.copy()
        transform = embeddings._TargetAwareTransform(target)
        z = transform.transform(x)
        self.assertEqual(z.shape, (3, 7))
        np.testing.assert_array_equal(z[:, :2], before)
        np.testing.assert_array_equal(x, before)
        np.testing.assert_array_equal(target, [2., 0.])
        np.testing.assert_allclose(z[:, -1], [.6, -.6, 0.])
        np.testing.assert_allclose(z[:, 2:4], [[.6, 0.], [-.6, 0.], [0., 0.]])
        np.testing.assert_allclose(z[:, 4:6], [[0., .8], [0., .8], [0., 1.]])
        unit = x / np.linalg.norm(x, axis=1, keepdims=True)
        np.testing.assert_allclose(z[:, 2:4] + z[:, 4:6], unit)
        normalized_z = transform.transform(unit)
        np.testing.assert_allclose(normalized_z[:, :2], normalized_z[:, 2:4] + normalized_z[:, 4:6])
        np.testing.assert_allclose(z[:, 4:6] @ transform.target_embedding, 0., atol=1e-12)

    def test_dimensions_metadata_reuse_and_no_labels(self):
        for columns in ({'Title'}, {'Title', 'Review Text'}):
            with self.subTest(columns=columns):
                x = pd.DataFrame({col: ['first', 'second'] for col in sorted(columns)})
                for i in range(8):
                    x[f'tabular_{i}'] = [i, i + 1]
                model = Mock()
                calls = []
                vector = np.ones(384, dtype=np.float32) / np.sqrt(384)

                def encode(texts, col_name, **kwargs):
                    self.assertIs(kwargs['model'], model)
                    calls.append((texts, col_name))
                    if col_name is None:
                        self.assertEqual(texts, ['Raw target name'])
                    else:
                        self.assertIn(col_name, columns)
                    return np.tile(vector, (len(texts), 1))

                with patch.object(embeddings, 'get_vanilla_e5', return_value=(model, Mock())), \
                        patch.object(embeddings, 'encode_texts_with_e5', side_effect=encode), \
                        patch.object(embeddings, 'PCA', side_effect=AssertionError('PCA must not run')):
                    encoders = embeddings.fit_text_encoders(
                        x=x, text_features=columns, device='cpu', y=object(),
                        target_aware_transform=True, target_column_name='Raw target name',
                    )  # no_pca defaults to False: this experiment must still bypass PCA.
                    self.assertEqual(calls, [(['Raw target name'], None)])
                    model.requires_grad_.assert_called_once_with(False)
                    shared = next(iter(encoders.values())).encoder
                    self.assertTrue(all(wrapper.encoder is shared for wrapper in encoders.values()))
                    for frame in (x, x.iloc[:1]):
                        result = embeddings.transform_text_features(frame, encoders, 'cpu')
                        self.assertEqual(result.shape, (len(frame), 1153 * len(columns) + 8))
                        pd.testing.assert_frame_equal(result.filter(like='tabular_'), frame.filter(like='tabular_'))
                        for col in columns:
                            e = result[[f'{col}_e5_{i}' for i in range(384)]].to_numpy()
                            parallel = result[[f'{col}_target_parallel_{i}' for i in range(384)]].to_numpy()
                            residual = result[[f'{col}_target_residual_{i}' for i in range(384)]].to_numpy()
                            np.testing.assert_allclose(e, parallel + residual, atol=1e-7)
                            np.testing.assert_allclose(result[f'{col}_target_cosine'], 1., atol=1e-6)
                    self.assertEqual(sum(col is None for _, col in calls), 1)

    def test_off_does_not_embed_target(self):
        x = pd.DataFrame({'Title': ['a', 'b']})
        with patch.object(embeddings, 'get_vanilla_e5', return_value=(Mock(), Mock())), \
                patch.object(embeddings, 'encode_texts_with_e5', return_value=np.ones((2, 384))) as encode:
            encoders = embeddings.fit_text_encoders(
                x=x, text_features={'Title'}, device='cpu', no_pca=True,
                target_column_name='Unused metadata', target_aware_transform=False,
            )
            result = embeddings.transform_text_features(x, encoders, 'cpu')
            self.assertEqual(result.shape, (2, 384))
            self.assertEqual(result.columns[0], 'Title_txt_pca_0')
            self.assertTrue(all(call.kwargs['col_name'] == 'Title' for call in encode.call_args_list))

    def test_invalid_modes_and_metadata(self):
        for invalid in ({'target_guided_fusion': True}, {'target_conditioned_embedding': True},
                        {'tune_e5': True}, {'e5_model_name': TF_IDF}, {'target_column_name': None}):
            with self.subTest(invalid=invalid):
                kwargs = dict(target_aware_transform=True, target_column_name='Raw target')
                kwargs.update(invalid)
                with self.assertRaises(ValueError):
                    embeddings.fit_text_encoders(x=None, text_features={'Text'}, device='cpu', **kwargs)
                with self.assertRaises(ValueError):
                    LightGBM(problem_type=SupervisedTask.MULTICLASS, device='cpu', **kwargs)

    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError):
            embeddings._TargetAwareTransform(np.zeros(2))
        with self.assertRaises(ValueError):
            embeddings._TargetAwareTransform(np.ones(2)).transform(np.zeros((1, 2)))


if __name__ == '__main__':
    unittest.main()
