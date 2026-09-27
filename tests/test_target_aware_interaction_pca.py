"""Frozen synthetic embeddings and real PCA; no model downloads or training."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from multabench.baselines.preprocessing import text_embeddings as embeddings
from multabench.baselines.lgbm import LightGBM
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import TF_IDF


class InteractionPCATest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.vectors = rng.normal(size=(70, 2, 384)).astype(np.float32)
        self.target = rng.normal(size=384).astype(np.float32)
        self.columns = ['Review Text', 'Title']
        self.x = pd.DataFrame({col: [str(i) for i in range(70)] for col in self.columns})
        for i in range(8):
            self.x[f'tabular_{i}'] = np.arange(70) + i
        self.calls = []
        self.model = Mock()

    def encode(self, texts, col_name, **kwargs):
        self.assertIs(kwargs['model'], self.model)
        self.calls.append((list(texts), col_name))
        if col_name is None:
            self.assertEqual(texts, ['Raw target metadata'])
            return self.target[None, :]
        return self.vectors[[int(v) for v in texts], self.columns.index(col_name)]

    def test_interactions_preservation_and_no_test_fit(self):
        train, test = self.x.iloc[:60], self.x.iloc[60:]
        pca = PCA(n_components=50, random_state=17)
        with patch.object(embeddings, 'get_vanilla_e5', return_value=(self.model, Mock())), \
                patch.object(embeddings, 'encode_texts_with_e5', side_effect=self.encode), \
                patch.object(embeddings, 'PCA', return_value=pca), \
                patch.object(pca, 'fit', wraps=pca.fit) as fit:
            transformer = embeddings.TargetAwareInteractionPCA.fit(
                train, set(self.columns), 'Raw target metadata', 'cpu', n_components=50)
            e1, e2 = self.vectors[:60, 0], self.vectors[:60, 1]
            expected = np.concatenate([e1 * self.target, e2 * self.target, e1 * e2], axis=1)
            self.assertEqual(expected.shape, (60, 1152))
            np.testing.assert_array_equal(transformer.interactions(e1, e2), expected)
            fit.assert_called_once()
            np.testing.assert_array_equal(fit.call_args.args[0], expected)
            self.model.requires_grad_.assert_called_once_with(False)
            with patch.object(pca, 'fit', side_effect=AssertionError('Inference refit')), \
                    patch.object(pca, 'fit_transform', side_effect=AssertionError('Inference fit_transform')):
                for frame in (train, test):
                    result = transformer.transform(frame, 'cpu')
                    self.assertEqual(result.shape, (len(frame), 826))
                    pd.testing.assert_frame_equal(result.filter(like='tabular_'), frame.filter(like='tabular_'))
                    text = result.drop(columns=[f'tabular_{i}' for i in range(8)])
                    ids = frame.index.to_numpy()
                    original = np.concatenate([self.vectors[ids, 0], self.vectors[ids, 1]], axis=1)
                    np.testing.assert_array_equal(text.iloc[:, :768], original)
                    self.assertEqual(text.columns[0], 'Review Text_txt_pca_0')
                    self.assertEqual(text.columns[-1], 'target_interaction_pca_49')
                    interactions = transformer.interactions(self.vectors[ids, 0], self.vectors[ids, 1])
                    np.testing.assert_allclose(text.iloc[:, 768:], pca.transform(interactions))
            self.assertEqual(sum(col is None for _, col in self.calls), 1)
            self.assertEqual(transformer.raw_interaction_dimensions, 1152)
            self.assertEqual(transformer.final_text_representation_dimensions, 818)

    def test_model_dispatch_passes_training_x_but_no_y(self):
        with patch.object(LightGBM, 'initialize_model', return_value=Mock()):
            model = LightGBM(problem_type=SupervisedTask.MULTICLASS, device='cpu',
                             target_aware_interaction_pca=True, target_column_name='Raw target metadata',
                             no_pca=True)
        model.text_features = set(self.columns)
        # The legacy no-PCA feature-count cap must not reject generalized interaction mode.
        model.raise_if_no_pca_and_too_multimodal(None)
        model.USE_CATEGORICAL_ENCODING = False
        labels = object()
        train = self.x.iloc[:60]
        with patch.object(model, 'do_model_agnostic_preprocessing', return_value=(train, labels)), \
                patch.object(embeddings.TargetAwareInteractionPCA, 'fit', return_value=Mock()) as fit:
            model.fit_preprocessor(train, labels)
            self.assertIs(fit.call_args.kwargs['x'], train)
            self.assertNotIn('y', fit.call_args.kwargs)
            self.assertEqual(fit.call_args.kwargs['n_components'], 50)
            self.assertIs(model.interaction_transformer, fit.return_value)

    def test_mode_off_keeps_baseline(self):
        with patch.object(embeddings, 'get_vanilla_e5', return_value=(self.model, Mock())), \
                patch.object(embeddings, 'encode_texts_with_e5', side_effect=self.encode):
            encoders = embeddings.fit_text_encoders(self.x, set(self.columns), 'cpu', no_pca=True)
            result = embeddings.transform_text_features(self.x, encoders, 'cpu')
            self.assertEqual(result.shape, (70, 776))
            self.assertFalse(any(c is None for _, c in self.calls))
            self.assertFalse(any(c.startswith('target_interaction') for c in result.columns))

    def test_column_count_and_component_limits(self):
        with self.assertRaisesRegex(ValueError, 'at least one text column'):
            embeddings.TargetAwareInteractionPCA.fit(self.x, set(), 'Target', 'cpu')
        for count in (0, -1, 71):
            with self.assertRaisesRegex(ValueError, 'training row count'):
                embeddings.TargetAwareInteractionPCA.fit(self.x, set(self.columns), 'Target', 'cpu', n_components=count)

        with patch.object(embeddings, 'get_vanilla_e5', return_value=(Mock(), Mock())), \
                patch.object(embeddings, 'encode_texts_with_e5', return_value=np.ones((1, 384))):
            with self.assertRaisesRegex(ValueError, 'effective maximum 384'):
                embeddings.TargetAwareInteractionPCA.fit(
                    pd.DataFrame({'Text': ['x'] * 400}), {'Text'}, 'Target', 'cpu', n_components=385)

    def test_generalized_order_preservation_and_leakage(self):
        rng = np.random.default_rng(42)
        for k in (1, 2, 3, 6):
            with self.subTest(k=k):
                columns = [f'Text {i}' for i in range(k)]
                values = rng.normal(size=(65, k, 384)).astype(np.float32)
                target = rng.normal(size=384).astype(np.float32)
                x = pd.DataFrame({col: [str(i) for i in range(65)] for col in reversed(columns)})
                calls = []

                def encode(texts, col_name, **kwargs):
                    calls.append(col_name)
                    if col_name is None:
                        self.assertEqual(texts, ['Only raw metadata'])
                        return target[None, :]
                    return values[[int(i) for i in texts], columns.index(col_name)]

                pca = PCA(n_components=50, random_state=17)
                with patch.object(embeddings, 'get_vanilla_e5', return_value=(Mock(), Mock())), \
                        patch.object(embeddings, 'encode_texts_with_e5', side_effect=encode), \
                        patch.object(embeddings, 'PCA', return_value=pca), \
                        patch.object(pca, 'fit', wraps=pca.fit) as fit:
                    transformer = embeddings.TargetAwareInteractionPCA.fit(
                        x.iloc[:60], set(columns), 'Only raw metadata', 'cpu')
                    e = [values[:60, i] for i in range(k)]
                    if k == 1:
                        blocks = [e[0] * target]
                    elif k == 2:
                        blocks = [e[0] * target, e[1] * target, e[0] * e[1]]
                    elif k == 3:
                        blocks = [e[0] * target, e[1] * target, e[2] * target,
                                  e[0] * e[1], e[0] * e[2], e[1] * e[2]]
                    else:
                        blocks = [v * target for v in e] + [e[i] * e[j] for i in range(k) for j in range(i + 1, k)]
                    expected = np.concatenate(blocks, axis=1)
                    np.testing.assert_array_equal(transformer.interactions(*e), expected)
                    fit.assert_called_once()
                    np.testing.assert_array_equal(fit.call_args.args[0], expected)
                    self.assertEqual(transformer.n_pairwise_interactions, k * (k - 1) // 2)
                    self.assertEqual(transformer.raw_interaction_dimensions, 384 * (k + k * (k - 1) // 2))
                    with patch.object(pca, 'fit', side_effect=AssertionError('refit')), \
                            patch.object(pca, 'fit_transform', side_effect=AssertionError('refit')):
                        for frame in (x.iloc[:60], x.iloc[60:62], x.iloc[62:]):
                            result = transformer.transform(frame, 'cpu')
                            self.assertEqual(result.shape, (len(frame), k * 384 + 50))
                            original = np.concatenate([values[frame.index, i] for i in range(k)], axis=1)
                            np.testing.assert_array_equal(result.iloc[:, :k * 384], original)
                            self.assertEqual(result.columns[0], 'Text 0_txt_pca_0')
                            self.assertEqual(result.columns[-1], 'target_interaction_pca_49')
                    self.assertEqual(calls.count(None), 1)

    def test_invalid_modes(self):
        for invalid in ({'target_guided_fusion': True}, {'target_conditioned_embedding': True},
                        {'target_aware_transform': True}, {'tune_e5': True}, {'e5_model_name': TF_IDF}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                LightGBM(problem_type=SupervisedTask.MULTICLASS, device='cpu',
                         target_aware_interaction_pca=True, target_column_name='Target', **invalid)


if __name__ == '__main__':
    unittest.main()
