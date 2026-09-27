"""Exercise actual E5 tokenization inputs using a tiny frozen model stub."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from multabench.baselines.preprocessing import text_embeddings as embeddings
from multabench.baselines.lgbm import LightGBM
from multabench.datasets.objects import SupervisedTask
from multabench.e5.constants import TF_IDF


class FakeE5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(384))

    def forward(self, input_ids, attention_mask):
        assert not torch.is_grad_enabled()
        return SimpleNamespace(last_hidden_state=self.weight.expand(len(input_ids), 1, 384) + 0)


class TargetConditionedTest(unittest.TestCase):
    def setUp(self):
        self.x = pd.DataFrame({'Review Text': ['Love this dress', 'Poor fit'],
                               'Title': ['Beautiful', 'Disappointed'], 'Age': [25, 40]})
        self.inputs = []
        self.model = FakeE5()

    def tokenize(self, texts, **kwargs):
        self.inputs.extend(texts)
        return {'input_ids': torch.ones(len(texts), 1, dtype=torch.long),
                'attention_mask': torch.ones(len(texts), 1, dtype=torch.long)}

    def test_exact_tokenizer_inputs_dimensions_and_inference(self):
        for enabled in (False, True):
            for columns in ({'Title'}, {'Review Text', 'Title'}):
                with self.subTest(enabled=enabled, columns=columns), \
                        patch.object(embeddings, 'get_vanilla_e5', return_value=(self.model, self.tokenize)), \
                        patch.object(embeddings, 'PCA', side_effect=AssertionError('Unexpected PCA')):
                    self.inputs.clear()
                    x = self.x[[*sorted(columns), 'Age']]
                    encoders = embeddings.fit_text_encoders(
                        x=x, text_features=columns, device='cpu', no_pca=True,
                        y=object(),  # No iterable/indexable labels are available to the encoder.
                        target_column_name='Rating', target_conditioned_embedding=enabled,
                    )
                    expected = [f'passage: {"Target: Rating. " if enabled else ""}{col}: {value}'
                                for col in sorted(columns) for value in x[col]]
                    self.assertEqual(self.inputs, expected)  # No separate target embedding call.
                    if enabled:
                        self.assertFalse(self.model.weight.requires_grad)
                    self.inputs.clear()
                    result = embeddings.transform_text_features(x, encoders, 'cpu')
                    self.assertEqual(self.inputs, expected)
                    self.assertEqual(result.shape, (2, 384 * len(columns) + 1))
                    pd.testing.assert_series_equal(result['Age'], x['Age'])
                    for col in columns:
                        suffix = 'target_conditioned' if enabled else 'txt_pca'
                        names = [f'{col}_{suffix}_{i}' for i in range(384)]
                        np.testing.assert_allclose(result[names], 1 / np.sqrt(384), rtol=1e-6)
                    self.inputs.clear()
                    embeddings.transform_text_features(x.iloc[:1], encoders, 'cpu')
                    self.assertEqual(self.inputs, [f'passage: {"Target: Rating. " if enabled else ""}{col}: {x[col].iloc[0]}'
                                                   for col in sorted(columns)])

    def test_invalid_configurations(self):
        for invalid in ({'target_guided_fusion': True}, {'tune_e5': True},
                        {'e5_model_name': TF_IDF}, {'target_column_name': None}):
            with self.subTest(invalid=invalid), patch.object(embeddings, 'get_vanilla_e5') as load:
                kwargs = dict(target_conditioned_embedding=True, target_column_name='Rating')
                kwargs.update(invalid)
                with self.assertRaises(ValueError):
                    embeddings.fit_text_encoders(x=self.x, text_features={'Title'}, device='cpu', **kwargs)
                with self.assertRaises(ValueError):
                    LightGBM(problem_type=SupervisedTask.MULTICLASS, device='cpu', **kwargs)
                load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
