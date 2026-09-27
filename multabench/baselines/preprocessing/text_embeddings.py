import os

from sklearn.decomposition import PCA

from tabstar.constants import SEED

from multabench.preprocessing.discretize import discretize_numerical

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppresses warning, avoids deadlock
from typing import Dict, Set, Optional, Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import torch

from multabench.e5.constants import E5_SMALL_V2, TF_IDF
from multabench.e5.e5_finetune import encode_texts_with_e5, get_vanilla_e5
from multabench.utils.pca_logging import log_pca_variance

PCA_COMPONENTS = 30


class _IdentityTransform:
    """No-op transformer: returns embeddings as-is (no PCA)."""
    def __init__(self, n_components: int):
        self.n_components = n_components

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _TargetGuidedTransform:
    """Fixed metadata-only interaction; no fitting or label access."""

    def __init__(self, target_embedding: np.ndarray):
        self.target_embedding = target_embedding
        self.n_components = 2 * target_embedding.shape[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.concatenate([X, X * self.target_embedding], axis=1)


class _TargetAwareTransform:
    """Metadata-only geometry on unit vectors, preserving original embedding features."""

    def __init__(self, target_embedding: np.ndarray):
        norm = np.linalg.norm(target_embedding)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Target embedding must have a finite, nonzero norm.')
        self.target_embedding = target_embedding / norm
        self.n_components = 3 * target_embedding.shape[0] + 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError('Text embeddings must have finite, nonzero norms.')
        unit = X / norms  # New array: the original E5 block is never modified.
        cosine = unit @ self.target_embedding[:, None]
        parallel = cosine * self.target_embedding
        residual = unit - parallel
        return np.concatenate([X, parallel, residual, cosine], axis=1)


class SkrubColumnEncoder:
    """Per-column text encoder using skrub.StringEncoder (TF-IDF + TruncatedSVD). CPU-only, no tokenizer needed."""

    def __init__(self, string_encoder: Any, col_name: str, n_components: int):
        self.string_encoder = string_encoder
        self.col_name = col_name
        self.n_components = n_components
        self.encoder = self  # so wrapper.encoder.transform(X) delegates to self.transform(X)

    def encode_texts(self, texts: list[str], device) -> np.ndarray:
        """Encode texts with skrub StringEncoder (device ignored — CPU-only)."""
        result = self.string_encoder.transform(pd.Series(texts, dtype=str))
        return np.asarray(result)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X  # encode_texts already returns final (N, n_components) array


class E5ColumnEncoder:
    """Per-column text encoder: holds E5 model, tokenizer (processor), and PCA (or identity). Uses passage: col_name: col_val format."""

    def __init__(self, model: Any, tokenizer: Any, encoder: Any, col_name: str,
                 target_column_name: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.col_name = col_name
        self.target_column_name = target_column_name
        self.n_components = encoder.n_components

    def encode_texts(self, texts: list[str], device: torch.device) -> np.ndarray:
        """Encode texts with this column's E5 model and tokenizer (passage: col_name: col_val)."""
        return encode_texts_with_e5(
            texts=texts,
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            col_name=_encoding_column_name(self.col_name, self.target_column_name),
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.encoder.transform(X)


class TargetAwareInteractionPCA:
    """Joint frozen text transformer. fit() receives training X and metadata only."""

    @classmethod
    def fit(cls, x: DataFrame, text_features: Set[str], target_column_name: str,
            device: torch.device, e5_model_name: str = E5_SMALL_V2,
            n_components: int = 50):
        columns = sorted(text_features)
        if len(columns) != 2:
            raise ValueError(f'Target-aware interaction PCA requires exactly two text columns; detected {columns}.')
        if not isinstance(target_column_name, str) or not target_column_name.strip():
            raise ValueError('Target-aware interaction PCA requires raw target column name metadata.')
        if e5_model_name == TF_IDF:
            raise ValueError('Target-aware interaction PCA requires frozen E5, not TF-IDF.')
        if not isinstance(n_components, int) or n_components < 1 or n_components > len(x):
            raise ValueError('interaction_pca_components must be positive and no greater than the training row count.')
        result = cls()
        model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
        model.requires_grad_(False)
        result.target_embedding = encode_texts_with_e5(
            texts=[target_column_name], col_name=None, model=model, tokenizer=tokenizer, device=device,
        )[0]
        dimension = result.target_embedding.shape[0]
        result.raw_interaction_dimensions = 3 * dimension
        if n_components > result.raw_interaction_dimensions:
            raise ValueError('interaction_pca_components exceeds the raw interaction dimensionality.')
        result.encoders = {col: E5ColumnEncoder(model, tokenizer, _IdentityTransform(dimension), col)
                           for col in columns}
        e1, e2 = result._encode(x, device)
        result.pca = PCA(n_components=n_components, random_state=SEED)
        result.pca.fit(result.interactions(e1, e2))
        result.final_text_representation_dimensions = 2 * dimension + n_components
        print(f"Target-aware interaction PCA: enabled\nTarget column: {target_column_name}\n"
              f"Text columns: {columns}\nOriginal text embedding dimensions: {2 * dimension}\n"
              f"Raw interaction dimensions: {result.raw_interaction_dimensions}\n"
              f"Interaction PCA components: {n_components}\n"
              f"Final text representation dimensions: {result.final_text_representation_dimensions}")
        return result

    def _encode(self, x: DataFrame, device: torch.device):
        return [encoder.encode_texts(x[col].astype(str).fillna('').tolist(), device)
                for col, encoder in self.encoders.items()]

    def interactions(self, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
        return np.concatenate([e1 * self.target_embedding, e2 * self.target_embedding, e1 * e2], axis=1)

    def transform(self, x: DataFrame, device: torch.device) -> DataFrame:
        e1, e2 = self._encode(x, device)
        compact = self.pca.transform(self.interactions(e1, e2))
        names = [f'{col}_txt_pca_{i}' for col, encoder in self.encoders.items()
                 for i in range(encoder.n_components)]
        names += [f'target_interaction_pca_{i}' for i in range(self.pca.n_components)]
        text = pd.DataFrame(np.concatenate([e1, e2, compact], axis=1), index=x.index, columns=names)
        return pd.concat([x.drop(columns=list(self.encoders)), text], axis=1)


def _encoding_column_name(col_name: str, target_column_name: str | None) -> str:
    if target_column_name is None:
        return col_name
    return f"Target: {target_column_name}. {col_name}"


def fit_text_encoders_skrub(
    x: DataFrame,
    text_features_list: list[str],
    pca_components: int = PCA_COMPONENTS,
) -> Dict[str, SkrubColumnEncoder]:
    """Fit one SkrubColumnEncoder per column using skrub.StringEncoder (TF-IDF + TruncatedSVD)."""
    from skrub import StringEncoder
    text_encoders: Dict[str, SkrubColumnEncoder] = {}
    for col in text_features_list:
        col_series = x[col].astype(str).fillna("")
        print(f"Fitting SkrubStringEncoder for column {col} with n_components={pca_components} for {len(col_series)} texts")
        string_enc = StringEncoder(n_components=pca_components)
        string_enc.fit(col_series)
        text_encoders[str(col)] = SkrubColumnEncoder(
            string_encoder=string_enc,
            col_name=str(col),
            n_components=pca_components,
        )
    return text_encoders


def fit_text_encoders_vanilla(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    e5_model_name: str = E5_SMALL_V2,
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
    target_guided_fusion: bool = False,
    target_column_name: str | None = None,
    target_conditioned_embedding: bool = False,
    target_aware_transform: bool = False,
) -> Dict[str, E5ColumnEncoder]:
    """Fit one E5ColumnEncoder per column using shared vanilla E5 + PCA per column. Uses passage: col_name: col_val format."""
    text_encoders: Dict[str, E5ColumnEncoder] = {}
    model, tokenizer = get_vanilla_e5(device, model_name=e5_model_name)
    if target_aware_transform:
        model.requires_grad_(False)
        target_embedding = encode_texts_with_e5(
            texts=[target_column_name], col_name=None, model=model, tokenizer=tokenizer, device=device,
        )[0]
        transform = _TargetAwareTransform(target_embedding)
        dimension = target_embedding.shape[0]
        print(f"Target-aware frozen transform: enabled\nTarget column: {target_column_name}\n"
              f"Target embedding dimension: {dimension}\nText columns: {text_features_list}\n"
              f"Original text embedding dimensions: {dimension * len(text_features_list)}\n"
              f"Final text representation dimensions: {transform.n_components * len(text_features_list)}")
        return {str(col): E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=transform, col_name=str(col))
                for col in text_features_list}
    conditioned_target = target_column_name if target_conditioned_embedding else None
    if target_conditioned_embedding:
        model.requires_grad_(False)
    if target_guided_fusion:
        model.requires_grad_(False)
        target_embedding = encode_texts_with_e5(
            texts=[target_column_name], col_name=None, model=model, tokenizer=tokenizer, device=device,
        )[0]
        dimension = target_embedding.shape[0]
        print(f"Target-guided fusion: enabled\nTarget column: {target_column_name}\n"
              f"Target embedding dimension: {dimension}\nText columns: {text_features_list}\n"
              f"Original text embedding dimensions: {dimension * len(text_features_list)}\n"
              f"Final text representation dimensions: {2 * dimension * len(text_features_list)}")
    for col in text_features_list:
        if target_guided_fusion:
            text_encoders[str(col)] = E5ColumnEncoder(
                model=model, tokenizer=tokenizer,
                encoder=_TargetGuidedTransform(target_embedding), col_name=str(col),
            )
            continue
        texts = x[col].astype(str).fillna("").tolist()
        print(f"Fitting E5ColumnEncoder for column {col} with model {e5_model_name} for {len(texts)} texts")
        col_embeddings = encode_texts_with_e5(
            texts=texts, model=model, tokenizer=tokenizer, device=device,
            col_name=_encoding_column_name(str(col), conditioned_target),
        )
        if no_pca:
            encoder = _IdentityTransform(n_components=col_embeddings.shape[1])
        else:
            encoder = PCA(n_components=pca_components, random_state=SEED)
            encoder.fit(col_embeddings)
            log_pca_variance(pca=encoder, col_name=col)
        text_encoders[str(col)] = E5ColumnEncoder(model=model, tokenizer=tokenizer, encoder=encoder,
                                               col_name=str(col), target_column_name=conditioned_target)
    if target_conditioned_embedding:
        print(f"Target-conditioned E5: enabled\nTarget column: {target_column_name}\n"
              f"Text columns: {text_features_list}\n"
              f"E5 embedding dimension per text column: {col_embeddings.shape[1]}\n"
              f"Final text representation dimensions: {sum(e.n_components for e in text_encoders.values())}")
    return text_encoders


def fit_text_encoders_tuned(
    x: DataFrame,
    text_features_list: list[str],
    device: torch.device,
    y: Series,
    e5_train_kwargs: Dict[str, Any],
    is_cls: bool,
    d_output: int,
    e5_model_name: str = E5_SMALL_V2,
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
) -> Dict[str, E5ColumnEncoder]:
    """Fit a single E5 model for all text columns with passage: col_name: col_val format. Each column gets an E5ColumnEncoder sharing the same tuned model."""
    from transformers import AutoTokenizer

    from multabench.e5.e5_finetune import finetune_e5_with_lora
    from multabench.preprocessing.splits import split_to_val

    if not is_cls:
        y = discretize_numerical(y, n_bins=20)
    x_tr, x_val, y_tr, y_val = split_to_val(x=x, y=y, is_cls=True)
    kwargs = e5_train_kwargs or {}

    # Build combined train/val: for each row, for each text col, create (col_name: col_val, label)
    train_texts: list[str] = []
    train_y: list = []
    for idx in x_tr.index:
        row_y = y_tr.loc[idx]
        for col in text_features_list:
            val = str(x_tr.loc[idx, col]).strip() if pd.notna(x_tr.loc[idx, col]) else ""
            train_texts.append(f"{col}: {val}")
            train_y.append(row_y)

    val_texts: list[str] = []
    val_y: list = []
    for idx in x_val.index:
        row_y = y_val.loc[idx]
        for col in text_features_list:
            val = str(x_val.loc[idx, col]).strip() if pd.notna(x_val.loc[idx, col]) else ""
            val_texts.append(f"{col}: {val}")
            val_y.append(row_y)

    train_y_arr = np.array(train_y)
    val_y_arr = np.array(val_y)

    tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
    print(f"Finetuning single E5 ({e5_model_name}) for {len(text_features_list)} columns with {len(train_texts)} train examples (passage: col_name: col_val)")
    tuned_model, tuned_tokenizer = finetune_e5_with_lora(
        train_texts=train_texts,
        train_y=train_y_arr,
        val_texts=val_texts,
        val_y=val_y_arr,
        device=device,
        tokenizer=tokenizer,
        model_name=e5_model_name,
        **kwargs,
    )
    tuned_model.to(device)

    text_encoders: Dict[str, E5ColumnEncoder] = {}
    for col in text_features_list:
        texts = x[col].astype(str).fillna("").tolist()
        col_embeddings = encode_texts_with_e5(texts=texts, model=tuned_model, tokenizer=tuned_tokenizer, device=device, col_name=str(col))
        if no_pca:
            encoder = _IdentityTransform(n_components=col_embeddings.shape[1])
        else:
            encoder = PCA(n_components=pca_components, random_state=SEED)
            encoder.fit(col_embeddings)
            log_pca_variance(pca=encoder, col_name=col)
        text_encoders[col] = E5ColumnEncoder(model=tuned_model, tokenizer=tuned_tokenizer, encoder=encoder, col_name=str(col))
    return text_encoders


def fit_text_encoders(
    x: DataFrame,
    text_features: Set[str],
    device: torch.device,
    y: Optional[Series] = None,
    tune_e5: bool = False,
    e5_train_kwargs: Optional[Dict[str, Any]] = None,
    is_cls: bool = True,
    d_output: int = 2,
    e5_model_name: str = E5_SMALL_V2,
    pca_components: int = PCA_COMPONENTS,
    no_pca: bool = False,
    target_guided_fusion: bool = False,
    target_column_name: str | None = None,
    target_conditioned_embedding: bool = False,
    target_aware_transform: bool = False,
) -> Dict[str, E5ColumnEncoder]:
    """
    Fit one E5 model per text column (or vanilla E5 shared across columns when not tuning).
    Each column gets an E5ColumnEncoder wrapper holding model, tokenizer, and PCA.
    Returns text_encoders mapping column -> E5ColumnEncoder.
    """
    text_features_list = sorted(text_features)
    if target_aware_transform:
        if target_guided_fusion or target_conditioned_embedding:
            raise ValueError('target_aware_transform cannot be combined with other target embedding experiments.')
        if tune_e5 or e5_model_name == TF_IDF:
            raise ValueError('Target-aware transform requires frozen E5, without fine-tuning or TF-IDF.')
        if not isinstance(target_column_name, str) or not target_column_name.strip():
            raise ValueError('Target-aware transform requires target column name metadata.')
        if not text_features_list:
            raise ValueError('Target-aware transform requires at least one detected text column.')
    if target_conditioned_embedding:
        if target_guided_fusion:
            raise ValueError('target_conditioned_embedding and target_guided_fusion cannot be combined.')
        if tune_e5 or e5_model_name == TF_IDF:
            raise ValueError('Target-conditioned embeddings require frozen E5, without fine-tuning or TF-IDF.')
        if not isinstance(target_column_name, str) or not target_column_name.strip():
            raise ValueError('Target-conditioned embeddings require target column name metadata.')
        if not text_features_list:
            raise ValueError('Target-conditioned embeddings require at least one detected text column.')
    if target_guided_fusion:
        if tune_e5 or e5_model_name == TF_IDF:
            raise ValueError('Target-guided fusion requires frozen E5, without fine-tuning or TF-IDF.')
        if not isinstance(target_column_name, str) or not target_column_name.strip():
            raise ValueError('Target-guided fusion requires target column name metadata.')
        if not text_features_list:
            raise ValueError('Target-guided fusion requires at least one detected text column.')
    if not text_features_list:
        return {}
    if e5_model_name == TF_IDF:
        return fit_text_encoders_skrub(
            x=x,
            text_features_list=text_features_list,
            pca_components=pca_components,
        )
    if tune_e5 and y is not None:
        return fit_text_encoders_tuned(
            x=x,
            text_features_list=text_features_list,
            device=device,
            y=y,
            e5_train_kwargs=e5_train_kwargs or {},
            is_cls=is_cls,
            d_output=d_output,
            e5_model_name=e5_model_name,
            pca_components=pca_components,
            no_pca=no_pca,
        )
    return fit_text_encoders_vanilla(
        x=x,
        text_features_list=text_features_list,
        device=device,
        e5_model_name=e5_model_name,
        pca_components=pca_components,
        no_pca=no_pca,
        target_guided_fusion=target_guided_fusion,
        target_column_name=target_column_name,
        target_conditioned_embedding=target_conditioned_embedding,
        target_aware_transform=target_aware_transform,
    )


def transform_text_features(
    x: DataFrame,
    text_encoders: Dict[str, E5ColumnEncoder],
    device: torch.device,
) -> DataFrame:
    for text_col, wrapper in text_encoders.items():
        texts = x[text_col].astype(str).fillna("").tolist()
        embeddings = wrapper.encode_texts(texts, device)
        n_components = wrapper.n_components
        pca_vec = wrapper.encoder.transform(embeddings)
        pca_cols = [f"{text_col}_txt_pca_{i}" for i in range(n_components)]
        if isinstance(wrapper.encoder, _TargetAwareTransform):
            dimension = wrapper.encoder.target_embedding.shape[0]
            pca_cols = ([f"{text_col}_e5_{i}" for i in range(dimension)] +
                        [f"{text_col}_target_parallel_{i}" for i in range(dimension)] +
                        [f"{text_col}_target_residual_{i}" for i in range(dimension)] +
                        [f"{text_col}_target_cosine"])
        if isinstance(wrapper, E5ColumnEncoder) and wrapper.target_column_name is not None:
            pca_cols = [f"{text_col}_target_conditioned_{i}" for i in range(n_components)]
        if isinstance(wrapper.encoder, _TargetGuidedTransform):
            dimension = wrapper.encoder.target_embedding.shape[0]
            pca_cols = ([f"{text_col}_e5_{i}" for i in range(dimension)] +
                        [f"{text_col}_target_product_{i}" for i in range(dimension)])
        pca_df = pd.DataFrame(pca_vec, index=x.index, columns=pca_cols)
        cols_before = len(x.columns)
        x = x.drop(columns=[text_col])
        x = pd.concat([x, pca_df], axis=1)
        assert len(x.columns) == cols_before + n_components - 1
    return x
