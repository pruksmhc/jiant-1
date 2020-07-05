from abc import ABC

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple

from jiant.tasks.core import (
    Task,
    TaskTypes,
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
)
from jiant.tasks.lib.templates.shared import (
    create_input_set_from_tokens_and_segments,
    add_cls_token,
)
from jiant.tasks.utils import truncate_sequences, pad_to_max_seq_length
from jiant.utils.retokenize import TokenAligner


@dataclass
class Example(BaseExample):

    guid: str
    passage: str
    question: str
    answer: str
    answer_char_span: (int, int)

    def tokenize(self, tokenizer):

        passage_tokens = tokenizer.tokenize(self.passage)
        token_aligner = TokenAligner(source=self.passage, target=passage_tokens)
        source_char_idx_to_target_token_idx = token_aligner.C.dot(
            token_aligner.V.T
        )  # maybe make this a function in retokenize?
        return TokenizedExample(
            guid=self.guid,
            passage=passage_tokens,
            question=tokenizer.tokenize(self.question),
            answer_str=self.answer,
            passage_str=self.passage,
            answer_token_span=tuple(
                map(
                    lambda x: source_char_idx_to_target_token_idx[x].nonzero()[1],
                    self.answer_char_span,
                )
            ),
            token_idx_to_char_idx_map=source_char_idx_to_target_token_idx.T,
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    passage: List[str]
    question: List[str]
    answer_str: str
    passage_str: str
    answer_token_span: Tuple[int, int]
    token_idx_to_char_idx_map: np.ndarray

    def featurize(self, tokenizer, feat_spec):

        if feat_spec.sep_token_extra:
            maybe_extra_sep = [tokenizer.sep_token]
            maybe_extra_sep_segment_id = [feat_spec.sequence_a_segment_id]
            special_tokens_count = 4  # CLS, SEP-SEP, SEP
        else:
            maybe_extra_sep = []
            maybe_extra_sep_segment_id = []
            special_tokens_count = 3  # CLS, SEP, SEP

        passage, question = truncate_sequences(
            tokens_ls=[self.passage, self.question],
            max_length=feat_spec.max_seq_length - special_tokens_count,
        )
        assert (
            len(passage) >= self.answer_token_span[1]
        ), "Answer span truncated, please raise max_seq_length."
        unpadded_inputs = add_cls_token(
            unpadded_tokens=(
                passage + [tokenizer.sep_token] + maybe_extra_sep + question + [tokenizer.sep_token]
            ),
            unpadded_segment_ids=(
                [feat_spec.sequence_a_segment_id] * (len(passage) + 1)
                + maybe_extra_sep_segment_id
                + [feat_spec.sequence_b_segment_id] * (len(question) + 1)
            ),
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        gt_span_idxs = list(map(lambda x: x + unpadded_inputs.cls_offset), self.answer_token_span)
        input_set = create_input_set_from_tokens_and_segments(
            unpadded_tokens=unpadded_inputs.unpadded_tokens,
            unpadded_segment_ids=unpadded_inputs.unpadded_segment_ids,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
        )
        pred_span_mask = pad_to_max_seq_length(
            ls=[0] * unpadded_inputs.cls_offset + [1] * len(passage),
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=0,
            pad_right=not feat_spec.pad_on_left,
        )
        token_idx_to_char_idx_start = pad_to_max_seq_length(
            ls=[-1] * unpadded_inputs.cls_offset
            + self.token_idx_to_char_idx_map.argmax(axis=1).tolist(),
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=-1,
            pad_right=not feat_spec.pad_on_left,
        )
        token_idx_to_char_idx_end = pad_to_max_seq_length(
            ls=[-1] * unpadded_inputs.cls_offset
            + self.token_idx_to_char_idx_map.cumsum(axis=1).argmax(axis=1).tolist(),
            max_seq_length=feat_spec.max_seq_length,
            pad_idx=-1,
            pad_right=not feat_spec.pad_on_left,
        )
        # when there are multiple greatest elements, argmax will return the index of the first one
        # so, argmax() will return the index of the first 1 in a 0-1 array
        # and cumsum().argmax() will return the index of the last 1 in a 0-1 array
        return DataRow(
            guid=self.guid,
            input_ids=input_set.input_ids,
            input_mask=input_set.input_mask,
            segment_ids=input_set.segment_ids,
            gt_span_str=self.answer_str,
            gt_span_idxs=gt_span_idxs,
            selection_str=self.passage_str,
            selection_token_mask=pred_span_mask,
            token_idx_to_char_idx_start=token_idx_to_char_idx_start,
            token_idx_to_char_idx_end=token_idx_to_char_idx_end,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    gt_span_str: str
    gt_span_idxs: np.ndarray
    selection_str: str
    selection_token_mask: np.ndarray
    token_idx_to_char_idx_start: np.ndarray
    token_idx_to_char_idx_end: np.ndarray


@dataclass
class Batch(BatchMixin):
    guid: List[str]
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    gt_span_str: List[str]
    gt_span_idxs: np.ndarray
    selection_str: List[str]
    selection_token_mask: np.ndarray
    token_idx_to_char_idx_start: np.ndarray
    token_idx_to_char_idx_end: np.ndarray


class AbstractSpanPredicationTask(Task, ABC):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch
    TASK_TYPE = TaskTypes.SPAN_PREDICTION
