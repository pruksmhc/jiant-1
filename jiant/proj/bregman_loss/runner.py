import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from pyutils.display import maybe_tqdm, maybe_trange
from nlpr.shared.runner import (
    BaseRunner,
    convert_examples_to_dataset,
    HybridLoader,
    complex_backpropagate,
    get_sampler,
    TrainGlobalState,
    optim_step_grad_accum,
)

@dataclass
class MeanTeacherParameters:
    alpha: float
    consistency_type: str
    consistency_weight: float
    consistency_ramp_up_steps: int
    use_unsup: bool
    unsup_ratio: int


@dataclass
class TrainDataDuplet:
    sup: Any
    unsup: Any


def create_teacher(model_wrapper: model_setup.ModelWrapper) -> model_setup.ModelWrapper:
    return model_setup.ModelWrapper(
        model=copy.deepcopy(model_wrapper.model),
        tokenizer=model_wrapper.tokenizer,
    )


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(global_step, mt_params: MeanTeacherParameters):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # Maybe do epochs?
    return (
        mt_params.consistency_weight
        * sigmoid_rampup(global_step, mt_params.consistency_ramp_up_steps)
    )


def softmax_mse_loss(input_logits, target_logits):
    # From https://github.com/CuriousAI/mean-teacher/
    #       blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/losses.py
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_kl_loss(input_logits, target_logits):
    # From https://github.com/CuriousAI/mean-teacher/
    #       blob/bd4313d5691f3ce4c30635e50fa207f49edf16fe/pytorch/mean_teacher/losses.py
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def compute_raw_consistency_loss(student_logits, teacher_logits, mt_params: MeanTeacherParameters):
    if mt_params.consistency_type == "kl":
        raw_consistency_loss = softmax_kl_loss(
            input_logits=student_logits,
            target_logits=teacher_logits,
        )
    elif mt_params.consistency_type == "mse":
        raw_consistency_loss = softmax_mse_loss(
            input_logits=student_logits,
            target_logits=teacher_logits,
        )
    else:
        raise KeyError(mt_params.consistency_type)
    return raw_consistency_loss


class MeanTeacherRunner(BaseRunner):
    def __init__(self, task, model_wrapper, jiant_task_container: JiantTaskContainer, optimizer_scheduler,  loss_criterion,
                 device, rparams: simple_runner.RunnerParameters, mt_params: MeanTeacherParameters, 
                 log_writer):
        self.task = task
        self.model_wrapper = model_wrapper
        self.optimizer_scheduler = optimizer_scheduler
        self.teacher_model_wrapper = create_teacher(model_wrapper) 
        self.loss_criterion = loss_criterion
        self.device = device
        self.rparams = rparams
        self.mt_params = mt_params
        self.log_writer = log_writer

        # Convenience
        self.model = self.model_wrapper.model

    def run_train(self, task_data, verbose=True):

        dataloader_duplet = self.get_train_dataloaders(
            task_data=task_data,
            verbose=verbose,
        )
        train_global_state = TrainGlobalState()
        for epoch_i in \
                maybe_trange(int(self.train_schedule.num_train_epochs), desc="Epoch", verbose=verbose):
            train_global_state.epoch = epoch_i
            self.run_train_epoch(dataloader_duplet, train_global_state)
            results = self.run_val(val_examples=self.task.get_val_examples())
            self.log_writer.write_entry("val_metric", {
                "epoch": train_global_state.epoch,
                "metric": results["metrics"].asdict(),
            })
            self.log_writer.flush()

    def run_train_epoch(self,
                        dataloader_duplet: TrainDataDuplet,
                        train_global_state: TrainGlobalState, verbose=True):
        for _ in self.run_train_epoch_context(
                dataloader_duplet=dataloader_duplet,
                train_global_state=train_global_state,
                verbose=verbose):
            pass

    def run_train_epoch_context(self,
                                dataloader_duplet: TrainDataDuplet,
                                train_global_state: TrainGlobalState, verbose=True):
        self.teacher_model_wrapper.requires_grad = False
        train_iterator = maybe_tqdm(zip(
            dataloader_duplet.sup,
            dataloader_duplet.unsup,
        ), desc="Training", verbose=verbose, total=len(dataloader_duplet.sup))

        for sup_batch, unsup_batch in train_iterator:
            train_dataloader_dict = TrainDataDuplet(
                sup=sup_batch,
                unsup=unsup_batch,
            )
            self.run_train_step(
                train_dataloader_dict=batch_duplet,
                train_global_state=train_global_state,
            )
            yield train_dataloader_dict, train_global_state
        train_global_state.step_epoch()

    def run_train_context(self, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        train_state = TrainState.from_task_name_list(
            self.jiant_task_container.task_run_config.train_task_list
        )
        for _ in maybe_tqdm(
            range(self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            verbose=verbose,
        ):
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state
            )
            yield train_state

   def resume_train_context(self, train_state, verbose=True):
        train_dataloader_dict = self.get_train_dataloader_dict()
        start_position = train_state.global_steps
        for _ in maybe_tqdm(
            range(start_position, self.jiant_task_container.global_train_config.max_steps),
            desc="Training",
            initial=start_position,
            total=self.jiant_task_container.global_train_config.max_steps,
            verbose=verbose,
        ):
            self.run_train_step(
                train_dataloader_dict=train_dataloader_dict, train_state=train_state
            )
            yield train_state

    def get_runner_state(self):
        # TODO: Add fp16  (Issue #46)
        state = {
            "model": torch_utils.get_model_for_saving(self.jiant_model).state_dict(),
            "optimizer": self.optimizer_scheduler.optimizer.state_dict(),
        }
        return state


    def run_train_step(self,  train_dataloader_dict: dict, train_state: TrainState):
        self.model.train()

        sup_batch = train_dataloader_dict.sup.to(self.device)

        # Classification [SUP]
        sup_logits = forward_batch_delegate(
            model=self.model,
            batch=sup_batch.batch,
            omit_label_ids=True,
            task_type=self.task.TASK_TYPE,
        )[0]
        classification_loss = compute_loss_from_model_output(
            logits=sup_logits,
            loss_criterion=self.loss_criterion,
            batch=sup_batch.batch,
            task_type=self.task.TASK_TYPE,
        )
        # Consistency
        with torch.no_grad():
            teacher_sup_logits = forward_batch_delegate(
                model=self.teacher_model_wrapper.model,
                batch=sup_batch.batch,
                omit_label_ids=True,
                task_type=self.task.TASK_TYPE,
            )[0]

        # Consistency
        if self.mt_params.use_unsup:
            unsup_batch = train_dataloader_dict.unsup.to(self.device)
            unsup_logits = forward_batch_delegate(
                model=self.model,
                batch=unsup_batch.batch,
                omit_label_ids=True,
                task_type=self.task.TASK_TYPE,
            )[0]
            teacher_unsup_logits = forward_batch_delegate(
                model=self.teacher_model_wrapper.model,
                batch=unsup_batch.batch,
                omit_label_ids=True,
                task_type=self.task.TASK_TYPE,
            )[0]
            student_logits = torch.cat([sup_logits, unsup_logits], dim=0)
            teacher_logits = torch.cat([teacher_sup_logits, teacher_unsup_logits], dim=0)
        else:
            student_logits = sup_logits
            teacher_logits = teacher_sup_logits

        raw_consistency_loss = compute_raw_consistency_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mt_params=self.mt_params,
        )
        consistency_weight = get_current_consistency_weight(
            global_step=train_global_state.global_step,
            mt_params=self.mt_params,
        )
        consistency_loss = consistency_weight * raw_consistency_loss

        # Combine
        loss = classification_loss + consistency_loss
        loss = self.complex_backpropagate(loss)

        optim_step_grad_accum(
            optimizer_scheduler=self.optimizer_scheduler,
            train_global_state=train_global_state,
            gradient_accumulation_steps=self.train_schedule.gradient_accumulation_steps,
        )
        self.teacher_model_wrapper = self.model_wrapper
        self.teacher_model_wrapper.requires_grad = False
        self.log_writer.write_entry("loss_train", {
            "epoch": train_global_state.epoch,
            "epoch_step": train_global_state.epoch_step,
            "global_step": train_global_state.global_step,
            "classification_loss": classification_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "total_loss": loss.item(),
            "pred_entropy": compute_pred_entropy_clean(sup_logits)
        })

    def _get_eval_dataloader_dict(self, phase, task_name_list, use_subset=False):
        val_dataloader_dict = {}
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            eval_cache = self.jiant_task_container.task_cache_dict[task_name][phase]
            task_specific_config = self.jiant_task_container.task_specific_configs[task_name]
            val_dataloader_dict[task_name] = get_eval_dataloader_from_cache(
                eval_cache=eval_cache,
                task=task,
                eval_batch_size=task_specific_config.eval_batch_size,
                subset_num=task_specific_config.eval_subset_num if use_subset else None,
            )
        return val_dataloader_dict

    def get_val_dataloader_dict(self, task_name_list, use_subset=False):
        return self._get_eval_dataloader_dict(
            phase="val", task_name_list=task_name_list, use_subset=use_subset,
        )


    def run_val(self, task_name_list, use_subset=None, return_preds=False, verbose=True):
        evaluate_dict = {}
        val_dataloader_dict = self.get_val_dataloader_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        val_labels_dict = self.get_val_labels_dict(
            task_name_list=task_name_list, use_subset=use_subset
        )
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_val(
                val_dataloader=val_dataloader_dict[task_name],
                val_labels=val_labels_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                return_preds=return_preds,
                verbose=verbose,
            )
        return evaluate_dict

    def run_test(self, task_name_list, verbose=True):
        evaluate_dict = {}
        test_dataloader_dict = self.get_test_dataloader_dict()
        for task_name in task_name_list:
            task = self.jiant_task_container.task_dict[task_name]
            evaluate_dict[task_name] = run_test(
                test_dataloader=test_dataloader_dict[task_name],
                jiant_model=self.jiant_model,
                task=task,
                device=self.device,
                local_rank=self.rparams.local_rank,
                verbose=verbose,
            )
        return evaluate_dict


    def run_test(self, test_examples, verbose=True):
        test_dataloader = self.get_eval_dataloader(test_examples)
        self.model.eval()
        all_logits = []
        for step, (batch, batch_metadata) in enumerate(
                maybe_tqdm(test_dataloader, desc="Predictions (Test)", verbose=verbose)):
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = forward_batch_delegate(
                    model=self.model,
                    batch=batch,
                    omit_label_ids=True,
                    task_type=self.task.TASK_TYPE,
                )[0]
            logits = logits.detach().cpu().numpy()
            all_logits.append(logits)

        all_logits = np.concatenate(all_logits, axis=0)
        return all_logits


    def get_train_dataloader_dict(self):
        # Not currently supported distributed parallel
        train_dataloader_dict = {}
        for task_name in self.jiant_task_container.task_run_config.train_task_list:
            task = self.jiant_task_container.task_dict[task_name]
            train_cache = self.jiant_task_container.task_cache_dict[task_name]["train"]
            train_batch_size = self.jiant_task_container.task_specific_configs[
                task_name
            ].train_batch_size
            train_dataloader_dict[task_name] = InfiniteYield(
                get_train_dataloader_from_cache(
                    train_cache=train_cache, task=task, train_batch_size=train_batch_size,
                )
            )
        return train_dataloader_dict

    def get_sup_train_dataloader(self, task_data, verbose=True):
        return self.get_single_train_dataloader(
            train_examples=task_data["sup"]["train"],
            verbose=verbose,
            batch_size=self.train_schedule.train_batch_size
        )


def run_val(
    val_dataloader,
    val_labels,
    jiant_model: JiantModel,
    task,
    device,
    local_rank,
    return_preds=False,
    verbose=True,
):
    # Reminder:
    #   val_dataloader contains mostly PyTorch-relevant info
    #   val_labels might contain more details information needed for full evaluation
    if not local_rank == -1:
        return
    jiant_model.eval()
    total_eval_loss = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(val_dataloader, desc=f"Eval ({task.name}, Val)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=True,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        batch_loss = model_output.loss.mean().item()
        total_eval_loss += batch_loss
        eval_accumulator.update(
            batch_logits=batch_logits,
            batch_loss=batch_loss,
            batch=batch,
            batch_metadata=batch_metadata,
        )

        nb_eval_examples += len(batch)
        nb_eval_steps += 1
    eval_loss = total_eval_loss / nb_eval_steps
    tokenizer = (
        jiant_model.tokenizer
        if not torch_utils.is_data_parallel(jiant_model)
        else jiant_model.module.tokenizer
    )
    output = {
        "accumulator": eval_accumulator,
        "loss": eval_loss,
        "metrics": evaluation_scheme.compute_metrics_from_accumulator(
            task=task, accumulator=eval_accumulator, labels=val_labels, tokenizer=tokenizer,
        ),
    }
    if return_preds:
        output["preds"] = evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        )
    return output


def run_test(test_dataloader, jiant_model: JiantModel, task, device, local_rank, verbose=True):
    if not local_rank == -1:
        return
    jiant_model.eval()
    evaluation_scheme = evaluate.get_evaluation_scheme_for_task(task=task)
    eval_accumulator = evaluation_scheme.get_accumulator()

    for step, (batch, batch_metadata) in enumerate(
        maybe_tqdm(test_dataloader, desc=f"Eval ({task.name}, Test)", verbose=verbose)
    ):
        batch = batch.to(device)

        with torch.no_grad():
            model_output = wrap_jiant_forward(
                jiant_model=jiant_model, batch=batch, task=task, compute_loss=False,
            )
        batch_logits = model_output.logits.detach().cpu().numpy()
        eval_accumulator.update(
            batch_logits=batch_logits, batch_loss=0, batch=batch, batch_metadata=batch_metadata,
        )
    return {
        "preds": evaluation_scheme.get_preds_from_accumulator(
            task=task, accumulator=eval_accumulator,
        ),
        "accumulator": eval_accumulator,
    }
