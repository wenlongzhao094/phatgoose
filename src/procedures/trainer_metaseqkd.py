import os
import re
from collections import defaultdict

import gin
import numpy as np
import torch
import copy
import src.utils.logging as logging
from src.procedures.procedure import Procedure
from src.procedures.utils.batcher import MultiTaskBatcher, SingleTaskBatcher
from src.procedures.utils.optimizer_scheduler import get_metaseqkd_optimizers, get_metaseqkd_schedulers
from src.procedures.utils.skd_utils import compute_kd_loss_ce, count_parameters

import learn2learn as l2l

# Make the order consistent and reasonable
@gin.configurable(
    allowlist=[
        "teacher_model",
        "model",
        "datasets",
        "batcher",
        "training_tracker",
        "ilr",
        "num_steps",
        "gradient_accumulation_factor",
        "gradient_clipping",
        "validate_procedure",
        "report_step_interval",
        "validation_step_interval",
        "save_model_step_interval",
        "checkpoint_step_interval",
        "step_moma_calls",
        "report_moma_calls",
        "save_model_moma_calls",
        "finish_moma_calls",
        "pass_current_step",
    ],
)
class TrainerMetaSeqKD(Procedure):
    linking_fields = ["teacher_model", "model", "datasets", "validate_procedure"]

    def __init__(
        self,
        teacher_model,
        model,
        datasets,
        ilr,
        batcher=SingleTaskBatcher(shuffle=True, drop_last=True, num_workers=8),
        num_steps=10000,
        gradient_accumulation_factor=1,
        gradient_clipping=1.0,
        validate_procedure=None,
        report_step_interval=50,
        validation_step_interval=None,
        save_model_step_interval=None,
        checkpoint_step_interval=None,
        step_moma_calls=[],
        report_moma_calls=[],
        save_model_moma_calls=[],
        finish_moma_calls=[],
        pass_current_step=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.model = model
        self.datasets = datasets
        self.validate_procedure = validate_procedure

        self.num_steps = num_steps
        self.gradient_accumulation_factor = gradient_accumulation_factor
        self.gradient_clipping = gradient_clipping

        self.batcher = batcher
        self.ilr = ilr
        self.loss_scaler = None
        self.optimizer = None
        self.teacher_optimizer = None
        self.scheduler = None
        self.teacher_scheduler = None
        self.report_tracker = Tracker()

        self.step_moma_calls = step_moma_calls
        self.report_moma_calls = report_moma_calls
        self.save_model_moma_calls = save_model_moma_calls
        self.finish_moma_calls = finish_moma_calls

        self.report_step_interval = report_step_interval
        self.validation_step_interval = validation_step_interval
        self.save_model_step_interval = save_model_step_interval
        self.checkpoint_step_interval = checkpoint_step_interval

        self.current_step = 0
        self.pass_current_step = pass_current_step

        if self.save_model_step_interval is not None and save_model_moma_calls == []:
            raise NotImplementedError()
            # TODO: provide a default save_model function
        if self.checkpoint_step_interval is not None:
            assert (
                self.save_model_step_interval is not None
                and self.checkpoint_step_interval % self.save_model_step_interval == 0
            ), "checkpoint_step_interval need to be a multiple of save_model_step_interval"

    def link(self):
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        super().link()

    def late_init(self):
        self.teacher_optimizer, self.optimizer = get_metaseqkd_optimizers(
            self.teacher_model, self.model,
        )

        self.teacher_scheduler, self.scheduler = get_metaseqkd_schedulers(
            teacher_optimizer=self.teacher_optimizer,
            optimizer=self.optimizer,
            num_steps=self.num_steps,
        )

        if self.model.mix_precision in ["bf16", "fp16"]:
            self.loss_scaler = torch.cuda.amp.GradScaler()

        if self.teacher_model.mix_precision in ["bf16", "fp16"]:
            self.teacher_loss_scaler = torch.cuda.amp.GradScaler()

        for dataset in self.datasets:
            dataset.set_tokenizer(self.model.tokenizer)

        self.batcher.set_tokenizer(self.model.tokenizer)
        self.batcher.set_seed(self.seed)
        self._data_loader = self.batcher.build(self.datasets)

    def _get_train_batches(self):
        while True:
            for batch_inputs in self._data_loader:
                yield batch_inputs

    def prepare_passing_global_hiddens(self):
        passing_global_hiddens = {}
        if self.pass_current_step:
            passing_global_hiddens["current_step"] = self.current_step
        return passing_global_hiddens

    def run(self):
        best_result = None
        logging.print_single_bar()
        print(f"Running {self.name}...")
        self.model.torch_model.train()
        data_iter = self._get_train_batches()


        # TODO: Jay: Make ilr gin configurable
        student_model_features = l2l.algorithms.MAML(self.model, lr=self.ilr,
                                                     allow_nograd=True)

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
            while self.current_step < self.num_steps:
                self.optimizer.zero_grad()
                task_model = student_model_features.clone()
                for _ in range(self.gradient_accumulation_factor):
                    batch_inputs = next(data_iter)
                    # TODO: Assuming all datasets have the same interface during training i.e lm
                    batch_dataset = self.datasets[0]
                    total_size = len(batch_inputs['input_ids'])
                    split_index = int(total_size * 0.7)
                    # Split the batch_inputs into support and query sets
                    support_set = {key: value[:split_index] for key, value in batch_inputs.items()}
                    query_set = {key: value[split_index:] for key, value in batch_inputs.items()}

                    # Check trainable params in the student and teacher model
                    # count_parameters(self.model.torch_model)
                    # count_parameters(self.teacher_model.torch_model)

                    # Use the teacher model to generate tokens and use them as ground truth labels for the student model
                    # deepcopy batch_dataset to a new variable
                    batch_dataset_override = copy.deepcopy(batch_dataset)
                    batch_dataset_override.interface_info.interface = 'gen'
                    # batch_dataset_override.
                    # teacher_interface_override = 'gen_4encdec' if 't5' in self.model.model_name_or_path else 'gen_4dec'
                    # teacher_interface_override = 'gen_4encdec'
                    # print("Print teacher trainable params")
                    # count_parameters(self.teacher_model.torch_model)
                    # bb
                    with torch.no_grad():
                        teacher_batch_outputs = self.teacher_model(support_set, batch_dataset_override.interface_info, {})

                    # Modify the teacher_batch_outputs['sequences'] by removing the first token from each sequence
                    teacher_batch_outputs['output_ids'] = [sequence[1:] for sequence in teacher_batch_outputs['output_ids']]
                    teacher_batch_outputs['output_ids'] = torch.stack(teacher_batch_outputs['output_ids'], dim=0)
                    # print("teacher_batch_outputs['output_ids'][0]: ", teacher_batch_outputs['output_ids'][:5])


                    # Modify the target_ids in batch_inputs by replacing them with the teacher_batch_outputs['sequences']
                    # print("Before: batch_inputs['target_ids'][0]: ", batch_inputs['target_ids'][:5])
                    batch_inputs['target_ids'] = teacher_batch_outputs['output_ids'].cpu().detach()
                    # print("After: batch_inputs['target_ids'][0]: ", batch_inputs['target_ids'][:5])



                    # print(support_set['input_ids'])
                    # print("Trainer size: ", support_set['input_ids'].size())
                    batch_outputs = task_model(
                        support_set,
                        batch_dataset.interface_info,
                        self.prepare_passing_global_hiddens(),
                    )


                    student_loss = batch_outputs["loss"]

                    # Compute KD Loss
                    kd_loss = compute_kd_loss_ce(
                        student_logits=batch_outputs["logits"],
                        teacher_logits=teacher_batch_outputs["logits"],
                        temperature=2.0
                    )

                    inner_loss = (student_loss + kd_loss)/2


                    scaled_inner_loss = inner_loss / self.gradient_accumulation_factor
                    if self.loss_scaler is not None:
                        scaled_inner_loss = self.loss_scaler.scale(scaled_inner_loss)

                    # scaled_inner_loss.backward()
                    task_model.adapt(scaled_inner_loss)

                    lm_head_outputs = task_model(query_set,
                                                 batch_dataset.interface_info,
                                                 self.prepare_passing_global_hiddens())

                    lm_head_loss = lm_head_outputs["loss"]

                    scaled_lm_head_loss = lm_head_loss / self.gradient_accumulation_factor

                    if self.loss_scaler is not None:
                        scaled_lm_head_loss = self.loss_scaler.scale(scaled_lm_head_loss)
                    scaled_lm_head_loss.backward(retain_graph=True)

                    self.report_tracker.add(
                        inner_loss=inner_loss,
                        loss=lm_head_loss,
                        global_hidden_dict=self.model.global_hidden_dict,
                    )


                if self.loss_scaler is not None:
                    self.loss_scaler.unscale_(self.optimizer)

                # if self.teacher_loss_scaler is not None:
                #     self.teacher_loss_scaler.unscale_(self.teacher_optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.named_trainable_parameters().values(),
                    self.gradient_clipping,
                    error_if_nonfinite=False,
                )

                teacher_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.teacher_model.named_trainable_parameters().values(),
                    self.gradient_clipping,
                    error_if_nonfinite=False,
                )

                self.report_tracker.add(
                    teacher_grad_norm=teacher_grad_norm,
                    grad_norm=grad_norm,
                    teacher_lr=self.teacher_optimizer.param_groups[0]["lr"],
                    lr=self.optimizer.param_groups[0]["lr"],
                )

                for moma_call in self.step_moma_calls:
                    moma_call(self.model)

                # if self.teacher_loss_scaler is not None:
                #     self.teacher_loss_scaler.step(self.teacher_optimizer)
                #     self.teacher_loss_scaler.update()
                # else:
                #     self.teacher_optimizer.step()
                self.teacher_optimizer.step()

                if self.loss_scaler is not None:
                    self.loss_scaler.step(self.optimizer)
                    self.loss_scaler.update()
                else:
                    self.optimizer.step()


                if self.teacher_scheduler is not None:
                    self.teacher_scheduler.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.current_step += 1

                logging.logger_step()

                if self.current_step % self.report_step_interval == 0:
                    for moma_call in self.report_moma_calls:
                        moma_call(self.model)
                    report = self.report_tracker.get_summary()
                    logging.log_scalar_dict(report)
                    print(f"\tStep {logging.global_step}: {report}")

                if (
                    self.validation_step_interval
                    and self.current_step % self.validation_step_interval == 0
                ):
                    self.validate_procedure.run(logging.global_step)
                    logging.print_single_bar()
                    print(f"Return to {self.name}...")
                    self.model.torch_model.train()
                if (
                    self.save_model_step_interval
                    and self.current_step % self.save_model_step_interval == 0
                ):
                    for moma_call in self.save_model_moma_calls:
                        moma_call(self.model)
                if (
                    self.checkpoint_step_interval
                    and self.current_step % self.checkpoint_step_interval == 0
                ):
                    self.save_states()

            print(f"Best result: {best_result}")
            print(f"Finished {self.name}")
            for moma_call in self.finish_moma_calls:
                moma_call(self.model)

    def save_states(self, checkpoint_path):
        pass

    def recover_states(self, checkpoint_path):
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Train {self.model.name} model on {len(self.datasets)} datasets ({[dataset.name for dataset in self.datasets]})",
            f"Optimizer: {self.optimizer}",
            f"Scheduler: {self.scheduler}",
            f"{self.num_steps} step x {self.gradient_accumulation_factor} grad accumulations",
        ]


@gin.configurable(allowlist=["global_hidden_to_keep"])
class Tracker:
    def __init__(self, global_hidden_to_keep=["inner_loss", "loss", "scale"]):
        self.global_hidden_to_keep = global_hidden_to_keep
        self._results = defaultdict(list)

    def add(self, inner_loss=None, loss=None, global_hidden_dict=None, teacher_grad_norm=None, grad_norm=None, teacher_lr=None, lr=None):
        if inner_loss is not None:
            self._results["inner_loss"].append(inner_loss.detach().cpu().item())
        if loss is not None:
            self._results["loss"].append(loss.detach().cpu().item())
        if teacher_grad_norm is not None:
            self._results["teacher_grad_norm"].append(teacher_grad_norm.detach().cpu().item())
        if grad_norm is not None:
            self._results["grad_norm"].append(grad_norm.detach().cpu().item())
        if teacher_lr is not None:
            self._results["teacher_lr"].append(teacher_lr)
        if lr is not None:
            self._results["lr"].append(lr)
        if global_hidden_dict is not None:
            for key, value in global_hidden_dict.items():
                if key[0] in self.global_hidden_to_keep:
                    report_key = f"{key[0]}/{'.'.join(key[1:])}"
                    self._results[report_key].append(value.detach().cpu().item())

    def get_summary(self, clear=True):
        summary = {}
        for key, value in self._results.items():
            value = np.mean(value).item()
            if value > 1e-4:
                summary[key] = round(value, 4)
            else:
                summary[key] = value
        if clear:
            self._results.clear()
        return summary
