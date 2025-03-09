import json
import os
from collections import OrderedDict

import gin
import torch

import src.utils.logging as logging
from src.data.metrics import Scorer
from src.procedures.procedure import Procedure
from src.procedures.utils.batcher import SingleTaskBatcher
from src.procedures.utils.result_aggregators import MainAggregator


@gin.configurable(
    allowlist=[
        "teacher_model",
        "model",
        "datasets",
        "adapt_datasets",
        "save_results",
        "batcher",
        "num_steps",
        "results_aggregators",
        "analysis_processors",
        "higher_is_better",
        "better_model_moma_calls",
        "adapt_examples",
        "ilr"
    ]
)
# Set the random seed
class EvaluatorMetaSeqKD(Procedure):
    linking_fields = ["model", "datasets", "adapt_datasets"]

    def __init__(
        self,
        model,
        datasets,
        adapt_datasets,
        save_results,
        batcher=SingleTaskBatcher(shuffle=False, drop_last=False, num_workers=8),
        num_steps=10000,
        results_aggregators=[MainAggregator()],
        analysis_processors=[],
        higher_is_better=True,
        better_model_moma_calls=[],
        adapt_examples=32,
        ilr=1e-6,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.datasets = datasets
        self.adapt_datasets = adapt_datasets
        self.batcher = batcher
        self.loss_scaler = None
        self.optimizer = None
        self.scheduler = None
        self.num_steps = num_steps
        self.save_results = save_results
        self.results_aggregators = results_aggregators
        self._current_results = OrderedDict()
        self.scorer = OrderedDict()
        self.analysis_processors = analysis_processors
        self.higher_is_better = higher_is_better
        self.better_model_moma_calls = better_model_moma_calls
        self.best_results = None
        self.adapt_examples = adapt_examples
        self.ilr = ilr
        self._adapt_dataloader = self.batcher.build(self.adapt_datasets)

    def link(self):
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]
        super().link()

    def late_init(self):
        self.optimizer = get_optimizer(
            self.model,
        )
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            num_steps=self.num_steps,
        )
        if self.model.mix_precision in ["bf16", "fp16"]:
            self.loss_scaler = torch.cuda.amp.GradScaler()
        for dataset in self.datasets:
            dataset.set_tokenizer(self.model.tokenizer)
            if dataset.metrics is not None:
                self.scorer[dataset.name] = Scorer(dataset.metrics)
        for dataset in self.adapt_datasets:
            dataset.set_tokenizer(self.teacher_model.tokenizer)

        self.batcher.set_tokenizer(self.model.tokenizer)
        self.batcher.set_seed(self.seed)

    def _get_train_batches(self):
        while True:
            for batch_inputs in self._adapt_dataloader:
                yield batch_inputs

    def run(self, step=None):
        logging.print_single_bar()


        ### Add logic for adaptation step
        print(f"Running adaptation step...")
        self.model.torch_model.train()
        data_iter = self._get_train_batches()

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
                    split_index = total_size
                    # Split the batch_inputs into support and query sets
                    support_set = {key: value[:split_index] for key, value in batch_inputs.items()}

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
                        teacher_batch_outputs = self.teacher_model(support_set, batch_dataset_override.interface_info,
                                                                   {})

                    # Modify the teacher_batch_outputs['sequences'] by removing the first token from each sequence
                    teacher_batch_outputs['output_ids'] = [sequence[1:] for sequence in
                                                           teacher_batch_outputs['output_ids']]
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

                    inner_loss = (student_loss + kd_loss) / 2

                    scaled_inner_loss = inner_loss / self.gradient_accumulation_factor
                    if self.loss_scaler is not None:
                        scaled_inner_loss = self.loss_scaler.scale(scaled_inner_loss)

                    # scaled_inner_loss.backward()
                    task_model.adapt(scaled_inner_loss)

                if self.loss_scaler is not None:
                    self.loss_scaler.unscale_(self.optimizer)

                # if self.teacher_loss_scaler is not None:
                #     self.teacher_loss_scaler.unscale_(self.teacher_optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.named_trainable_parameters().values(),
                    self.gradient_clipping,
                    error_if_nonfinite=False,
                )

                for moma_call in self.step_moma_calls:
                    moma_call(self.model)

                if self.loss_scaler is not None:
                    self.loss_scaler.step(self.optimizer)
                    self.loss_scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()


        print(f"Running {self.name}...")
        self.model.torch_model.eval()
        with torch.no_grad():
            for dataset in self.datasets:
                print(f"\tEvaluating {dataset.name}...")
                if dataset.name in self._current_results:
                    continue
                if dataset.metrics is None:
                    continue
                data_loader = self.batcher.build(dataset)
                for batch_idx, batch_inputs in enumerate(data_loader):
                    batch_outputs = self.model(batch_inputs, dataset.interface_info, {})
                    self.scorer[dataset.name].add_batch(batch_inputs, batch_outputs)
                    for analysis_processor in self.analysis_processors:
                        analysis_processor.batch_process(
                            batch_inputs, batch_outputs, self.model.global_hidden_dict
                        )

                self._current_results[dataset.name] = self.scorer[
                    dataset.name
                ].get_score()
                self._current_results[dataset.name]["score"] = sum(
                    self._current_results[dataset.name].values()
                ) / len(self._current_results[dataset.name])
                self.save_states()

                for analysis_processor in self.analysis_processors:
                    analysis_processor.dataset_process(dataset.name)
                print(
                    f"\t{dataset.name} results: {self._current_results[dataset.name]}"
                )

        for aggregator in self.results_aggregators:
            aggregator(self._current_results)
        results = self._current_results.copy()
        if step is not None:
            results["step"] = step
        self._current_results.clear()

        print(f"\tAll results: {results}")
        print(f"Finished {self.name}")

        logging.log_scalar_dict(
            {f"{self.name}/{key}": value for key, value in results.items()}
        )
        self.save_results(results, step=step)
        for analysis_processor in self.analysis_processors:
            analysis_processor.cross_dataset_process()
            analysis_processor.save(step)

        if (
            self.best_results is None
            or (
                results["average_score"] >= self.best_results["average_score"]
                and self.higher_is_better
            )
            or (
                results["average_score"] <= self.best_results["average_score"]
                and not self.higher_is_better
            )
        ):
            print("\t New best results!")
            self.best_results = results
            for moma_call in self.better_model_moma_calls:
                moma_call(self.model)
        return results

    def save_states(self):
        # TODO(Checkpointing): save results and rng state
        pass

    def recover_states(self):
        # TODO(Checkpointing): load results and rng state
        pass

    def get_description(self):
        return [
            f"Procedure class: {self.__class__.__name__}",
            f"Evalutes {self.model.name} model on {len(self.datasets)} datasets ({[dataset.name for dataset in self.datasets]})",
        ]
