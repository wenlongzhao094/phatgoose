import math
import uuid

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from src.models.addons.addon import Addon
from src.utils.constants import BOOL_PLACEHOLDER, LIST_PLACEHOLDER


class MoELink:
    """
    An MoE group consists of some number of moe layers and a router shared among them.
    This class helps to connect the router and moe_layers together while avoid pytorch automatic registering submodules.
    In particular, this class maintains an inventory: the list expert_identifiers tracks the meaning of each expert present in the router and moe layers.
    """

    def __init__(self):
        self.router = None
        self.moe_layers = []
        self.expert_identifiers = []

    def set_router(self, router):
        self.router = router

    def add_moe_layer(self, moe_layer):
        self.moe_layers.append(moe_layer)

    def extend(self, num_new_experts, identifier_stem):
        start_number = 0
        for key_number in self.expert_identifiers[::-1]:
            key_number_split = key_number.split("_")
            number = int(key_number_split[-1])
            key = "_".join(key_number_split[:-1])
            if key == identifier_stem:
                start_number = number + 1
                break
        for number in range(start_number, start_number + num_new_experts):
            self.expert_identifiers.append(f"{identifier_stem}_{number}")


@gin.configurable
class ExtendableAddon(Addon):
    """
    An addon that can be extended to more experts.
    The _extendable_parameters is a list of the names of the parameters that can be extended.
    This class helps manage extending the number of experts, as well as saving and loading the extended parameters.
    All ExtendableAddon has two working modes: separate_experts and not separate_experts.
        1. In separate_experts mode, the parameters are separated by experts. When doing forward, we stack the experts to create a single tensor for better parallization.
        2. In not separate_experts mode, the parameter of all experts are always stored in a single tensor. And the single tensor is split into separate experts in state_dict,
        for the convenience of model management systems (e.g. git-theta).
    """

    _extendable_parameters = None

    def __init__(
        self,
        global_hidden_dict,
        moe_link,
        separate_experts=False,
        pretrained_component=False,
        debug=False,
    ):
        super().__init__(global_hidden_dict)
        self.moe_link = moe_link
        self.separate_experts = separate_experts
        self.pretrained_component = pretrained_component
        self.debug = debug

        if not self.separate_experts:
            for parameter_name in self._extendable_parameters:
                self.register_parameter(parameter_name, nn.UninitializedParameter())

    def _get_init_weights(self, num_new_experts):
        """
        Get the initial weights for the extendable parameters, as a dict of tensors.
        """
        raise NotImplementedError()

    def extend(self, num_new_experts, weight_init="from_scratch"):
        """
        Extend the number of experts.
        Args:
            num_new_experts: int, the number of new experts to be added
            weight_init: str, "from_scratch" or "average"
        """
        with torch.no_grad():
            from_uninitialized = self.num_experts == 0
            if self.separate_experts:
                self._assemble_extendable_parameters()
            if from_uninitialized:
                assert (
                    weight_init == "from_scratch"
                ), """weight_init should be "from_scratch" when extending from uninitialized"""
            else:
                from_uninitialized = False

            if weight_init == "from_scratch":
                new_parameter_data_dict = self._get_init_weights(num_new_experts)

            assert (
                self._extendable_parameters is not None
            ), f"self._extendable_parameters is not specified in {self.__class__.__name__}"
            for parameter_name in self._extendable_parameters:
                parameter = getattr(self, parameter_name)
                if weight_init == "average":
                    if self.pretrained_component:
                        new_parameter_data = (
                            parameter[1:]
                            .data.mean(dim=0, keepdim=True)
                            .repeat(
                                num_new_experts, *([1] * (parameter.data.dim() - 1))
                            )
                        )
                    else:
                        new_parameter_data = parameter.data.mean(
                            dim=0, keepdim=True
                        ).repeat(num_new_experts, *([1] * (parameter.data.dim() - 1)))
                elif weight_init == "from_scratch":
                    new_parameter_data = new_parameter_data_dict[parameter_name]
                if from_uninitialized and not self.separate_experts:
                    parameter.materialize(new_parameter_data.size())
                if self.separate_experts:
                    new_expert_identifiers = self.moe_link.expert_identifiers[
                        -num_new_experts:
                    ]
                    for idx, identifier in enumerate(new_expert_identifiers):
                        setattr(
                            self,
                            f"{parameter_name}_{identifier}",
                            nn.Parameter(new_parameter_data[idx]),
                        )
                else:
                    new_parameter_data = torch.cat(
                        [parameter.data, new_parameter_data], dim=0
                    )
                    parameter.data = new_parameter_data

            self.num_experts += num_new_experts

    def _assemble_extendable_parameters(self):
        """
        Assemble the extendable parameters. This function is only needed under separate_experts=True.
        There are some cases where reassemble is unnecessary, but we do it all the time for simplicity.
        """
        for parameter_name in self._extendable_parameters:
            if self.num_experts == 0:
                assembled_parameter = None
            else:
                per_expert_parameters = [
                    getattr(self, f"{parameter_name}_{identifier}")
                    for identifier in self.moe_link.expert_identifiers[
                        : self.num_experts
                    ]
                ]
                assembled_parameter = torch.stack(per_expert_parameters, dim=0)
            setattr(
                self,
                parameter_name,
                assembled_parameter,
            )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self.separate_experts:
            super()._save_to_state_dict(destination, prefix, keep_vars)
        else:
            raise NotImplementedError()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.separate_experts:
            if len(self.moe_link.expert_identifiers) == 0:
                assert (
                    self.num_experts == 0
                ), "num_experts should be 0 when expert_identifiers is empty"
                # clear prefix names from keys in state_dict to get identifiers (key_numbers) since parameter name are "self._extendable_parameters[0]_identifier"
                prefix_name = prefix + self._extendable_parameters[0] + "_"
                key_numbers = []
                for key in state_dict:
                    if prefix_name in key:
                        key_number = key.replace(prefix_name, "")
                        key_numbers.append(key_number)
                # add the expert numbers to each identifier stem
                key_numbers = set(key_numbers)
                identifier_stems = {}
                for key_number in key_numbers:
                    key_number_split = key_number.split("_")
                    identifier_stem = "_".join(key_number_split[:-1])
                    number = int(key_number_split[-1])
                    if identifier_stem in identifier_stems:
                        identifier_stems[identifier_stem].append(number)
                    else:
                        identifier_stems[identifier_stem] = [number]
                # sort the numbers for each identifier stem
                for identifier_stem in identifier_stems:
                    identifier_stems[identifier_stem] = sorted(
                        identifier_stems[identifier_stem]
                    )
                for identifier_stem in identifier_stems:
                    self.moe_link.expert_identifiers.extend(
                        [
                            f"{identifier_stem}_{number}"
                            for number in identifier_stems[identifier_stem]
                        ]
                    )
            if self.num_experts == 0:
                # when intialized from scratch, self.num_experts is zero even when self.moe_link.expert_identifiers is not empty since self.num_experts is not tied across router and expert layer
                # we extend to create the required keys in the object so that load works
                self.extend(len(self.moe_link.expert_identifiers))

            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
        else:
            raise NotImplementedError()


@gin.configurable(
    allowlist=[
        "d_router",
        "router_type",
        "score_type",
        "epsilon",
        "expert_dropout",
        "scaling_scores",
        "elementwise_affine",
        "orthogonal_penalty_weight",
        "orthogonal_type",
        "write_orthogonal_penalty_key",
        "tag_penalty_weight",
        "tag_penalty_type",
        "tag_supervision_ratio",
        "write_tag_penalty_key",
        "entropy_penalty_weight",
        "write_entropy_penalty_key",
        "temperature",
        "anneal_step",
        "anneal_rate",
        "subspace_penalty_type",
        "write_subspace_penalty_key",
        "subspace_penalty_weight",
        "ol_aggregate_type",
        "router_norm_type",
        "remove_pretrained_method",
        "mask_out_zero_expert",
        "trainable_index",
        "position",
        "is_retriever",
    ],
)
class Router(ExtendableAddon):
    """
    A router that can be extended to more experts.
    """

    has_pre_forward = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER
    _extendable_parameters = ["expert_embeddings"]
    _ref_attr_names = ["d_router"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        read_hidden_key,
        write_routing_weights_key,
        moe_link,
        d_router,
        position="before",
        router_type="smear",
        score_type="original",
        epsilon=1e-6,
        parallel_axis="none",
        expert_dropout=0,
        scaling_scores=True,
        elementwise_affine=False,
        orthogonal_penalty_weight=0,
        orthogonal_type=None,
        write_orthogonal_penalty_key=("loss", "router", "ol"),
        tag_penalty_weight=0,
        tag_penalty_type=None,
        tag_supervision_ratio=0.1,
        write_tag_penalty_key=("loss", "router", "tl"),
        entropy_penalty_weight=0,
        write_entropy_penalty_key=("loss", "router", "el"),
        subspace_penalty_type="remove_orth_component",
        subspace_penalty_weight=0,
        write_subspace_penalty_key=("loss", "router", "sl"),
        temperature=10,
        anneal_step=10,
        anneal_rate=1e-2,
        ol_aggregate_type="mean",
        router_norm_type="layer_norm",
        remove_pretrained_method=None,
        mask_out_zero_expert=False,
        trainable_index=None,
        is_retriever=False,
    ):
        """
        Args:
            global_hidden_dict: dict, the global hidden dict
            read_hidden_key: str, the key of the hidden state to be read by the router ex: ("expose_hiddens", "module_name"/"encoder", "pre", "hidden_states")
            write_routing_weights_key: str, the key of the routing weights to be written by the router ex: ("router", "module_name", "routing_weights")
            moe_link: MoELink, the MoELink object that contains the router and experts
            d_router: int, the dimension of the router
            router_type: str, "smear" or "st_gumbel"
            score_type: str, "dot", "cosine" or "weighted_cosine"
            epsilon: float, the epsilon for cosine and weighted_cosine
            parallel_axis: str, which axis to split when parallelize the module, can be "none", "d_router", or "num_experts"
        """
        super().__init__(global_hidden_dict, moe_link)
        self.moe_link.set_router(self)
        self.read_hidden_key = read_hidden_key
        self.write_routing_weights_key = write_routing_weights_key
        self.num_experts = 0
        self.d_router = d_router
        self.router_type = router_type
        self.score_type = score_type
        self.position = position
        assert self.position in ["before", "after"]
        if self.position == "before":
            self.has_pre_forward = True
            self.has_post_forward = False
        elif self.position == "after":
            self.has_pre_forward = False
            self.has_post_forward = True
        self.expert_dropout = expert_dropout
        assert self.score_type in [
            "dot",
            "cosine",
            "weighted_cosine",
            "original",
            "Arrow",
        ]
        self.epsilon = epsilon
        self.parallel_axis = parallel_axis
        self.scaling_scores = scaling_scores
        self.elementwise_affine = elementwise_affine
        self._resolve_ref_attrs(host_module)
        self.orthogonal_penalty_weight = orthogonal_penalty_weight
        self.orthogonal_type = orthogonal_type
        self.write_orthogonal_penalty_key = write_orthogonal_penalty_key
        self.tag_penalty_weight = tag_penalty_weight
        self.tag_penalty_type = tag_penalty_type
        self.tag_supervision_ratio = tag_supervision_ratio
        self.write_tag_penalty_key = write_tag_penalty_key
        self.entropy_penalty_weight = entropy_penalty_weight
        self.write_entropy_penalty_key = write_entropy_penalty_key
        self.anneal_rate = anneal_rate
        self.anneal_step = anneal_step
        self.temperature = temperature
        self.temperature_schedule = self._get_temperature_schedule(temperature)
        self.subspace_penalty_type = subspace_penalty_type
        self.write_subspace_penalty_key = write_subspace_penalty_key
        self.subspace_penalty_weight = subspace_penalty_weight
        self.ol_aggregate_type = ol_aggregate_type
        self.router_norm_type = router_norm_type
        self.remove_pretrained_method = remove_pretrained_method
        self.mask_out_zero_expert = mask_out_zero_expert
        self.trainable_index = trainable_index
        self.is_retriever = is_retriever
        assert self.tag_penalty_type is None or self.tag_penalty_type in [
            "l1",
            "l2",
            "ce",
        ]
        assert self.orthogonal_type is None or self.orthogonal_type in [
            "l1",
            "l1_relu",
            "l2",
            "l2_relu",
            "remove_sub_component",
        ]

        if self.score_type == "dot":
            pass
        elif self.score_type == "cosine":
            pass
        elif self.score_type == "weighted_cosine":
            self.weights = nn.Parameter(torch.ones(d_router))
        elif self.score_type == "original":
            self.router_layer_norm = nn.LayerNorm(
                self.d_router, elementwise_affine=self.elementwise_affine
            )
            self.input_layer_norm = nn.LayerNorm(
                self.d_router, elementwise_affine=self.elementwise_affine
            )

    def _get_init_weights(self, num_new_experts):
        expert_embeddings = torch.zeros(num_new_experts, self.d_router)
        return {"expert_embeddings": expert_embeddings}

    def _get_temperature_schedule(self, curr_temperature):
        temperature_schedule = {1: curr_temperature}
        for current_step in range(self.anneal_step, 10_000, self.anneal_step):
            curr_temperature = np.maximum(
                np.exp(-self.anneal_rate * current_step) * curr_temperature, 0.5
            )
            temperature_schedule[current_step] = curr_temperature
        return temperature_schedule

    def _gram_schmidt(self, vectors):
        basis = []
        for v in vectors:
            w = v - sum(torch.dot(v, u) / torch.dot(u, u) * u for u in basis)
            if torch.norm(w) > 1e-6:  # Avoid adding zero-length vectors
                basis.append(w)
        orthonormal_basis = [u / torch.norm(u + self.epsilon) for u in basis]
        orthonormal_basis = torch.stack(orthonormal_basis)
        return orthonormal_basis

    def _forward(self, router_hidden_states):
        """
        Args:
            router_hidden_states: (..., d_router)
        Returns:
            routing_weights: (..., num_experts)
        """
        # TODO: (Should we normalize router_hidden_states using LayerNorm? or Is cosine routing good enough to prevent degeneracy?)
        hidden_states_dtype = router_hidden_states.dtype
        if self.separate_experts:
            self._assemble_extendable_parameters()
        if self.is_retriever:
            # (num_experts, num_examples, 384)
            expert_embeddings = self.expert_embeddings.reshape(
                self.num_experts, -1, router_hidden_states.shape[-1]
            )
            expert_embeddings = expert_embeddings / (
                torch.norm(expert_embeddings, dim=-1, keepdim=True) + self.epsilon
            )
            # (num_experts * num_examples, 384)
            expert_embeddings = expert_embeddings.reshape(
                -1, router_hidden_states.shape[-1]
            )
            # (B, 384)
            router_hidden_states = router_hidden_states / (
                torch.norm(router_hidden_states, dim=-1, keepdim=True) + self.epsilon
            )
            routing_scores = torch.matmul(router_hidden_states, expert_embeddings.T)
            routing_scores = routing_scores.reshape(
                router_hidden_states.shape[0], self.num_experts, -1
            )
            routing_scores, _ = routing_scores.max(dim=-1)
            routing_scores, routing_indices = routing_scores.max(dim=-1)
            routing_weights = F.one_hot(routing_indices, num_classes=self.num_experts)
            routing_weights = routing_weights.to(self.expert_embeddings.dtype)
            return routing_weights

        if (
            self.remove_pretrained_method
            and "before_layer_norm" in self.remove_pretrained_method
        ):
            # remove pretrained component from router hidden states
            pretrained_expert_embedding = self.expert_embeddings[0]
            pretrained_expert_embedding_normalized = pretrained_expert_embedding / (
                torch.norm(pretrained_expert_embedding) + self.epsilon
            )
            pretrained_expert_embedding_normalized = (
                pretrained_expert_embedding_normalized.unsqueeze(0)
            )
            if "remove_from_input" in self.remove_pretrained_method:
                router_hidden_states = (
                    router_hidden_states
                    - torch.matmul(
                        router_hidden_states,
                        pretrained_expert_embedding_normalized.T,
                    )
                    * pretrained_expert_embedding_normalized
                )
            self.expert_embeddings = (
                self.expert_embeddings
                - torch.matmul(
                    self.expert_embeddings,
                    pretrained_expert_embedding_normalized.T,
                )
                * pretrained_expert_embedding_normalized
            )
        if self.score_type == "dot" or self.score_type == "Arrow":
            expert_embeddings = self.expert_embeddings
        elif self.score_type in ["cosine", "weighted_cosine"]:
            router_hidden_states = router_hidden_states / (
                torch.norm(router_hidden_states, dim=-1, keepdim=True) + self.epsilon
            )
            expert_embeddings = self.expert_embeddings / (
                torch.norm(self.expert_embeddings, dim=-1, keepdim=True) + self.epsilon
            )
            if self.score_type == "weighted_cosine":
                expert_embeddings = expert_embeddings * self.weights
        elif self.score_type == "original":
            router_hidden_states = self.input_layer_norm(router_hidden_states)
            if self.router_norm_type == "layer_norm":
                expert_embeddings = self.router_layer_norm(self.expert_embeddings)
            elif self.router_norm_type == "l2_norm":
                expert_embeddings = self.expert_embeddings / (
                    torch.norm(self.expert_embeddings, dim=-1, keepdim=True)
                    + self.epsilon
                )
            else:
                expert_embeddings = self.expert_embeddings
        scores = torch.matmul(router_hidden_states, expert_embeddings.T)
        if self.score_type == "Arrow":
            scores = torch.abs(scores / self.temperature)
        if self.scaling_scores:
            scores = scores * math.sqrt(1 / self.d_router)
        if self.mask_out_zero_expert:
            mask = torch.zeros_like(scores)
            mask[:, 0] = float("-inf")
            scores = scores + mask
        if self.router_type in ["smear", "top1"]:
            routing_weights = torch.softmax(scores, dim=-1)
        elif self.router_type == "st-gumbel":
            # TODO(Muqeeth): implement st-gumbel routing
            if self.training:
                U = torch.rand(scores.shape).to(self.config.device)
                modified_scores = scores + (
                    -torch.log(-torch.log(U + self.epsilon) + self.epsilon)
                )
                current_step = self.global_hidden_dict["current_step"]
                if current_step % self.anneal_step == 0:
                    curr_temperature = torch.min(
                        torch.exp(-self.anneal_rate * current_step) * self.temperature,
                        1,
                    )
                    self.temperature = curr_temperature
                routing_weights = F.softmax(modified_scores / self.temperature, dim=-1)
                top1_idx = torch.argmax(routing_weights, dim=-1, keepdim=True)
                top1_mask = torch.zeros_like(routing_weights).scatter_(-1, top1_idx, 1)
                routing_weights = top1_mask * routing_weights
                routing_weights = top1_mask + routing_weights.detach() - routing_weights
            else:
                routing_weights = torch.argmax(scores, dim=-1)
                routing_weights = F.one_hot(
                    routing_weights, num_classes=self.num_experts
                )
        elif self.router_type in ["smear_gumbel", "smear_st_gumbel"]:
            if self.training:
                U = torch.rand(scores.shape).to(scores.device)
                probs = F.softmax(scores, dim=-1)
                log_probs = torch.log(probs + self.epsilon)
                modified_scores = log_probs + (
                    -torch.log(-torch.log(U + self.epsilon) + self.epsilon)
                )
                current_step = self.global_hidden_dict["current_step"] + 1
                if current_step % self.anneal_step == 0:
                    self.temperature = self.temperature_schedule[current_step]
                routing_weights = F.softmax(modified_scores / self.temperature, dim=-1)
                self.global_hidden_dict[
                    ("temperature", "router", "curr_temp")
                ] = torch.tensor(self.temperature)
                if "st" in self.router_type:
                    top1_idx = torch.argmax(routing_weights, dim=-1, keepdim=True)
                    top1_mask = torch.zeros_like(routing_weights).scatter_(
                        -1, top1_idx, 1.0
                    )
                    routing_weights = top1_mask * routing_weights
                    routing_weights = (
                        top1_mask + routing_weights.detach() - routing_weights
                    )
            else:
                if "st" in self.router_type:
                    routing_weights = torch.argmax(scores, dim=-1)
                    routing_weights = F.one_hot(
                        routing_weights, num_classes=self.num_experts
                    )
                    routing_weights = routing_weights.float()
                else:
                    routing_weights = torch.softmax(scores, dim=-1)

        if self.training and self.expert_dropout > 0:
            ones = torch.ones(routing_weights.shape)
            zeros = torch.zeros(routing_weights.shape)
            random_tensor = torch.rand(routing_weights.shape)
            mask = torch.where(random_tensor < self.expert_dropout, zeros, ones)
            mask = mask.to(routing_weights.device)
            routing_weights = routing_weights * mask
            routing_weights = (routing_weights + self.epsilon) / torch.sum(
                (routing_weights + self.epsilon), dim=-1, keepdim=True
            )

        if self.router_type == "top1":
            top1_idx = torch.argmax(routing_weights, dim=-1, keepdim=True)
            top1_mask = torch.zeros_like(routing_weights).scatter_(-1, top1_idx, 1)
            routing_weights = top1_mask * routing_weights
        if self.orthogonal_type and self.orthogonal_penalty_weight > 0:
            if self.orthogonal_type == "remove_sub_component":
                if self.remove_pretrained_method is None:
                    if self.pretrained_component:
                        pretrained_expert_embedding = expert_embeddings[0]
                        pretrained_expert_embedding_normalized = (
                            pretrained_expert_embedding
                            / (torch.norm(pretrained_expert_embedding) + self.epsilon)
                        )
                        expert_embeddings_orthogonal_to_pretrained = []
                        for index in range(1, self.num_experts):
                            expert_embedding_along_pretrained = (
                                torch.matmul(
                                    expert_embeddings[index],
                                    pretrained_expert_embedding_normalized,
                                )
                                * pretrained_expert_embedding_normalized
                            )
                            expert_embedding_orthogonal_to_pretrained = (
                                expert_embeddings[index]
                                - expert_embedding_along_pretrained
                            )
                            expert_embeddings_orthogonal_to_pretrained.append(
                                expert_embedding_orthogonal_to_pretrained
                            )
                        if self.trainable_index:
                            basis = self._gram_schmidt(
                                expert_embeddings_orthogonal_to_pretrained[
                                    : self.trainable_index
                                ]
                                + expert_embeddings_orthogonal_to_pretrained[
                                    self.trainable_index + 1 :
                                ]
                            )
                            new_expert_embedding_normalized = (
                                expert_embeddings_orthogonal_to_pretrained[
                                    self.trainable_index
                                ]
                                / (
                                    torch.norm(
                                        expert_embeddings_orthogonal_to_pretrained[
                                            self.trainable_index
                                        ]
                                    )
                                    + self.epsilon
                                )
                            )
                        else:
                            expert_embeddings_orthogonal_to_pretrained = torch.stack(
                                expert_embeddings_orthogonal_to_pretrained
                            )
                            basis = self._gram_schmidt(
                                expert_embeddings_orthogonal_to_pretrained[:-1]
                            )
                            new_expert_embedding_normalized = (
                                expert_embeddings_orthogonal_to_pretrained[-1]
                                / (
                                    torch.norm(
                                        expert_embeddings_orthogonal_to_pretrained[-1]
                                    )
                                    + self.epsilon
                                )
                            )
                    else:
                        basis = self._gram_schmidt(expert_embeddings[:-1])
                        new_expert_embedding_normalized = expert_embeddings[-1] / (
                            torch.norm(expert_embeddings[-1]) + self.epsilon
                        )
                else:
                    basis = self._gram_schmidt(expert_embeddings[:-1])
                    new_expert_embedding_normalized = expert_embeddings[-1] / (
                        torch.norm(expert_embeddings[-1]) + self.epsilon
                    )
                subspace_expert_embedding = torch.matmul(
                    torch.matmul(new_expert_embedding_normalized.unsqueeze(0), basis.T),
                    basis,
                )
                orthogonal_penalty_loss = torch.sum(
                    subspace_expert_embedding**2, dim=-1
                ).mean()
            else:
                normalized_expert_embeddings = expert_embeddings / (
                    torch.norm(expert_embeddings, dim=-1, keepdim=True) + self.epsilon
                )
                similarity = torch.matmul(
                    normalized_expert_embeddings, normalized_expert_embeddings.T
                )
                if "relu" in self.orthogonal_type:
                    orthogonal_penalty_loss = F.relu(similarity)
                else:
                    orthogonal_penalty_loss = similarity
                if "l1" in self.orthogonal_type:
                    orthogonal_penalty_loss = torch.abs(orthogonal_penalty_loss)
                elif "l2" in self.orthogonal_type:
                    orthogonal_penalty_loss = orthogonal_penalty_loss**2

                orthogonal_penalty_loss = orthogonal_penalty_loss - torch.diag(
                    torch.diag(orthogonal_penalty_loss)
                )
                # TODO: Fix the below line based on the number of newly added experts
                orthogonal_penalty_loss = orthogonal_penalty_loss[-1]
                if self.ol_aggregate_type == "mean":
                    orthogonal_penalty_loss = orthogonal_penalty_loss.mean()
                elif self.ol_aggregate_type == "sum":
                    orthogonal_penalty_loss = orthogonal_penalty_loss.sum()
            orthogonal_penalty_loss = (
                orthogonal_penalty_loss * self.orthogonal_penalty_weight
            )
            if self.write_orthogonal_penalty_key in self.global_hidden_dict:
                self.global_hidden_dict[self.write_orthogonal_penalty_key].append(
                    orthogonal_penalty_loss
                )
            else:
                self.global_hidden_dict[self.write_orthogonal_penalty_key] = [
                    orthogonal_penalty_loss
                ]
        if self.tag_penalty_type and self.tag_penalty_weight > 0:
            if self.tag_penalty_type == "l1":
                # take last expert embedding and make it close to router hidden states
                tag_penalty_loss = torch.abs(
                    expert_embeddings[-1] - router_hidden_states
                )
            elif self.tag_penalty_type == "l2":
                tag_penalty_loss = (expert_embeddings[-1] - router_hidden_states) ** 2
                mask = torch.rand(tag_penalty_loss.shape[0]) > (
                    1 - self.tag_supervision_ratio
                )
                mask = mask.to(tag_penalty_loss.device)
                tag_penalty_loss = mask[:, None] * tag_penalty_loss
            elif self.tag_penalty_type == "ce":
                # take cross entropy loss of routing weights and one hot vector of last expert
                targets = torch.ones(routing_weights.shape[0]).to(
                    routing_weights.device
                ).long() * (self.num_experts - 1)
                # fill (1 - tag_supervision_ratio)% of values in targets with -100 using random mask
                mask = torch.rand(targets.shape) < (1 - self.tag_supervision_ratio)
                mask = mask.to(targets.device)
                targets = targets.masked_fill(mask, -100)
                tag_penalty_loss = F.cross_entropy(scores, targets, reduction="none")
            tag_penalty_loss = tag_penalty_loss.mean()
            tag_penalty_loss = tag_penalty_loss * self.tag_penalty_weight
            if self.write_tag_penalty_key in self.global_hidden_dict:
                self.global_hidden_dict[self.write_tag_penalty_key].append(
                    tag_penalty_loss
                )
            else:
                self.global_hidden_dict[self.write_tag_penalty_key] = [tag_penalty_loss]
        if self.entropy_penalty_weight > 0:
            entropy_penalty_loss = -torch.sum(
                routing_weights * torch.log(routing_weights + self.epsilon), dim=-1
            )
            entropy_penalty_loss = entropy_penalty_loss.mean()
            entropy_penalty_loss = entropy_penalty_loss * self.entropy_penalty_weight
            if self.write_entropy_penalty_key in self.global_hidden_dict:
                self.global_hidden_dict[self.write_entropy_penalty_key].append(
                    entropy_penalty_loss
                )
            else:
                self.global_hidden_dict[self.write_entropy_penalty_key] = [
                    entropy_penalty_loss
                ]
        if self.subspace_penalty_weight > 0:
            if self.subspace_penalty_type == "remove_orth_component":
                basis = self._gram_schmidt(expert_embeddings)
                router_hidden_states_normalized = router_hidden_states / (
                    torch.norm(router_hidden_states, dim=-1, keepdim=True)
                    + self.epsilon
                )
                orth_router_hidden_states = (
                    router_hidden_states_normalized
                    - torch.matmul(
                        torch.matmul(router_hidden_states_normalized, basis.T),
                        basis,
                    )
                )
                subspace_penalty_loss = torch.sum(
                    orth_router_hidden_states**2, dim=-1
                )
            else:
                X = torch.matmul(router_hidden_states, expert_embeddings.T)
                # FIX IT
                recons_router_hidden_states = (
                    torch.matmul(X, expert_embeddings) / 1024.0
                )
                subspace_penalty_loss = (
                    recons_router_hidden_states - router_hidden_states
                ) ** 2
            subspace_penalty_loss = subspace_penalty_loss.mean()
            subspace_penalty_loss = subspace_penalty_loss * self.subspace_penalty_weight
            if self.write_subspace_penalty_key in self.global_hidden_dict:
                self.global_hidden_dict[self.write_subspace_penalty_key].append(
                    subspace_penalty_loss
                )
            else:
                self.global_hidden_dict[self.write_subspace_penalty_key] = [
                    subspace_penalty_loss
                ]
        routing_weights = routing_weights.to(hidden_states_dtype)
        return routing_weights

    def pre_forward(self, *args, **kwargs):
        router_hidden_states = self.global_hidden_dict[self.read_hidden_key]
        routing_weights = self._forward(router_hidden_states)
        # for analysis routing weights in decoder get overidden for each new token, so we need to save them
        if (
            "decoder" in self.write_routing_weights_key[2]
            and "EncDecAttention.k" not in self.write_routing_weights_key[2]
            and "EncDecAttention.v" not in self.write_routing_weights_key[2]
        ):
            save_key = (
                self.write_routing_weights_key[0],
                self.write_routing_weights_key[1],
                self.write_routing_weights_key[2],
                "concatenated",
            )
            if save_key in self.global_hidden_dict:
                self.global_hidden_dict[save_key] = torch.cat(
                    (self.global_hidden_dict[save_key], routing_weights), dim=1
                )
            else:
                self.global_hidden_dict[save_key] = routing_weights
        self.global_hidden_dict[self.write_routing_weights_key] = routing_weights

    def post_forward(self, module_outputs, *args, **kwargs):
        router_hidden_states = self.global_hidden_dict[self.read_hidden_key]
        routing_weights = self._forward(router_hidden_states)
        self.global_hidden_dict[self.write_routing_weights_key] = routing_weights


@gin.configurable(
    allowlist=[
        "d_in",
        "d_out",
        "d_bottleneck",
        "non_linearity",
        "position",
        "residual_connection",
        "layer_norm_position",
        "one_hot",
        "normalize_topk",
        "topk_value",
        "divide_by_d_bottleneck",
        "init_scale",
        "epsilon",
        "replace_with_weighted_hiddens",
        "replace_weighted_hiddens_type",
        "learn_output_gate",
        "learn_input_gate",
        "use_input_gate",
    ],
)
class FFNExperts(ExtendableAddon):
    """
    This module is a MoE version of the adpater module.
    When setting non_linearity to identity, we can also use it to implement LoRA.
    """

    has_pre_forward = BOOL_PLACEHOLDER
    has_post_forward = BOOL_PLACEHOLDER
    pre_forward_returns = BOOL_PLACEHOLDER
    post_forward_returns = BOOL_PLACEHOLDER
    _extendable_parameters = LIST_PLACEHOLDER
    _ref_attr_names = ["d_in", "d_out"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        read_routing_weights_key,
        moe_link,
        d_in,
        d_out,
        d_bottleneck,
        non_linearity,
        normalize_topk=False,
        one_hot=False,
        topk_value=None,
        position="beside",
        residual_connection=False,
        layer_norm_position=None,
        parallel_axis="none",
        divide_by_d_bottleneck=False,
        replace_with_weighted_hiddens=False,
        replace_weighted_hiddens_type=None,
        learn_output_gate=None,
        learn_input_gate=None,
        use_input_gate=False,
        init_scale=0.01,
        epsilon=1e-6,
    ):
        """
        Args:
            global_hidden_dict: dict, global hidden states visible to all addon modules
            read_routing_weights_key: str, key to read routing weights ex: ("module_name", "routing_weights")
            moe_link: MoELink, link to the MoE module
            d_in: int, input dimension
            d_out: int, output dimension
            d_bottleneck: int, dimension of the bottleneck layer
            non_linearity: str, activation function
            position: str, position of the moe module, can be "before" "beside" or "after"
            residual_connection: bool, whether to use residual connection (helpful for adapters)
            parallel_axis: str,  which axis to split when parallelize the module, can be "none", "d_in", "d_bottleneck", "d_out", or "num_experts"
        """
        super().__init__(global_hidden_dict, moe_link)
        self.moe_link.add_moe_layer(self)
        self.read_routing_weights_key = read_routing_weights_key
        self.num_experts = 0
        self.d_in = d_in
        self.d_out = d_out
        self.d_bottleneck = d_bottleneck
        self.divide_by_d_bottleneck = divide_by_d_bottleneck
        self.non_linearity = non_linearity
        self.position = position
        self.one_hot = one_hot
        self.topk_value = topk_value
        self.residual_connection = residual_connection
        self.layer_norm_position = layer_norm_position
        self.epsilon = epsilon
        self.parallel_axis = parallel_axis
        self.init_scale = init_scale
        self.normalize_topk = normalize_topk
        self.replace_with_weighted_hiddens = replace_with_weighted_hiddens
        self.replace_weighted_hiddens_type = replace_weighted_hiddens_type
        self.learn_output_gate = learn_output_gate
        self.learn_input_gate = learn_input_gate
        self.use_input_gate = use_input_gate
        self._resolve_ref_attrs(host_module)
        if self.learn_output_gate is not None:
            self.expert_output_gate = nn.Parameter(torch.zeros(self.d_out))
            self.temperature = None
        if self.learn_input_gate is not None:
            self.expert_input_gate = nn.Parameter(torch.zeros(self.d_in))
            self.temperature = None
        if (
            self.replace_weighted_hiddens_type
            and "outgate_learned" in self.replace_weighted_hiddens_type
        ):
            self.expert_output_gate = nn.Parameter(torch.zeros(self.d_out))
        if (
            self.replace_weighted_hiddens_type
            and "ingate_learned" in self.replace_weighted_hiddens_type
        ):
            self.expert_input_gate = nn.Parameter(torch.zeros(self.d_in))
        assert not self.residual_connection or self.d_in == self.d_out
        assert self.parallel_axis in ["none", "d_in", "d_bottleneck", "d_out"]
        assert self.position in ["before", "beside", "after"]

        if self.position == "before":
            self.has_pre_forward = self.pre_forward_returns = True
            self.has_post_forward = self.post_forward_returns = False
        elif self.position == "beside":
            self.has_pre_forward = True
            self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True
            self._temp_hidden_key = None
        elif self.position == "after":
            self.has_pre_forward = self.pre_forward_returns = False
            self.has_post_forward = self.post_forward_returns = True
        if non_linearity == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = ACT2FN[non_linearity]
        if self.layer_norm_position == "before":
            self._extendable_parameters = [
                "layer1",
                "layer2",
                "input_layer_norm_weight",
                "input_layer_norm_bias",
            ]
        elif self.layer_norm_position == "after":
            self._extendable_parameters = [
                "layer1",
                "layer2",
                "output_layer_norm_weight",
                "output_layer_norm_bias",
            ]
        else:
            self._extendable_parameters = ["layer1", "layer2"]

    def _get_init_weights(self, num_new_experts):
        layer1 = nn.Parameter(
            torch.randn(num_new_experts, self.d_in, self.d_bottleneck) * self.init_scale
        )
        if len(self._extendable_parameters) > 2:
            layer2 = nn.Parameter(
                torch.randn(num_new_experts, self.d_bottleneck, self.d_out)
                * self.init_scale
            )
        else:
            layer2 = nn.Parameter(
                torch.randn(num_new_experts, self.d_bottleneck, self.d_out) * 0.0
            )
        if self.layer_norm_position == "before":
            input_layer_norm_weight = nn.Parameter(
                torch.ones(num_new_experts, self.d_in)
            )
            input_layer_norm_bias = nn.Parameter(
                torch.zeros(num_new_experts, self.d_in)
            )
            return {
                "layer1": layer1,
                "layer2": layer2,
                "input_layer_norm_weight": input_layer_norm_weight,
                "input_layer_norm_bias": input_layer_norm_bias,
            }
        elif self.layer_norm_position == "after":
            output_layer_norm_weight = nn.Parameter(
                torch.ones(num_new_experts, self.d_out)
            )
            output_layer_norm_bias = nn.Parameter(
                torch.zeros(num_new_experts, self.d_out)
            )
            return {
                "layer1": layer1,
                "layer2": layer2,
                "output_layer_norm_weight": output_layer_norm_weight,
                "output_layer_norm_bias": output_layer_norm_bias,
            }
        else:
            return {"layer1": layer1, "layer2": layer2}

    def _forward(self, input_hidden, routing_weights):
        """
        Args:
            input_hidden: (..., seq_len, d_in)
            routing_weights: (..., num_experts), one-hot or soft distribution
        Returns:
            output_hidden: (..., seq_len, d_out)
        """
        if self.separate_experts:
            self._assemble_extendable_parameters()
        if self.layer_norm_position == "before":
            raise NotImplementedError

        if len(routing_weights.shape) == 3:
            # if routing_weights has token dim, then route using dispatcher
            assert self.topk_value is not None
            bs, seq_len, _ = input_hidden.shape
            input_hidden = input_hidden.reshape(-1, self.d_in)
            routing_weights = routing_weights.reshape(-1, self.num_experts)
            if routing_weights.shape[0] != input_hidden.shape[0]:
                # handle decoder case
                import ipdb

                ipdb.set_trace()
            topk_weights, topk_indices = torch.topk(
                routing_weights, self.topk_value, dim=-1
            )
            # normalize topk_weights again? Not sure
            if self.normalize_topk:
                topk_weights = topk_weights / (
                    torch.sum(topk_weights, dim=-1, keepdim=True) + self.epsilon
                )
            topk_weights = topk_weights.to(routing_weights.dtype)
            zeros = torch.zeros_like(routing_weights)
            gates = zeros.scatter(1, topk_indices, topk_weights)
            sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
            _, expert_index = sorted_experts.split(1, dim=1)
            batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
            part_sizes = (gates > 0).sum(0).tolist()
            gates_exp = gates[batch_index.flatten()]
            nonzero_gates = torch.gather(gates_exp, 1, expert_index)
            inp_exp = input_hidden[batch_index].squeeze(1)
            expert_inputs = torch.split(inp_exp, part_sizes, dim=0)
            expert_outputs = [
                self.activation_fn(torch.matmul(expert_inputs[i], self.layer1[i]))
                for i in range(self.num_experts)
            ]
            if self.use_input_gate:
                expert_outputs = [
                    expert_outputs[i]
                    * torch.sigmoid(expert_outputs[i][:, 0].unsqueeze(-1))
                    for i in range(self.num_experts)
                ]
            expert_outputs = [
                torch.matmul(expert_outputs[i], self.layer2[i])
                for i in range(self.num_experts)
            ]
            stitched = torch.cat(expert_outputs, 0)
            stitched = stitched.mul(nonzero_gates)
            zeros = torch.zeros(
                gates.size(0),
                expert_outputs[-1].size(1),
                device=stitched.device,
                dtype=stitched.dtype,
            )
            output_hidden = zeros.index_add(0, batch_index, stitched)
            output_hidden = output_hidden.reshape(bs, seq_len, self.d_out)
            input_hidden = input_hidden.reshape(bs, seq_len, self.d_in)
        else:
            if self.topk_value is not None:
                bs = input_hidden.shape[0]
                input_hidden = input_hidden.unsqueeze(-3)  # (..., 1, seq_len, d_in)
                topk_weights, topk_indices = torch.topk(
                    routing_weights, self.topk_value, dim=-1
                )
                if self.normalize_topk:
                    topk_weights = topk_weights / torch.sum(
                        topk_weights, dim=-1, keepdim=True
                    )
                layer1_expanded = self.layer1.unsqueeze(0).repeat(bs, 1, 1, 1)
                topk_indices_layer1_expanded = (
                    topk_indices.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, self.d_in, self.d_bottleneck)
                )
                active_layer1 = torch.gather(
                    layer1_expanded, 1, topk_indices_layer1_expanded
                )  # (...,topk_value, d_in, d_bottleneck)
                layer2_expanded = self.layer2.unsqueeze(0).repeat(bs, 1, 1, 1)
                topk_indices_layer2_expanded = (
                    topk_indices.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(1, 1, self.d_bottleneck, self.d_out)
                )
                # check for dtype here, changes to float32
                active_layer2 = (
                    torch.gather(layer2_expanded, 1, topk_indices_layer2_expanded)
                    # * topk_weights[:, :, None, None]
                )  # (...,topk_value, d_bottleneck, d_out)
            else:
                active_layer1 = torch.matmul(
                    routing_weights, self.layer1.view(-1, self.d_in * self.d_bottleneck)
                ).view(
                    -1, self.d_in, self.d_bottleneck
                )  # (..., d_in, d_bottleneck)
                active_layer2 = torch.matmul(
                    routing_weights,
                    self.layer2.view(-1, self.d_bottleneck * self.d_out),
                ).view(
                    -1, self.d_bottleneck, self.d_out
                )  # (..., d_bottleneck, d_out)
            # how is  output_hidden back in bfloat16?
            if self.use_input_gate:
                inner_activations = self.activation_fn(
                    torch.matmul(input_hidden, active_layer1)
                )
                gate_scores = inner_activations[:, :, :, 0].unsqueeze(-1)
                gate_scores = torch.sigmoid(gate_scores)
                inner_activations = inner_activations * gate_scores
                # modify these inner activations
                output_hidden = torch.matmul(inner_activations, active_layer2)
            else:
                output_hidden = torch.matmul(
                    self.activation_fn(torch.matmul(input_hidden, active_layer1)),
                    active_layer2,
                )  # (..., seq_len, d_out)
            if self.topk_value is not None:
                # do layer norms according to topk_indices from self.output_layer_norms
                if self.layer_norm_position == "after":
                    # unsqueeze for seq dim
                    output_layer_norm_weight = torch.gather(
                        self.output_layer_norm_weight.unsqueeze(0).repeat(bs, 1, 1),
                        1,
                        topk_indices.unsqueeze(-1).repeat(1, 1, self.d_out),
                    ).unsqueeze(-2)
                    output_layer_norm_bias = torch.gather(
                        self.output_layer_norm_bias.unsqueeze(0).repeat(bs, 1, 1),
                        1,
                        topk_indices.unsqueeze(-1).repeat(1, 1, self.d_out),
                    ).unsqueeze(-2)
                    # normalize ouput_hidden with above terms
                    output_hidden_mean = torch.mean(output_hidden, dim=-1, keepdim=True)
                    output_hidden_var = torch.mean(
                        (output_hidden - output_hidden_mean) ** 2, dim=-1, keepdim=True
                    )
                    output_hidden = (output_hidden - output_hidden_mean) / torch.sqrt(
                        output_hidden_var + self.epsilon
                    )
                    output_hidden = (
                        output_hidden * output_layer_norm_weight
                        + output_layer_norm_bias
                    )
                if self.residual_connection:
                    output_hidden = output_hidden + input_hidden
                # weight output_hiddens according to topk_weights
                # should be same as mulitplying to active_layer2
                output_hidden = output_hidden * topk_weights[:, :, None, None]
                # output_hidden: (...,topk_value, seq_len, d_out)
                output_hidden = torch.sum(output_hidden, dim=1)  # (..., seq_len, d_out)
            else:
                if self.layer_norm_position == "after":
                    output_hidden_mean = torch.mean(output_hidden, dim=-1, keepdim=True)
                    output_hidden_var = torch.mean(
                        (output_hidden - output_hidden_mean) ** 2, dim=-1, keepdim=True
                    )
                    output_hidden = (output_hidden - output_hidden_mean) / torch.sqrt(
                        output_hidden_var + self.epsilon
                    )
                    output_hidden = output_hidden * self.output_layer_norm_weight.mean(
                        dim=0
                    ) + self.output_layer_norm_bias.mean(dim=0)
                if self.residual_connection:
                    output_hidden = output_hidden + input_hidden
        if self.divide_by_d_bottleneck:
            if self.use_input_gate:
                output_hidden = output_hidden / (self.d_bottleneck - 1)
            else:
                output_hidden = output_hidden / self.d_bottleneck

        return output_hidden

    def pre_forward(self, hidden_states, *args, **kwargs):
        routing_weights = self.global_hidden_dict[self.read_routing_weights_key]
        if self.replace_with_weighted_hiddens:
            overwrite_key = self.moe_link.router.read_hidden_key
            if "pre_expose_hiddens" in overwrite_key:
                self.global_hidden_dict[overwrite_key] = hidden_states
        if self.learn_input_gate is not None:
            input_gate_scores = torch.sum(
                hidden_states * self.expert_input_gate, dim=-1
            )
            input_gate_probs = torch.sigmoid(input_gate_scores)
            if self.learn_input_gate == "only_sigmoid":
                input_gate = input_gate_probs
            else:
                if self.training:
                    U = torch.rand(
                        input_gate_probs.shape, device=input_gate_probs.device
                    )
                    input_gate_modified_scores = torch.log(
                        input_gate_probs + self.epsilon
                    ) + (-torch.log(-torch.log(U + self.epsilon) + self.epsilon))
                    current_step = self.global_hidden_dict["current_step"] + 1
                    anneal_step = self.moe_link.router.anneal_step
                    temperature_schedule = self.moe_link.router.temperature_schedule
                    if current_step in temperature_schedule:
                        self.temperature = temperature_schedule[current_step]
                    self.global_hidden_dict[
                        ("temperature", "router", "curr_temp")
                    ] = torch.tensor(self.temperature)
                    input_gate = torch.sigmoid(
                        input_gate_modified_scores / self.temperature
                    )
                    if self.learn_input_gate == "with_st":
                        # straight-through estimator
                        input_gate_onehot = torch.where(
                            input_gate < 0.5,
                            torch.zeros_like(input_gate),
                            torch.ones_like(input_gate),
                        )
                        input_gate = (
                            input_gate_onehot - input_gate.detach() + input_gate
                        )
                else:
                    if self.learn_input_gate == "with_st":
                        input_gate = torch.where(
                            input_gate_probs < 0.5,
                            torch.zeros_like(input_gate_probs),
                            torch.ones_like(input_gate_probs),
                        )
                    else:
                        input_gate = input_gate_probs

            hidden_states = hidden_states * input_gate.unsqueeze(-1)
        output_hidden = self._forward(hidden_states, routing_weights)
        if self.position == "beside":
            while (
                self._temp_hidden_key is None
                or self._temp_hidden_key in self.global_hidden_dict
            ):
                self._temp_hidden_key = str(uuid.uuid4())[:8]
            self.global_hidden_dict[self._temp_hidden_key] = output_hidden
        elif self.position == "before":
            args.insert(0, output_hidden)
            return args, kwargs

    def post_forward(self, module_outputs, *args, **kwargs):
        if self.position == "beside":
            output_hidden = self.global_hidden_dict[self._temp_hidden_key]
            del self.global_hidden_dict[self._temp_hidden_key]
            if self.replace_with_weighted_hiddens:
                expert_hidden = output_hidden
                if isinstance(module_outputs, tuple):
                    module_hidden = module_outputs[0]
                else:
                    module_hidden = module_outputs
                if self.replace_weighted_hiddens_type == "variance":
                    if self.debug:
                        weights = torch.var(expert_hidden, dim=-1) / torch.var(
                            module_hidden, dim=-1
                        )
                        encoder_mask = self.global_hidden_dict[
                            ("mask", "prepare_mask", "encoder")
                        ]
                        if weights.shape[1] == encoder_mask.shape[1]:
                            weights[encoder_mask == 0] = 0
                        print(f"weights: {weights[0]}")
                        print(
                            f"first example: {[self.global_hidden_dict['example_first'][i] if i < len(self.global_hidden_dict['example_first']) else 'EOS' for i in torch.sort(weights[0], descending=True)[1][0:20]]}"
                        )
                        print(
                            f"last example: {[self.global_hidden_dict['example_last'][i] if i < len(self.global_hidden_dict['example_last']) else 'EOS' for i in torch.sort(weights[-1], descending=True)[1][0:20]]}"
                        )
                        import ipdb

                        ipdb.set_trace()
                    else:
                        weights = torch.var(expert_hidden, dim=-1) / torch.var(
                            module_hidden, dim=-1
                        )
                elif (
                    self.replace_weighted_hiddens_type
                    and "outgate_learned" in self.replace_weighted_hiddens_type
                ):
                    output_gate_scores = torch.sum(
                        output_hidden * self.expert_output_gate, dim=-1
                    )
                    weights = torch.sigmoid(output_gate_scores)
                    if "with_st" in self.replace_weighted_hiddens_type:
                        weights = torch.where(
                            weights < 0.5,
                            torch.zeros_like(weights),
                            torch.ones_like(weights),
                        )
                else:
                    weights = torch.ones(
                        expert_hidden.shape[0],
                        expert_hidden.shape[1],
                        device=expert_hidden.device,
                    )

                if (
                    self.global_hidden_dict[("mask", "prepare_mask", "encoder")].shape[
                        1
                    ]
                    == weights.shape[1]
                ):
                    weights[
                        self.global_hidden_dict[("mask", "prepare_mask", "encoder")]
                        == 0
                    ] = 0
                weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-6)
                if self.debug:
                    print(
                        f"first example: {[self.global_hidden_dict['example_first'][i] if i < len(self.global_hidden_dict['example_first']) else 'EOS' for i in torch.sort(weights[0], descending=True)[1][0:20]]}"
                    )
                    print(
                        f"last example: {[self.global_hidden_dict['example_last'][i] if i < len(self.global_hidden_dict['example_last']) else 'EOS' for i in torch.sort(weights[-1], descending=True)[1][0:20]]}"
                    )
                    import ipdb

                    ipdb.set_trace()
                overwrite_key = self.moe_link.router.read_hidden_key
                if "pre_expose_hiddens" in overwrite_key:
                    input_hidden = self.global_hidden_dict[overwrite_key]
                    weighted_hidden = torch.sum(
                        weights.unsqueeze(-1) * input_hidden, dim=1
                    )
                    self.global_hidden_dict[overwrite_key] = weighted_hidden

            if self.learn_output_gate is not None:
                output_gate_scores = torch.sum(
                    output_hidden * self.expert_output_gate, dim=-1
                )
                output_gate_probs = torch.sigmoid(output_gate_scores)
                if self.training:
                    U = torch.rand(
                        output_gate_probs.shape, device=output_gate_probs.device
                    )
                    output_gate_modified_scores = torch.log(
                        output_gate_probs + self.epsilon
                    ) + (-torch.log(-torch.log(U + self.epsilon) + self.epsilon))
                    current_step = self.global_hidden_dict["current_step"] + 1
                    anneal_step = self.moe_link.router.anneal_step
                    temperature_schedule = self.moe_link.router.temperature_schedule
                    if current_step in temperature_schedule:
                        self.temperature = temperature_schedule[current_step]
                    self.global_hidden_dict[
                        ("temperature", "router", "curr_temp")
                    ] = torch.tensor(self.temperature)
                    output_gate = torch.sigmoid(
                        output_gate_modified_scores / self.temperature
                    )
                    if self.learn_output_gate == "with_st":
                        # straight-through estimator
                        output_gate_onehot = torch.where(
                            output_gate < 0.5,
                            torch.zeros_like(output_gate),
                            torch.ones_like(output_gate),
                        )
                        output_gate = (
                            output_gate_onehot - output_gate.detach() + output_gate
                        )
                    if self.learn_output_gate == "with_entropy":
                        entropy_penalty_loss = -(
                            output_gate * torch.log(output_gate + self.epsilon)
                            + (1 - output_gate)
                            * torch.log(1 - output_gate + self.epsilon)
                        )
                        entropy_penalty_loss = entropy_penalty_loss.mean()
                        entropy_penalty_loss = (
                            entropy_penalty_loss
                            * self.moe_link.router.entropy_penalty_weight
                        )
                        if (
                            self.moe_link.router.write_entropy_penalty_key
                            in self.global_hidden_dict
                        ):
                            self.global_hidden_dict[
                                self.moe_link.router.write_entropy_penalty_key
                            ].append(entropy_penalty_loss)
                        else:
                            self.global_hidden_dict[
                                self.moe_link.router.write_entropy_penalty_key
                            ] = [entropy_penalty_loss]
                else:
                    if self.debug:
                        encoder_mask = self.global_hidden_dict[
                            ("mask", "prepare_mask", "encoder")
                        ]
                        if output_gate_probs.shape[1] == encoder_mask.shape[1]:
                            output_gate_probs[encoder_mask == 0] = 0
                        if self.learn_output_gate == "with_st":
                            output_gate_probs = torch.where(
                                output_gate_probs < 0.5,
                                torch.zeros_like(output_gate_probs),
                                torch.ones_like(output_gate_probs),
                            )
                        print(f"shape: {output_gate_probs.shape}")
                        print(f"probs: {output_gate_probs[0]}")
                        print(
                            f"sparsity percentage: {round(torch.sum(output_gate_probs == 0).item() / output_gate_probs.numel(), 2)}"
                        )
                        print(
                            f"first example: {[self.global_hidden_dict['example_first'][i] if i < len(self.global_hidden_dict['example_first']) else 'EOS' for i in torch.sort(output_gate_probs[0], descending=True)[1][0:20]]}"
                        )
                        print(
                            f"last example: {[self.global_hidden_dict['example_last'][i] if i < len(self.global_hidden_dict['example_last']) else 'EOS' for i in torch.sort(output_gate_probs[-1], descending=True)[1][0:20]]}"
                        )
                        import ipdb

                        ipdb.set_trace()
                    if self.learn_output_gate == "with_st":
                        output_gate = torch.where(
                            output_gate_probs < 0.5,
                            torch.zeros_like(output_gate_probs),
                            torch.ones_like(output_gate_probs),
                        )
                    else:
                        output_gate = output_gate_probs
                    output_hidden = output_hidden * output_gate.unsqueeze(-1)

            if isinstance(module_outputs, tuple):
                return (
                    (module_outputs[0] + output_hidden,) + module_outputs[1:],
                    args,
                    kwargs,
                )
            else:
                return module_outputs + output_hidden, args, kwargs
        elif self.position == "after":
            routing_weights = self.global_hidden_dict[self.read_routing_weights_key]
            if isinstance(module_outputs, tuple):
                output_hidden = self._forward(module_outputs[0], routing_weights)
                return (output_hidden,) + module_outputs[1:], args, kwargs
            else:
                output_hidden = self._forward(module_outputs, routing_weights)
                return output_hidden, args, kwargs


@gin.configurable(allowlist=["d_hidden", "one_hot"])
class ScalerExperts(ExtendableAddon):
    """
    This module scales the features of the input_hidden, and it can be used to implement IA3.
    Different experts have different scaling vectors.
    """

    has_post_forward = True
    post_forward_returns = True
    _extendable_parameters = ["scalers"]
    _ref_attr_names = ["d_hidden"]

    def __init__(
        self,
        host_module,
        global_hidden_dict,
        read_routing_weights_key,
        moe_link,
        d_hidden,
        one_hot=False,
        parallel_axis="none",
    ):
        """
        Args:
            global_hidden_dict: dict, global hidden states visible to all addon modules
            read_routing_weights_key: str, key to read the routing weights from global_hidden_dict ex: ("module_name", "routing_weights")
            moe_link: MOELink, link to the MOE module
            d_hidden: int, hidden dimension
            one_hot: bool, whether the routing weights are one-hot vectors
            parallel_axis: str, which axis to split when parallelize the module, can be "none", "d_hidden", or "num_experts"
        """
        super().__init__(global_hidden_dict, moe_link)
        self.moe_link.add_moe_layer(self)
        self.read_routing_weights_key = read_routing_weights_key
        self.num_experts = 0
        self.d_hidden = d_hidden
        self.one_hot = one_hot
        self.parallel_axis = parallel_axis
        self._resolve_ref_attrs(host_module)
        assert self.parallel_axis in ["none", "d_hidden", "num_experts"]

    def _get_init_weights(self, num_new_experts):
        scalers = torch.ones(num_new_experts, self.d_hidden)
        return {"scalers": scalers}

    def _forward(self, input_hidden, routing_weights):
        """
        Args:
            input_hidden: (..., seq_len, d_hidden)
            routing_weights: (..., num_experts), one-hot or soft distribution
        Returns:
            output_hidden: (..., seq_len, d_hidden)
        """
        if self.separate_experts:
            self._assemble_extendable_parameters()
        if self.one_hot:
            one_hot_value, one_hot_indices = routing_weights.max(dim=-1)  # (..., )
            active_scaling_vector = (
                self.scalers[one_hot_indices] * one_hot_value
            )  # (..., d_hidden)
        else:
            active_scaling_vector = torch.matmul(
                routing_weights, self.scalers
            )  # (..., d_hidden)
            output_hidden = input_hidden * active_scaling_vector.unsqueeze(
                -2
            )  # (..., seq_len, d_hidden)
        return output_hidden

    def post_forward(self, module_outputs, *args, **kwargs):
        if type(module_outputs) is tuple:
            hidden_states = module_outputs[0]
        else:
            hidden_states = module_outputs
        routing_weights = self.global_hidden_dict[self.read_routing_weights_key]
        output_hidden = self._forward(hidden_states, routing_weights)
        return output_hidden, args, kwargs