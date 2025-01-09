import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
def compute_kd_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute the knowledge distillation loss.

    Args:
        student_logits (torch.Tensor): Logits from the student model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
        teacher_logits (torch.Tensor): Logits from the teacher model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
        temperature (float): Temperature for distillation.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    """

    if len(student_logits.shape) == 3:
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))

    # Apply temperature scaling
    student_soft_logits = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft_logits = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL divergence (scaled by T^2 for proper gradient scale)
    kd_loss = F.kl_div(student_soft_logits, teacher_soft_logits, reduction="batchmean") * (temperature**2)
    return kd_loss

# def compute_kd_loss_ce(student_logits, teacher_logits, temperature):
#     """
#     Compute the knowledge distillation loss using Cross Entropy.
#
#     Args:
#         student_logits (torch.Tensor): Logits from the student model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
#         teacher_logits (torch.Tensor): Logits from the teacher model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
#         temperature (float): Temperature for distillation.
#
#     Returns:
#         torch.Tensor: Knowledge distillation loss.
#     """
#     # Apply temperature scaling
#     T = temperature
#
#     if len(student_logits.shape) == 3:
#         student_logits = student_logits.view(-1, student_logits.size(-1))
#         teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
#
#     # Softmax probabilities from the teacher
#     teacher_probs = F.softmax(teacher_logits / T, dim=-1)
#     # Log probabilities from the student
#     student_log_probs = F.log_softmax(student_logits / T, dim=-1)
#
#     # Compute Cross Entropy Loss
#     ce_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
#
#     # Scale the loss by T^2
#     kd_loss = ce_loss * (T ** 2)
#     return kd_loss

def compute_kd_loss_ce(student_logits, teacher_logits, temperature):
    """
    Compute the knowledge distillation loss using Cross Entropy.

    Args:
        student_logits (torch.Tensor): Logits from the student model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
        teacher_logits (torch.Tensor): Logits from the teacher model (batch_size, seq_len, num_classes) / (batch_size, num_classes).
        temperature (float): Temperature for distillation.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    """
    # Apply temperature scaling
    T = temperature

    if len(student_logits.shape) == 3:
        # Ensure both tensors have the same sequence length
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]

        # Reshape to 2D
        student_logits = student_logits.reshape(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1))


    # Additional size check
    if student_logits.size(0) != teacher_logits.size(0):
        raise ValueError(
            f"Mismatched batch sizes after reshaping: "
            f"student={student_logits.size(0)}, teacher={teacher_logits.size(0)}"
        )

    # Softmax probabilities from the teacher
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    # Log probabilities from the student
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)

    # Compute Cross Entropy Loss
    ce_loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

    # Scale the loss by T^2
    kd_loss = ce_loss * (T ** 2)
    return kd_loss

def count_parameters(model: nn.Module, verbose: bool = True) -> dict:
    """
    Counts the total, trainable, and non-trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to inspect.
        verbose (bool, optional): If True, prints the parameter counts. Defaults to True.

    Returns:
        dict: A dictionary containing the counts of total, trainable, and non-trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    if verbose:
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-Trainable Parameters: {non_trainable_params:,}")

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-Trainable Parameters": non_trainable_params
    }

def inner_loop(task_model, teacher_model, batch_inputs, batch_dataset):
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

    teacher_batch_outputs = teacher_model(batch_inputs, batch_dataset_override.interface_info, {})

    # Modify the teacher_batch_outputs['sequences'] by removing the first token from each sequence
    teacher_batch_outputs['output_ids'] = [sequence[1:] for sequence in teacher_batch_outputs['output_ids']]
    teacher_batch_outputs['output_ids'] = torch.stack(teacher_batch_outputs['output_ids'], dim=0)
    # print("teacher_batch_outputs['output_ids'][0]: ", teacher_batch_outputs['output_ids'][:5])

    # Modify the target_ids in batch_inputs by replacing them with the teacher_batch_outputs['sequences']
    # print("Before: batch_inputs['target_ids'][0]: ", batch_inputs['target_ids'][:5])
    batch_inputs['target_ids'] = teacher_batch_outputs['output_ids'].cpu().detach()
    # print("After: batch_inputs['target_ids'][0]: ", batch_inputs['target_ids'][:5])

    batch_outputs = task_model(
        batch_inputs,
        batch_dataset.interface_info,
        self.prepare_passing_global_hiddens(),
    )

    loss_1 = batch_outputs["loss"]

    # Compute KD Loss
    loss = compute_kd_loss_ce(
        student_logits=batch_outputs["logits"],
        teacher_logits=teacher_batch_outputs["logits"],
        temperature=2.0
    )

    loss = (loss_1 + loss) / 2
    return loss


def outer_loop():
    pass