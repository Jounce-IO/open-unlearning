from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_wga_loss, compute_wga_loss_dllm


class WGA(GradDiff):
    def __init__(self, beta=1.0, gamma=1.0, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        # Ref model only needed for KL retain; NLL retain and WGA forget loss use the training model only.
        if self.ref_model is None and self.retain_loss_type == "KL":
            self.ref_model = self._prepare_ref_model(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        adapter_config = getattr(model, "adapter_config", None)
        if adapter_config is not None:
            prev = getattr(adapter_config, "return_per_token_loss", False)
            adapter_config.return_per_token_loss = True
            try:
                outputs = model(**forget_inputs)
            finally:
                adapter_config.return_per_token_loss = prev
            if getattr(outputs, "per_token_nll", None) is not None and getattr(
                outputs, "masked_indices", None
            ) is not None:
                forget_loss, forget_outputs = compute_wga_loss_dllm(outputs, self.beta)
            else:
                forget_loss, forget_outputs = compute_wga_loss(
                    model=model, inputs=forget_inputs, beta=self.beta
                )
        else:
            forget_loss, forget_outputs = compute_wga_loss(
                model=model, inputs=forget_inputs, beta=self.beta
            )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = (
            self.gamma * forget_loss + self.alpha * retain_loss
        )  # default gamma=1.0 alpha=1.0
        return (loss, forget_outputs) if return_outputs else loss
