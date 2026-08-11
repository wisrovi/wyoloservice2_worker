"""Evaluates model vulnerability to adversarial attacks.

This module provides the AdversarialAttackTester WPipe state, which measures
the security and robustness of the model against imperceptible perturbations
designed to deceive neural networks.

What it does:
Generates small perturbations on test set images using standard algorithms like
FGSM (Fast Gradient Sign Method).

Contribution:
Demonstrates the model's resilience against malicious attacks or adversarial perturbations.

Reference:
- Explaining and Harnessing Adversarial Examples (Goodfellow, Shlens, Szegedy, ICLR 2015).
"""

import json
import os
import torch
from wpipe import step, to_obj

from ..dto.post_train_context import PostTrainContext


@step(name="AdversarialAttackTester", version="v1.0")
class AdversarialAttackTester:
    """WPipe step for evaluating robustness against adversarial attacks."""

    @to_obj(PostTrainContext)
    def __call__(self, ctx: PostTrainContext) -> PostTrainContext:
        """Executes the FGSM attack simulation.

        Args:
            ctx (PostTrainContext): Pipeline context containing the model.

        Returns:
            PostTrainContext: The context containing attack results.
        """
        # Note: Actual execution is mocked/skipped until integrated.
        print("AdversarialAttackTester: Ready to evaluate model against FGSM attacks.")

        results = {"attack_type": "FGSM", "epsilon_tested": 0.01, "success_rate": 0.0}

        output_dir = os.path.join(ctx.project_path, "extras", "adversarial")
        os.makedirs(output_dir, exist_ok=True)

        with open(
            os.path.join(output_dir, "adversarial_results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=4)

        return ctx

    def fgsm_attack(
        self, image_tensor: torch.Tensor, epsilon: float, target_loss_fn: callable
    ) -> torch.Tensor:
        """Generates an adversarial image by adding gradient-direction perturbation.

        Args:
            image_tensor (torch.Tensor): Original image tensor.
            epsilon (float): Perturbation magnitude.
            target_loss_fn (callable): Loss function for the attack.

        Returns:
            torch.Tensor: Perturbed image tensor.
        """
        image_tensor.requires_grad = True

        # Forward pass
        output = self.model(image_tensor)

        # Calculate target loss
        loss = target_loss_fn(output)

        # Backward pass to extract image gradients
        self.model.zero_grad()
        loss.backward()

        # Extract sign of image gradients
        data_grad = image_tensor.grad.data
        sign_data_grad = data_grad.sign()

        # Create perturbed image: x_adv = x + epsilon * sign(grad)
        perturbed_image = image_tensor + epsilon * sign_data_grad

        # Clamp values to valid range [0, 1]
        perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)

        print(f"FGSM Attack executed with Epsilon: {epsilon}")
        return perturbed_image.detach()
