import torch
import torch.nn.functional as F

class FGSMAttack:
    """
    Encapsulates the Fast Gradient Sign Method (FGSM) attack logic[cite: 34].
    This class is initialized with the target model.
    """
    def __init__(self, model):
        # Store the target model
        self.model = model
        # Set the model to evaluation mode (essential for consistent attacks)
        self.model.eval()

    def generate_adversarial_example(self, image, label, epsilon):
        """
        Generates an adversarial example using the FGSM formula.

        Args:
            image (torch.Tensor): The original input image.
            label (torch.Tensor): The true label of the image.
            epsilon (float): The magnitude of the perturbation (FGSM strength)[cite: 42].

        Returns:
            torch.Tensor: The resulting adversarial image.
        """
        # 1. Enable gradient tracking for the input image (x)
        # This tells PyTorch to calculate the gradient of the loss with respect to the input data
        image.requires_grad = True

        # 2. Forward pass: Get the prediction
        output = self.model(image)
        
        # 3. Calculate Loss (J): We use the Negative Log Likelihood Loss (NLLLoss) 
        # as the model outputs log_softmax probabilities.
        loss = F.nll_loss(output, label)

        # Zero existing gradients before the backward pass
        self.model.zero_grad()
        
        # 4. Backward pass: Calculate the gradient of the loss w.r.t the input image (dJ/dx)
        loss.backward()

        # Collect the sign of the data gradient (sign(dJ/dx))
        sign_data_grad = image.grad.data.sign()
        
        # 5. Generate Adversarial Image (FGSM Formula): x_adv = x + epsilon * sign(gradient)
        adversarial_image = image.data + epsilon * sign_data_grad

        # 6. Clip the adversarial image to maintain valid pixel range [0, 1]
        adversarial_image = torch.clamp(adversarial_image, 0, 1)

        return adversarial_image

if __name__ == '__main__':
    # This block is just a placeholder to test the class structure
    print("FGSMAttack class structure implemented in backend/fgsm.py.")