import os
import torch
import torch.nn as nn
import timm


# Reference : https://amaarora.github.io/posts/2020-08-30-gempool.html

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        """
        Initialize the Generalized Mean Pooling layer.

        Args:
            p (float, optional): The power for the pooling. Defaults to 3.
            eps (float, optional): The epsilon value to avoid division by zero. Defaults to 1e-6.
        """
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        """
        Forward pass of the Generalized Mean Pooling layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        """
        Applies Generalized Mean Pooling to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            p (float, optional): The power for the pooling. Defaults to 3.
            eps (float, optional): The epsilon value to avoid division by zero. Defaults to 1e-6.

        Returns:
            torch.Tensor: The output tensor after the pooling.
        """
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        """
        Returns a string representation of the module.

        Returns:
            str: The string representation of the module.
        """
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
    


class EffNetModel(nn.Module):
    def __init__(self, model_name, num_classes=4, pretrained=True, checkpoint_path=None):
        """
        Initialize the EfficientNet model.

        Args:
            model_name (str): The name of the EfficientNet model.
            num_classes (int, optional): The number of classes for the final linear layer. Defaults to 4.
            pretrained (bool, optional): Whether to use the pre-trained weights for the model. Defaults to True.
            checkpoint_path (str, optional): The path to the checkpoint file for the model. Defaults to None.
        """
        super(EffNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        """
        Forward pass of the EfficientNet model.

        Args:
            images (torch.Tensor): The input tensor of images.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.sigmoid(self.linear(pooled_features))
        return output
