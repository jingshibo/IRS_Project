import torch
import torch.nn as nn


feature_mlp_hidden = (256, 128)
feature_mlp_dropout = 0.2
feature_mlp_num_classes = 3
feature_mlp_leaky_relu_slope = 0.05


class FeatureMLPClassifier(nn.Module):
    """MLP classifier for handcrafted tabular feature vectors [N, F]."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = feature_mlp_num_classes,
        hidden_sizes=feature_mlp_hidden,
        dropout: float = feature_mlp_dropout,
        leaky_relu_slope: float = feature_mlp_leaky_relu_slope,
    ):
        super().__init__()
        if in_features < 1:
            raise ValueError(f"in_features must be >= 1, got {in_features}")
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one layer width.")

        layers = []
        prev_features = in_features
        for hidden_features in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_features, hidden_features),
                    nn.BatchNorm1d(hidden_features),
                    nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            prev_features = hidden_features
        layers.append(nn.Linear(prev_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"FeatureMLPClassifier expects x shape [N, F], got {tuple(x.shape)}")
        return self.classifier(x)
