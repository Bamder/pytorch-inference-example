from torch import nn

class Classifier(nn.Module):

    def __init__(
        self, 
        input_resolution : tuple[int, int] | None = None,
        num_classes : int = 10
    ):
        super().__init__()

        # gap switch
        self.use_gap = input_resolution is None

        # value sequence of channels 
        channels = (32, 64)
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1),

            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2) if input_resolution else nn.AdaptiveAvgPool2d(output_size=1),

            nn.Flatten()
        )

        if self.use_gap:
            cls_head_input_size = channels[-1]
        else:
            height, width = input_resolution or (0, 0)
            cls_head_input_size = channels[-1] * (height // 4) * (width // 4)


        
        self.classification_head = nn.Sequential(
            nn.Linear(cls_head_input_size, num_classes)
        )

    def forward(self, input_sample):
        return self.classification_head(self.feature_extractor(input_sample))