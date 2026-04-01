# MULTI-TASK NEURAL NETWORK

class MultiTaskADMETModel(nn.Module):
    def __init__(self, input_dim, task_names, task_types, hidden_dims=[1024, 512, 256]):
        super(MultiTaskADMETModel, self).__init__()
        self.task_names = task_names
        self.task_types = task_types

        # Shared Backbone
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.3)
            ])
            prev_dim = h
        self.backbone = nn.Sequential(*layers)

        # Task-Specific Heads
        self.heads = nn.ModuleDict()
        for task in task_names:
            # We use Linear(1) here; the Loss function will handle Sigmoid for classification
            self.heads[task] = nn.Sequential(
                nn.Linear(hidden_dims[-1], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, x):
        shared = self.backbone(x)
        return {task: self.heads[task](shared) for task in self.task_names}
