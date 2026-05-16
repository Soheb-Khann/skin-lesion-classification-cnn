import torch
import torch.nn as nn
import torchvision.models as models


class HybridModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg


        # BACKBONE (ResNet18)

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        # Remove classifier
        self.backbone.fc = nn.Identity()


        # TABULAR BRANCH

        if cfg.use_metadata or cfg.use_handcrafted:
            self.use_tabular = True

            self.tabular = nn.Sequential(
                nn.Linear(cfg.num_tabular_features, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(0.3)
            )
        else:
            self.use_tabular = False


        # FINAL CLASSIFIER

        if self.use_tabular:
            input_dim = 512 + 32
        else:
            input_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, cfg.num_classes)
        )

        # Debug flag
        self.debug_done = False


    # FEATURE EXTRACTOR (IMPORTANT FOR SMOTE)

    def extract_features(self, img, tabular=None):
        img_feat = self.backbone(img)

        if self.use_tabular and tabular is not None:
            tab_feat = self.tabular(tabular)
            combined = torch.cat([img_feat, tab_feat], dim=1)
        else:
            combined = img_feat

        return combined


    # FORWARD PASS

    def forward(self, img, tabular=None):

        combined = self.extract_features(img, tabular)

        # Debug (ONLY ONCE)
        if not self.debug_done:
            print("\n[MODEL CHECK]")
            print(f" Experiment: {self.cfg.experiment}")

            if self.use_tabular:
                print(" Hybrid model active")
                print(" Combined feature size:", combined.shape)
            else:
                print(" Baseline model (image-only)")
                print(" Image features:", combined.shape)

            print("----------------------------\n")

            self.debug_done = True

        # CLASSIFICATION
        out = self.classifier(combined)

        return out