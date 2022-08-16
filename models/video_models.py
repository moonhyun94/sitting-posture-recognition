import torch
import torch.nn as nn

class Conv_5_C3D(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(Conv_5_C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # self.pool3b = nn.AdaptiveAvgPool3d(output_size=(5,7,7))
        # self.pool3c = nn.AdaptiveMaxPool3d(output_size=(1,1,1))
        # self.pool3d = nn.AdaptiveMaxPool3d(output_size=(5,7,7))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1,1,1)) # padding 0,0,0

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1), padding=(0, 1, 1)) # stride 2,2,2
        
        # (N,256,D,H,W) 
        self.pool3a = nn.AdaptiveAvgPool3d(output_size=(1,1,1)) # (N, 256, 1, 1, 1)
        
        self.fc6, self.fc7, self.fc8 = self.fc_layer(512, 1,1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool3a(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        
        x = self.dropout(x)
        x = self.fc8(x)
        
        logits = x

        return logits, None
    
    def fc_layer(self, C, H, W):
        layer1 = nn.Linear(C*H*W, 256)
        layer2 = nn.Linear(256, 128)
        layer3 = nn.Linear(128, 6)
        return layer1, layer2, layer3
        
    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        }

        p_dict = torch.load('/home/moon/research/sitting-posture-recognition/weights/c3d-pretrained.pth')
        s_dict = self.state_dict()
        
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
class Conv_3_C3D(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(Conv_3_C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # (N,256,D,H,W) 
        self.pool3a = nn.AdaptiveAvgPool3d(output_size=(1,1,1)) # (N, 256, 1, 1, 1)
        
        self.fc6, self.fc7, self.fc8 = self.fc_layer(256, 1,1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3a(x)
        
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3]*x.shape[4])
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        
        x = self.dropout(x)
        x = self.fc8(x)
        
        logits = x

        return logits, None
    
    def fc_layer(self, C, H, W):
        layer1 = nn.Linear(C*H*W, 256)
        layer2 = nn.Linear(256, 128)
        layer3 = nn.Linear(128, 6)
        return layer1, layer2, layer3
        
    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        }

        p_dict = torch.load('/home/moon/research/sitting-posture-recognition/weights/c3d-pretrained.pth')
        s_dict = self.state_dict()
        
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()