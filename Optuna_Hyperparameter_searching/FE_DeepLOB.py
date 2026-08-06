import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score

class FE_DeepLOB(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 3):
        super().__init__()

        # Padding layer
        self.causal_pad = nn.ZeroPad2d((0, 0, 3, 0))  # Pad only on the top for time dimension

        # Momentum 
        self.conv_momentum1 = nn.Conv2d(1, 8, kernel_size = (4, 1))
        self.conv_momentum2 = nn.Conv2d(8, 8, kernel_size = (4, 1))
        self.conv_momentum3 = nn.Conv2d(8, 8, kernel_size = (4, 1))

        # Snapshot
        self.conv_feat1 = nn.Conv2d(1, 8, kernel_size=(1, 2), stride=(1, 2))
        self.conv_time1 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_time2 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_time3 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_time4 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_time5 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_time6 = nn.Conv2d(8, 8, kernel_size=(4, 1))
        self.conv_feat2 = nn.Conv2d(8, 8, kernel_size=(1, num_features))

        # Long-term Sensor
        self.causal_pad_dila1 = nn.ReplicationPad2d((0, 0, 2, 0))
        self.dila_conv_time1 = nn.Conv2d(1, 1, kernel_size = (3, 1), dilation = (1, 1))
        self.causal_pad_dila2 = nn.ReplicationPad2d((0, 0, 4, 0))
        self.dila_conv_time2 = nn.Conv2d(1, 1, kernel_size = (3, 1), dilation = (2, 1))
        self.causal_pad_dila3 = nn.ReplicationPad2d((0, 0, 8, 0))
        self.dila_conv_time3 = nn.Conv2d(1, 1, kernel_size = (3, 1), dilation = (4, 1))
        self.causal_pad_dila45 = nn.ReplicationPad2d((0, 0, 16, 0))
        self.dila_conv_time4 = nn.Conv2d(1, 1, kernel_size = (3, 1), dilation = (8, 1))
        self.dila_conv_time5 = nn.Conv2d(1, 1, kernel_size = (3, 1), dilation = (8, 1))

        # Inception layer would be defined here (not implemented in this simple version)
        self.incp_Path1_1 = nn.Conv2d(8, 12, kernel_size=(1,1))
        self.incp_Path1_pad = nn.ZeroPad2d((0, 0, 2, 0))  # Pad only on the top for time dimension
        self.incp_Path1_2 = nn.Conv2d(12, 12, kernel_size=(3,1))

        self.incp_Path2_1 = nn.Conv2d(8, 12, kernel_size=(1,1))
        self.incp_Path2_pad = nn.ZeroPad2d((0, 0, 4, 0))  # Pad only on the top for time dimension
        self.incp_Path2_2 = nn.Conv2d(12, 12, kernel_size=(5,1))

        self.incp_Path3_pad = nn.ConstantPad2d((0, 0, 2, 0), float('-inf'))  # Pad only on the top for time dimension
        self.incp_Path3_1 = nn.MaxPool2d(kernel_size=(3,1), stride=(1,1))
        self.incp_Path3_2 = nn.Conv2d(8, 12, kernel_size=(1,1))

        #self.pic_bottleneck = nn.Sequential(nn.Linear(72, 16), nn.LayerNorm(16), nn.LeakyReLU(negative_slope = 0.01))

        # LSTM layer with 64 hidden units
        self.lstm = nn.LSTM(input_size=47, hidden_size=32, batch_first=True, bidirectional=False)

        # Fully connected layer for classification
        self.head = nn.Linear(32, num_classes)

    def forward(self, x_momentum: torch.Tensor, x_pic: torch.Tensor, x_lt_sensor: torch.Tensor, x_st_sensor: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time_steps, num_features]
        # [batch, channel:1, time_steps, num_features]

        x_momentum = x_momentum.unsqueeze(1)
        x_pic = x_pic.unsqueeze(1)
        x_lt_sensor = x_lt_sensor.unsqueeze(1)
        x_st_sensor = x_st_sensor

        # Mid Price Diff Convolutional Blocks
        x_momentum = self.causal_pad(x_momentum)
        x_momentum = F.leaky_relu(self.conv_momentum1(x_momentum), negative_slope = 0.01)

        x_momentum = self.causal_pad(x_momentum)
        x_momentum = F.leaky_relu(self.conv_momentum2(x_momentum), negative_slope = 0.01)

        x_momentum = self.causal_pad(x_momentum)
        x_momentum = F.leaky_relu(self.conv_momentum3(x_momentum), negative_slope = 0.01)
        x_momentum = x_momentum.squeeze(3).transpose(1, 2)
        
        # Mid Price Convolutional Blocks
        # Dilated Time Convolutions Block1 (3*1)@8 dilation(1)
        x_lt_sensor = self.causal_pad_dila1(x_lt_sensor)  # Apply causal padding before the time convolution
        x_lt_sensor = F.leaky_relu(self.dila_conv_time1(x_lt_sensor), negative_slope=0.01)

        x_lt_sensor = self.causal_pad_dila2(x_lt_sensor)
        x_lt_sensor = F.leaky_relu(self.dila_conv_time2(x_lt_sensor), negative_slope=0.01)

        x_lt_sensor = self.causal_pad_dila3(x_lt_sensor)
        x_lt_sensor = F.leaky_relu(self.dila_conv_time3(x_lt_sensor), negative_slope=0.01)

        x_lt_sensor = self.causal_pad_dila45(x_lt_sensor)
        x_lt_sensor = F.leaky_relu(self.dila_conv_time4(x_lt_sensor), negative_slope=0.01)

        x_lt_sensor = self.causal_pad_dila45(x_lt_sensor)
        x_lt_sensor = F.leaky_relu(self.dila_conv_time5(x_lt_sensor), negative_slope=0.01)
        x_lt_sensor = x_lt_sensor.squeeze(1)

        # Snapshot Convolutional Blocks
        # Spatial Convolution Block
        x_pic = F.leaky_relu(self.conv_feat1(x_pic), negative_slope=0.01)

        # Time Convolutions Block1 (4*1)@8
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time1(x_pic), negative_slope=0.01)
       
        # Time Convolutions Block2 (4*1)@8
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time2(x_pic), negative_slope=0.01)

        # Time Convolutions Block3 (4*1)@8
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time3(x_pic), negative_slope=0.01)
        
        # Time Convolutions Block4 (4*1)@8
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time4(x_pic), negative_slope=0.01)

        # Time Convolutions Block5 (4*1)@8
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time5(x_pic), negative_slope=0.01)

        # Time Convolutions Block6 (4*1)@8  RF = 1 + 6 * (4-1) = 19
        x_pic = self.causal_pad(x_pic)  # Apply causal padding before the time convolution
        x_pic = F.leaky_relu(self.conv_time6(x_pic), negative_slope=0.01)

        # Feature Convolution (1*feature_nums)@8
        x_pic = F.leaky_relu(self.conv_feat2(x_pic), negative_slope=0.01)
        
        # Output shape after conv_time6: [batch, 8, time_steps, features:1] 

        # Inception Block
        # Path 1: (1*1)@16 -> (3*1)@16
        x_path1 = F.leaky_relu(self.incp_Path1_1(x_pic), negative_slope=0.01)
        x_path1 = self.incp_Path1_pad(x_path1)
        x_path1 = F.leaky_relu(self.incp_Path1_2(x_path1), negative_slope=0.01)

        # Path 2: (1*1)@16 -> (5*1)@16
        x_path2 = F.leaky_relu(self.incp_Path2_1(x_pic), negative_slope=0.01)
        x_path2 = self.incp_Path2_pad(x_path2)
        x_path2 = F.leaky_relu(self.incp_Path2_2(x_path2), negative_slope=0.01)

        # Path 3: MaxPool(3*1) -> (1*1)@16
        x_path3 = self.incp_Path3_pad(x_pic)
        x_path3 = self.incp_Path3_1(x_path3)
        x_path3 = F.leaky_relu(self.incp_Path3_2(x_path3), negative_slope=0.01)

        x_pic = torch.cat([x_path1, x_path2, x_path3], dim=1)  # Concatenate along the channel dimension
        x_pic = x_pic.squeeze(3).transpose(1, 2) # Reshape from [batch, 48, time_steps, features:1] to [batch, time_steps, features:48]
        #x_pic = self.pic_bottleneck(x_pic)
        # Output shape after concatenation: [batch, 96, time_steps, features:1]

        # LSTM Block
        x = torch.cat([x_momentum, x_pic, x_lt_sensor, x_st_sensor], dim = 2)
        output, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]  # Take the last hidden state from the LSTM and reshape to [batch, hidden_size]

        # Dense
        x = self.head(x)        
        
        return x
    
def train_engine(model, train_loader, optimizer, criterion, device, lr_scheduler = None):
    model.train()  # Set model to training mode
    total_loss = 0.0
    correct = 0
    total = 0
        
    for data, label in train_loader:
        x_momentum = data['momentum'].to(device)
        x_pic = data['pic'].to(device)
        x_lt_sensor = data['lt_sensor'].to(device)
        x_st_sensor = data['st_sensor'].to(device)
        label = label.to(device).long().squeeze()  # Ensure labels are the correct shape and type

        optimizer.zero_grad()
        output = model(x_momentum, x_pic, x_lt_sensor, x_st_sensor)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct/total

    return avg_loss, avg_acc

def validate_engine(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, label in val_loader:
            x_momentum = data['momentum'].to(device)
            x_pic = data['pic'].to(device) 
            x_lt_sensor = data['lt_sensor'].to(device)
            x_st_sensor = data['st_sensor'].to(device)
            label = label.to(device).long().squeeze()

            output = model(x_momentum, x_pic, x_lt_sensor, x_st_sensor)
            loss = criterion(output, label)
            total_loss += loss.item()
            probs = F.softmax(output.data, dim=1)
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # 1. Down (Class 0) PR-AUC
    y_true_down = (all_labels == 0).astype(int)
    y_score_down = all_probs[:, 0] 
    pr_auc_down = average_precision_score(y_true_down, y_score_down) if y_true_down.sum() > 0 else 0.0

    # 2. Up (Class 2) PR-AUC
    y_true_up = (all_labels == 2).astype(int)
    y_score_up = all_probs[:, 2] 
    pr_auc_up = average_precision_score(y_true_up, y_score_up) if y_true_up.sum() > 0 else 0.0

    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, pr_auc_down, pr_auc_up