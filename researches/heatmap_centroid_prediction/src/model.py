import torch
import torch.nn as nn

class UNetHeatmapModel(nn.Module):
    def __init__(self, crop_size=64, heatmap_size=64, dropout_p=0.1):
        super(UNetHeatmapModel, self).__init__()
        self.crop_size = crop_size
        self.heatmap_size = heatmap_size
        self.dropout_p = dropout_p

        self.enc1 = self._make_enc_block(4, 64, dropout_p)
        self.pool1 = nn.MaxPool2d(2)    
        self.enc2 = self._make_enc_block(64, 128, dropout_p)
        self.pool2 = nn.MaxPool2d(2)                
        self.enc3 = self._make_enc_block(128, 256, dropout_p) 
        self.pool3 = nn.MaxPool2d(2)                
        self.enc4 = self._make_enc_block(256, 512, dropout_p) 
        self.pool4 = nn.MaxPool2d(2)                

        self.bottleneck = self._make_enc_block(512, 1024, dropout_p)

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) 
        self.dec3 = self._make_enc_block(1024, 512, dropout_p)                 
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  
        self.dec2 = self._make_enc_block(512, 256, dropout_p)                 
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = self._make_enc_block(256, 128, dropout_p)              
        
        self.dropout_before_upsample = nn.Dropout2d(p=dropout_p)

        self.upsample_to_heatmap = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False) 
        self.heatmap_out = nn.Conv2d(128, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def _make_enc_block(self, in_channels, out_channels, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )(self, x):
        enc1_out = self.enc1(x)                       
        enc2_out = self.enc2(self.pool1(enc1_out))      
        enc3_out = self.enc3(self.pool2(enc2_out))     
        enc4_out = self.enc4(self.pool3(enc3_out))    
        bottleneck_in = self.pool4(enc4_out)            
        bottleneck_out = self.bottleneck(bottleneck_in)  

        dec3_input = torch.cat([self.upconv3(bottleneck_out), enc4_out], dim=1) 
        dec3_out = self.dec3(dec3_input)                 

        dec2_input = torch.cat([self.upconv2(dec3_out), enc3_out], dim=1)     
        dec2_out = self.dec2(dec2_input)            

        dec1_input = torch.cat([self.upconv1(dec2_out), enc2_out], dim=1)     
        dec1_out = self.dec1(dec1_input)                  

        heatmap_features = self.dropout_before_upsample(dec1_out) 
        heatmap_features = self.upsample_to_heatmap(heatmap_features) 
        heatmap_raw = self.heatmap_out(heatmap_features)
        B, C, H, W = heatmap_raw.shape
        heatmap_flat = heatmap_raw.view(B, C, -1)
        heatmap_softmax_flat = self.softmax(heatmap_flat)
        heatmap_softmax = heatmap_softmax_flat.view(B, C, H, W)

        return heatmap_softmax

    def forward(self, x):
        enc1_out = self.enc1(x)                         
        enc2_out = self.enc2(self.pool1(enc1_out))       
        enc3_out = self.enc3(self.pool2(enc2_out))      
        enc4_out = self.enc4(self.pool3(enc3_out))     
        bottleneck_in = self.pool4(enc4_out)            
        bottleneck_out = self.bottleneck(bottleneck_in) 

        dec3_input = torch.cat([self.upconv3(bottleneck_out), enc4_out], dim=1) 
        dec3_out = self.dec3(dec3_input)                

        dec2_input = torch.cat([self.upconv2(dec3_out), enc3_out], dim=1)    
        dec2_out = self.dec2(dec2_input)              

        dec1_input = torch.cat([self.upconv1(dec2_out), enc2_out], dim=1)      
        dec1_out = self.dec1(dec1_input)               

        heatmap_features = self.dropout_before_upsample(dec1_out) 
        heatmap_features = self.upsample_to_heatmap(heatmap_features)
        heatmap_raw = self.heatmap_out(heatmap_features)
        B, C, H, W = heatmap_raw.shape
        heatmap_flat = heatmap_raw.view(B, C, -1)
        heatmap_softmax_flat = self.softmax(heatmap_flat)
        heatmap_softmax = heatmap_softmax_flat.view(B, C, H, W)

        return heatmap_softmax