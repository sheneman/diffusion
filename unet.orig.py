import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
		self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.01)
    
	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		return x

class EncoderBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(EncoderBlock, self).__init__()
		self.conv_block = ConvBlock(in_channels, out_channels)
		self.pool = nn.MaxPool2d(2)

	def forward(self, x):
		x = self.conv_block(x)
		p = self.pool(x)
		return x, p

class DecoderBlock(nn.Module):
	def __init__(self, in_channels, middle_channels, out_channels):
		super(DecoderBlock, self).__init__()
		self.up = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)
		#self.conv_block = ConvBlock(middle_channels + out_channels, out_channels)
		self.conv_block = ConvBlock(middle_channels + out_channels, out_channels)

	def forward(self, x, skip_connection):
		x = self.up(x)
		x = torch.cat([x, skip_connection], dim=1)
		x = self.conv_block(x)
		return x

class DropoutBlock(nn.Module):
	def __init__(self, p=0.33):
		super(DropoutBlock, self).__init__()
		self.dropout = nn.Dropout(p)

	def forward(self, x):
		x = self.dropout(x)
		return x

class UNet(nn.Module):
	def __init__(self):
		super(UNet, self).__init__()
		self.encoder1 = EncoderBlock(3, 64)
		self.encoder2 = EncoderBlock(64, 128)
		self.encoder3 = EncoderBlock(128, 256)
		self.encoder4 = EncoderBlock(256, 512)
	
		self.dropout1 = DropoutBlock(0.33)

		self.center = ConvBlock(512, 1024)

		self.dropout2 = DropoutBlock(0.33)

		self.decoder4 = DecoderBlock(1024, 512, 256)
		self.decoder3 = DecoderBlock(256, 256, 128)
		self.decoder2 = DecoderBlock(128, 128, 64)
		self.decoder1 = DecoderBlock(64, 64, 64)

		self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

	def forward(self, x):
		enc1, x = self.encoder1(x)
		enc2, x = self.encoder2(x)
		enc3, x = self.encoder3(x)
		enc4, x = self.encoder4(x)

		x = self.dropout1(x)

		x = self.center(x)

		x = self.dropout2(x)

		x = self.decoder4(x, enc4)
		x = self.decoder3(x, enc3)
		x = self.decoder2(x, enc2)
		x = self.decoder1(x, enc1)

		x = self.final_conv(x)
		return x

# Create the model
#model = UNet()
#print(model)

