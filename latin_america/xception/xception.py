import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# SECTION 1: MODEL-SPECIFIC HYPERPARAMETERS (Xception U-Net)
# ==============================================================================
INPUT_SEQ_LEN = 6
OUTPUT_SEQ_LEN = 6
MODEL_INPUT_CHANNELS_PER_STEP = 1
MODEL_OUTPUT_CHANNELS_PER_STEP = 1

XCEPTION_INIT_FEATURES = 32       # channels after the first conv
XCEPTION_DEPTH = 3                # number of down/up-sampling stages
XCEPTION_BLOCKS_PER_STAGE = 2     # SeparableConvBlocks per encoder/decoder stage


# ==============================================================================
# SECTION 4: MODEL DEFINITION (Xception-Style U-Net)
# ==============================================================================
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConvBlock, self).__init__()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Pointwise (1x1) convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class XceptionUNet(nn.Module):
    def __init__(self, input_seq_len, output_seq_len,
                 input_channels_per_step=1, output_channels_per_step=1,
                 init_features=32, depth=3, blocks_per_stage=2):
        super(XceptionUNet, self).__init__()

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.input_channels_per_step = input_channels_per_step
        self.output_channels_per_step = output_channels_per_step
        self.depth = depth

        total_in_channels = input_seq_len * input_channels_per_step
        total_out_channels = output_seq_len * output_channels_per_step

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # Initial conv to set initial channels
        self.conv_in = nn.Sequential(
            nn.Conv2d(total_in_channels, init_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
        )
        current_channels = init_features

        # Encoder path
        for i in range(depth):
            encoder_stage_modules = []
            stage_in_channels = current_channels
            stage_out_channels = init_features * (2 ** i)

            # First block may change channels
            encoder_stage_modules.append(SeparableConvBlock(stage_in_channels, stage_out_channels))
            for _ in range(1, blocks_per_stage):
                encoder_stage_modules.append(SeparableConvBlock(stage_out_channels, stage_out_channels))

            self.encoders.append(nn.Sequential(*encoder_stage_modules))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = stage_out_channels

        # Bottleneck
        bottleneck_modules = []
        for _ in range(blocks_per_stage):
            bottleneck_modules.append(SeparableConvBlock(current_channels, current_channels))
        self.bottleneck = nn.Sequential(*bottleneck_modules)

        # Decoder path
        for i in reversed(range(depth)):
            target_channels_after_upsample = init_features * (2 ** i)

            self.upsamples.append(
                nn.ConvTranspose2d(
                    current_channels, target_channels_after_upsample,
                    kernel_size=2, stride=2
                )
            )

            concat_channels = target_channels_after_upsample * 2

            decoder_stage_modules = []
            decoder_stage_modules.append(
                SeparableConvBlock(concat_channels, target_channels_after_upsample)
            )
            current_out_decoder_stage = target_channels_after_upsample

            for _ in range(1, blocks_per_stage):
                decoder_stage_modules.append(
                    SeparableConvBlock(current_out_decoder_stage, current_out_decoder_stage)
                )

            self.decoders.append(nn.Sequential(*decoder_stage_modules))
            current_channels = current_out_decoder_stage

        # Final output conv
        self.conv_out = nn.Conv2d(current_channels, total_out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, T_in, C_in_step, H, W)
        b, t_in, c_in_step, h_orig, w_orig = x.shape
        x = x.view(b, t_in * c_in_step, h_orig, w_orig)  # combine time and channels

        x_init = self.conv_in(x)

        skip_connections = []
        x_enc = x_init

        # Encoder
        for i in range(self.depth):
            x_enc = self.encoders[i](x_enc)
            skip_connections.append(x_enc)
            x_enc = self.pools[i](x_enc)

        x_bottleneck = self.bottleneck(x_enc)

        # Decoder
        x_dec = x_bottleneck
        skip_connections = skip_connections[::-1]

        for i in range(self.depth):
            x_dec = self.upsamples[i](x_dec)
            skip_to_concat = skip_connections[i]

            # If shapes mismatch due to odd sizes, align by interpolation
            if x_dec.shape[2:] != skip_to_concat.shape[2:]:
                x_dec = F.interpolate(
                    x_dec, size=skip_to_concat.shape[2:], mode='bilinear', align_corners=False
                )

            x_dec = torch.cat((skip_to_concat, x_dec), dim=1)
            x_dec = self.decoders[i](x_dec)

        out = self.conv_out(x_dec)
        # Reshape to (B, T_out, C_out_step, H, W)
        out = out.view(b, self.output_seq_len, self.output_channels_per_step, h_orig, w_orig)
        return out
