import argparse
from core.update import UpdateBlock
from core.extractor import GhostNet
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import *

try:
    autocast = torch.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

parser = argparse.ArgumentParser()
parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
parser.add_argument('--train_iters', type=int, default=22,
                    help="number of updates to the disparity field in each forward pass.")
parser.add_argument('--valid_iters', type=int, default=32,
                    help='number of flow-field updates during validation forward pass')
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                    help="hidden state and context dimensions")

class hourglass(nn.Module):
    def __init__(self, in_channels, channel):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels * 2, channel[1])
        self.feature_att_16 = FeatureAtt(in_channels * 4, channel[2])
        self.feature_att_32 = FeatureAtt(in_channels * 6, channel[3])
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, channel[2])
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, channel[1])

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, norm_fn='batch'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = self._make_conv(c1, 2 * self.c, 1, 1, norm_fn)
        self.cv2 = self._make_conv((2 + n) * self.c, c2, 1, 1, norm_fn)
        self.m = nn.ModuleList([self._make_bottleneck(self.c, self.c, shortcut, g, norm_fn) for _ in range(n)])
        self.norm_fn = norm_fn

    def _make_conv(self, c1, c2, k, s, norm_fn):
        conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        norm = self._get_norm_layer(c2, norm_fn)
        return nn.Sequential(conv, norm, nn.SiLU())

    def _get_norm_layer(self, channels, norm_fn):
        if norm_fn == 'group':
            num_groups = min(8, channels // 4)
            return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        elif norm_fn == 'batch':
            return nn.BatchNorm2d(channels)
        elif norm_fn == 'instance':
            return nn.InstanceNorm2d(channels)
        elif norm_fn == 'none':
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported norm_fn: {norm_fn}")

    def _make_bottleneck(self, c1, c2, shortcut, g, norm_fn):
        c_ = int(c2 * 1.0)
        block = nn.Sequential(
            nn.Conv2d(c1, c_, 3, 1, 1, bias=False),
            self._get_norm_layer(c_, norm_fn),
            nn.SiLU(),
            nn.Conv2d(c_, c2, 3, 1, 1, groups=g, bias=False),
            self._get_norm_layer(c2, norm_fn),
            nn.SiLU()
        )
        block.add = shortcut and c1 == c2
        return block

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            if hasattr(m, 'add') and m.add:
                y.append(y[-1] + m[:-2](y[-1]))
            else:
                y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class Transfer(nn.Module):
    def __init__(self, channels, output_dim, *args):
        super(Transfer, self).__init__()
        self.chans = channels
        self.output_dim = output_dim
        self.transfer_operation = nn.ModuleList(
            [BasicConv(c, 128, kernel_size=3, stride=1, padding=1) for c in self.chans])
        self.c4 = nn.ModuleList([BottleneckCSP(128, dim[0], 2, True, norm_fn='instance') for dim in self.output_dim])
        self.c8 = nn.ModuleList([BottleneckCSP(128, dim[1], 2, True, norm_fn='instance') for dim in self.output_dim])
        self.c16 = nn.ModuleList([BottleneckCSP(128, dim[2], 2, True, norm_fn='instance') for dim in self.output_dim])

    def forward(self, x):
        with autocast('cuda', enabled=True):
            for i, item in enumerate(x):
                x[i] = self.transfer_operation[i](item)
            x4, x8, x16 = x
            x4 = [c4(x4) for c4 in self.c4]
            x8 = [c8(x8) for c8 in self.c8]
            x16 = [c16(x16) for c16 in self.c16]
            return x4, x8, x16

class IGEVStereoLite(nn.Module):
    def __init__(self, max_disp=192, cat_dim=3):

        super().__init__()
        self.args = parser.parse_args()
        self.max_disp = max_disp
        self.cat_dim = cat_dim

        context_dims = self.args.hidden_dims
        self.feature = GhostNet()
        self.chans = self.feature.chans
        self.chans.pop(0)
        self.chans[0] = self.chans[0]*2+48
        self.chans[1] = self.chans[1]*2
        self.chans[2] = self.chans[2]*2
        self.cnet = Transfer(self.chans[:3], [self.args.hidden_dims, context_dims])

        self.update_block = UpdateBlock(self.args, hidden_dims=self.args.hidden_dims)
        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], self.args.hidden_dims[i]*2, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
        )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8, self.chans)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        with autocast('cuda', enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)
            spx_pred = self.spx_gru(xspx)
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)

        return up_disp

    def forward(self, x):
        """ Estimate disparity between pair of frames """
        if self.training:
            iters = self.args.train_iters
        else:
            iters = self.args.valid_iters

        with autocast('cuda', enabled=self.args.mixed_precision):
            features = self.feature(x)
            stem_2 = self.stem_2(x)
            stem_4x = self.stem_4(stem_2)

            features[0] = torch.cat((features[0], stem_4x), 1)

            match = self.desc(self.conv(features[0]))

            match_left, match_right = torch.chunk(match, 2, dim=self.cat_dim)
            stem_2x, stem_2y = torch.chunk(stem_2, 2, dim=self.cat_dim)

            features_left, features_right = [None]*4, [None]*4
            for i, item in enumerate(features):
                features_left[i], features_right[i] = torch.chunk(item, 2, dim=self.cat_dim)

            gwc_volume = build_gwc_volume(match_left, match_right, self.max_disp // 4, 8)
            gwc_volume = self.corr_stem(gwc_volume)
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.max_disp // 4)

            del prob, gwc_volume

            if self.training:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            cnet_list = self.cnet(features_left[:3])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 2, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(),
                           radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp
        disp_preds = []
        disp_up = None

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords)
            with autocast('cuda', enabled=self.args.mixed_precision):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp,
                                                                      iter16=self.args.n_gru_layers == 3,
                                                                      iter08=self.args.n_gru_layers >= 2)

            disp = disp + delta_disp
            if not self.training and itr < iters - 1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if not self.training:
            if disp_up is not None:
                return disp_up
            else:
                with autocast('cuda', enabled=self.args.mixed_precision):
                    xspx = self.spx_4(features_left[0])
                    xspx = self.spx_2(xspx, stem_2x)
                    spx_pred = self.spx(xspx)
                    spx_pred = F.softmax(spx_pred, 1)
                return context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)

        init_disp = context_upsample(init_disp * 4., spx_pred.float()).unsqueeze(1)

        return init_disp, disp_preds




