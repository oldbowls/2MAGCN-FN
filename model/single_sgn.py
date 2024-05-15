from torch import nn
import torch
from model.layers import *
from model.init_transforms import Transforms


class Single_SGN(nn.Module):
    def __init__(self, num_classes, num_joint, seg, bias=True, dim=256, adaptive_transform=False, num_joint_ori=25,
                 gcn_type='mid'):
        super(Single_SGN, self).__init__()

        self.seg = seg

        self.spa_net = SpatialNet(num_joint, bias, dim, gcn_type)
        self.tem_net = TempolNet(seg, bias, dim)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(dim * 2, num_classes)

    def forward(self, input):
        if len(input.shape) == 6:
            bs, s, c, step, num_joints, m = input.shape
            input = input.view(bs * s, c, step, num_joints, m)
        elif len(input.shape) == 5:
            bs, c, step, num_joints, m = input.shape
            s = 1
            input = input.permute(0, 4, 1, 3, 2).contiguous().view(bs * m * s, c, num_joints, step)  # nctvm->nmcvt
        else:
            bs, step, num_joints, c = input.shape
            s = 1
            m = 1
            input = input.permute(0, 3, 2, 1).contiguous()  # b,t,v,c -> b,c,v,t
        angle = input[:, 0:3, :, :]
        # dif = input[:, 3:, :, :]
        # Angle = input
        dif = torch.cat(
            [torch.zeros([*input.shape[:3], 1], device=input.device), input[:, :, :, 1:] - input[:, :, :, 0:-1]],
            dim=-1)
        # Acc = input[:, 3:6, :, :]
        # Gyro = input[:, 6:, :, :]

        input = self.spa_net(angle, dif)  # b c 1 t
        # input = self.spa_net(Angle, Acc)  # b c 1 t
        input = self.tem_net(input)  # b c 1 t
        # Classification
        output = self.maxpool(input)  # b c 1 1
        output = torch.flatten(output, 1)  # b c
        output = self.fc(output)  # b p
        output = output.view(bs, m * s, -1).mean(1)

        return output


if __name__ == '__main__':
    import os
    from model.flops_count import get_model_complexity_info
    from thop import profile
    from thop import clever_format
    num_class=11
    num_j_o = 5
    num_j = 5
    num_t = 10
    os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    model = Single_SGN(11, num_j, num_t, adaptive_transform=True, gcn_type='small', num_joint_ori=num_j_o)
    # torch.save(model.state_dict(), '../../pretrain_models/single_sgn_jpt{}.state'.format(num_j),)
    dummy_data = torch.randn([1, 3, num_t, num_j_o, 1])
    # a = model(dummy_data)
    # a.mean().backward()

    hooks = {}
    flops, params = profile(model, inputs=(dummy_data,), custom_ops=hooks)
    flops, params = clever_format([flops, params], "%.3f")
    print('flops:', flops, 'params:', params)
    #
    print(flops)
    print(params)

    flops, params = get_model_complexity_info(model, (3, num_t, 5, 1), as_strings=True)  # not support

    print(flops)  # 0.16 gmac
    print(params)  # 0.69 m
