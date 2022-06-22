# EEG-based Reading Skill Evaluation Model Design with DeepLearning Architecture
# Authors: Jihyeok Kim dkdnfk314@naver.com
# MODEL III: Braindecode Shallownet + GRU Model with two (Raw data & Maxpooling) Connection Network
# Move this file to "~~your environment dir\Lib\site-packages\braindecode\models"

from torch import nn
from torch.nn import init
import torch
from braindecode.models.modules import Expression
from braindecode.models.functions import (
    safe_log, square,  squeeze_final_output
)

class ShallowFBCSPNet(nn.Sequential):
    """Shallow ConvNet model from Schirrmeister et al 2017.

    Model described in [Schirrmeister2017]_.

    Parameters
    ----------
    in_chans : int
        XXX

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., Springenberg, J. T., Fiederer,
       L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F.
       & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017.
       Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(
        self,
        in_chans,
        n_classes,
        input_window_samples=None,
        n_filters_time=40,
        filter_time_length=25,
        n_filters_spat=40,
        pool_time_length=57,
        pool_time_stride=11,
        final_conv_length=30,
        conv_nonlin=square,
        pool_mode="mean",
        pool_nonlin=safe_log,
        split_first_layer=True,
        batch_norm=True,
        batch_norm_alpha=0.1,
        drop_prob=0.5,
        feature_length = 400
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None
        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.n_filters_time = n_filters_time
        self.filter_time_length = filter_time_length
        self.n_filters_spat = n_filters_spat
        self.pool_time_length = pool_time_length
        self.pool_time_stride = pool_time_stride
        self.final_conv_length = final_conv_length
        self.conv_nonlin = conv_nonlin
        self.pool_mode = pool_mode
        self.pool_nonlin = pool_nonlin
        self.split_first_layer = split_first_layer
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.drop_prob = drop_prob
        self.feature_length = feature_length
        self.hidden_size = 55
        
        
        
        
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]

        # conv1 - time
        # for the 1st connection network
        self.conv_time = nn.Conv2d(1, 32,
                                    (self.filter_time_length, 1),
                                    stride=1, padding=(13,0))
        # for the 2nd connection network
        self.conv_time2 = nn.Conv2d(1, self.n_filters_time,
                                    (self.filter_time_length, 1),
                                    stride=1)
        
        # conv2 - spat
        # for the 1st connection network
        self.conv_spat = nn.Conv2d(32, 32,
                                    (1, self.in_chans),
                                    stride=1, bias=not self.batch_norm)
        # for the 2nd connection network
        self.conv_spat2 = nn.Conv2d(self.n_filters_time, self.n_filters_spat,
                                    (1, self.in_chans),
                                    stride=1, bias=not self.batch_norm)
        
        n_filters_conv = self.n_filters_spat
        
        self.bnorm1 = nn.BatchNorm2d(32, momentum=self.batch_norm_alpha, affine=True)
        self.bnorm = nn.BatchNorm2d(n_filters_conv, momentum=self.batch_norm_alpha, affine=True)
        
        self.conv_nonlin_exp = Expression(self.conv_nonlin)
        
        self.pool = pool_class(
            kernel_size=(self.pool_time_length, 1),
            stride=(self.pool_time_stride, 1),
        )
        
        self.pool_nonlin_exp = Expression(self.pool_nonlin)

        


        self.drop = nn.Dropout(p=self.drop_prob)
        self.relu1 = nn.ReLU()
        self.elu = nn.ELU()
        self.squeeze = Expression(squeeze_final_output)
        
        # Pooling layer for the first connection
        self.first_pool = nn.MaxPool2d(kernel_size=(27 , 1), stride=1, padding = (13, 0))
        
        # Pooling layer for the second connection
        self.last_pool = nn.MaxPool2d(kernel_size=(37,13), stride=(25,1), padding=0)
        
        # Additional Convolutional layer
        self.conv_last = nn.Conv2d(1, 1, 4, stride=2, padding=1)
        
        self.flatten = nn.Flatten(2,3)
        
        # GRU Layer 추가
        # [input_size, hidden_size, num_layers]
        self.GRU = nn.GRU(self.feature_length, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linear1 = nn.Linear(self.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(self.conv_time.weight, gain=1)
        init.xavier_uniform_(self.conv_time2.weight, gain=1)
        
        # maybe no bias in case of no split layer and batch norm
        if self.split_first_layer or (not self.batch_norm):
            init.constant_(self.conv_time.bias, 0)
            init.constant_(self.conv_time2.bias, 0)
            
        if self.split_first_layer:
            init.xavier_uniform_(self.conv_spat.weight, gain=1)
            init.xavier_uniform_(self.conv_spat2.weight, gain=1)
            
            if not self.batch_norm:
                init.constant_(self.conv_spat.bias, 0)
                init.constant_(self.conv_spat2.bias, 0)
        if self.batch_norm:
            init.constant_(self.bnorm.weight, 1)
            init.constant_(self.bnorm.bias, 0)
            init.constant_(self.bnorm1.weight, 1)
            init.constant_(self.bnorm1.bias, 0)

    
    def forward(self, x):
        # (batch, sequence_length, 1, channel, window length)
        if not self.split_first_layer:
            print("Make it split first layer")
            return 0
            
        
        x3 = x.permute(0, 1, 2, 4, 3)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device).requires_grad_()
        x2 = x3.view(-1, x3.shape[2], x3.shape[3], x3.shape[4])


        x = self.conv_time(x2)
        x = self.bnorm1(x)
        x = self.elu(x)
        
        x = self.conv_spat(x)
        
        if self.batch_norm:
            x = self.bnorm1(x)
        
        x = self.conv_nonlin_exp(x)
        
        # (N, 32, 512, 1) -> (N, 1, 512, 32)
        x = x.permute(0, 3, 2, 1)
        
        # skip connection
        x_1 = x + x2
        
        x_skip2 = self.last_pool(x_1)
        
        # second block
        x_1 = self.conv_time2(x_1)
        x_1 = self.bnorm(x_1)
        x_1 = self.elu(x_1)
        
        
        x_1 = self.conv_spat2(x_1)
        if self.batch_norm:
            x_1 = self.bnorm(x_1)
        x_1 = self.conv_nonlin_exp(x_1)
        
        
        x_1 = self.pool(x_1)
        x_1 = self.drop(x_1)
        
        
        # (N, 32, 512, 1) -> (N, 1, 512, 32)
        x_1 = x_1.permute(0, 3, 2, 1)

        x_1 = self.conv_last(x_1)
        
        
        # max pool connection
        x = x_1 + x_skip2                    
        
        # (1, 1, 20, 20)
        x = x.view(x3.shape[0], x3.shape[1], -1)

        # (b, seq_length, 400)
        y, h0 = self.GRU(x, h0.detach())
        
        # y: batch_size, seq_length, hidden_size
        y = y[:, -1, :]

        x = self.linear1(y)
        x = self.softmax(x)
        return x