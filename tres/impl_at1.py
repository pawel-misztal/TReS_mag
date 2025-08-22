import torch
from torch import nn, Tensor
import torchvision.models as models
from typing import Tuple, List
import torch.nn.functional as F
from helpersmag.accuracy import calc_SRCC, calc_PLCC


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedd_dim:int=64, # parametr d z pracy
                 width:int = 7,
                 num_heads:int=16, 
                 dropout:float=0,
                 addPoseEveryLayer = False,
                 normalizePose = False,
                 normalizeBefore = False,
                 layerNormBias = True):
        super().__init__()

        self.msa = nn.MultiheadAttention(
            embed_dim=embedd_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(normalized_shape=embedd_dim, bias=layerNormBias)
        if(addPoseEveryLayer):
            self.poseEncode = PosSineEncoding(width,embedd_dim,normalizePose)
        self.addPoseEveryLayer = addPoseEveryLayer
        self.normalizeBefore = normalizeBefore


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if(self.normalizeBefore):
            x = self.layerNorm.forward(x)


        if(self.addPoseEveryLayer):
            q = k = self.poseEncode.forward(x)
            v = x
        else:
            q = k = v = x
        
        atten, _ = self.msa.forward(query=q,
                                    key=k,
                                    value=v,
                                    need_weights=False)
        
        add = x + self.dropout(atten)

        if(self.normalizeBefore == False):
            normalized = self.layerNorm.forward(add)
        else:
            normalized = add

        return normalized
    
class FeedForwardNetworkBlock(nn.Module):
    def __init__(self,
                 embed_dim:int=3840,
                 ffn_size:int=64, # wartość 4 została wzięta z "Attention is all you need", w pracy nie podano ile ffn posiada neuronów # parametr d z pracy
                 dropout:int=0.1,
                 activation_function=nn.ReLU(),
                 last_ffn_bias=True,
                 layerNorm_bias=True,
                 normalizeBefore = False,
                 ffn_extraDropout=True
                 ):
        super().__init__()
        self.normalizeBefore = normalizeBefore

        self.ffn = nn.Sequential(
            nn.Linear(in_features=embed_dim,
                      out_features=ffn_size),
            activation_function,
            nn.Dropout(dropout),
            nn.Linear(in_features=ffn_size,
                      out_features=embed_dim,
                      bias=last_ffn_bias),
            nn.Dropout(dropout)
        )

        #TODO: dodac ekstra funkcje aktywacji
        #TODO: przetestować inne funkcje aktywacji, gelu, lrelu 
        self.ffn_extraDropout = ffn_extraDropout
        if(ffn_extraDropout):
            self.dropout = nn.Dropout(dropout)

        self.layerNorm = nn.LayerNorm(normalized_shape=embed_dim, bias=layerNorm_bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if(self.normalizeBefore):
            x = self.layerNorm.forward(x)
        ffn = self.ffn.forward(x)

        if(self.ffn_extraDropout):
            add = self.dropout(ffn) + x
        else:
            add = ffn + x
        
        if(self.normalizeBefore == False):
            norm = self.layerNorm.forward(add)
        else:
            norm = add

        return norm


class TransformerBlock(nn.Module):
    def __init__(self,
                 embedd_dim:int=3840, 
                 width:int=7,
                 num_heads:int=2, 
                 mhsa_dropout:float=0, # MultiHeadSelfAttentionDropout
                 mhsa_addPoseEveryLayer = False,
                 mhsa_normalizePose = False,
                 mhsa_layerNorm_bias = True,
                 ffn_size:int=64, # wartość 4 została wzięta z "Attention is all you need", w pracy nie podano ile ffn posiada neuronów # parametr d z pracy
                 ffn_dropout:int=0.1,
                 ffn_activation_function=nn.ReLU(),
                 ffn_last_ffn_bias=True,
                 ffn_layerNorm_bias=True,
                 ffn_extraDropout=True,
                 normalizeBefore = False,
                 init_xavier = True
                 ):
        super().__init__()

        self.mhsa = MultiHeadSelfAttentionBlock(
            embedd_dim=embedd_dim,
            num_heads=num_heads,
            dropout=mhsa_dropout,
            addPoseEveryLayer=mhsa_addPoseEveryLayer,
            normalizePose=mhsa_normalizePose,
            normalizeBefore=normalizeBefore,
            layerNormBias=mhsa_layerNorm_bias,
            width=width
        )

        self.ffn = FeedForwardNetworkBlock(
            embed_dim=embedd_dim,
            ffn_size=ffn_size,
            dropout=ffn_dropout,
            activation_function=ffn_activation_function,
            last_ffn_bias=ffn_last_ffn_bias,
            layerNorm_bias=ffn_layerNorm_bias,
            normalizeBefore=normalizeBefore,
            ffn_extraDropout=ffn_extraDropout
        )
        self.normalizeBefore = normalizeBefore
        self.layerNorm = nn.LayerNorm(normalized_shape=embedd_dim)

        # TODO:opcjonalnie wyłączyć inicjalizacje 
        if init_xavier:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        #         nn.init.kaiming_normal_(p,nonlinearity="relu")
        #         if p.bias is not None:
        #             nn.init.zeros_(p.bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.mhsa.forward(x)
        x = self.ffn.forward(x)
        # if(self.normalizeBefore):
        #     x = self.layerNorm.forward(x)
        # TODO: przenieść layernorm poza transformery
        return x
    

class ConvNeXt_TinyBlocks(models.ConvNeXt):
    def __init__(self,  **kwargs):

        block_setting = [
            models.convnext.CNBlockConfig(96, 192, 3),
            models.convnext.CNBlockConfig(192, 384, 3),
            models.convnext.CNBlockConfig(384, 768, 9),
            models.convnext.CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
        super().__init__(block_setting, stochastic_depth_prob,   **kwargs)

        models.convnext_tiny

    def forward(self, x:Tensor) -> List[Tensor]:
        x = self.features[0].forward(x)
        x1 = self.features[1].forward(x)
        x2 = self.features[2].forward(x1)
        x3 = self.features[3].forward(x2)
        x4 = self.features[4].forward(x3)
        x5 = self.features[5].forward(x4)
        x6 = self.features[6].forward(x5)
        x7 = self.features[7].forward(x6)

        return [x1,x3,x5,x7]

def createPretrainedConvNeXt_TinyBlocks(weights, progress=True) -> Tuple[ConvNeXt_TinyBlocks, int,int,int, List[int]]:
    """
    return[resnet, embeddim, w, h]
    """
    resnet = ConvNeXt_TinyBlocks()

    if weights is not None:
        resnet.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    embedd_dim = 1440 # added length of layers from 1 to 4
    last_layer_width = 7
    last_layer_height = 7 
    layers_sizes = [96,192,384,768]

    return resnet, embedd_dim, last_layer_width, last_layer_height, layers_sizes

class Resnet50Blocks(models.ResNet):
    def __init__(self):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])

    def forward(self, x:Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
        """
        return: [layer1, layer2, layer3, layer]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        # return x
        
        return l1, l2, l3, l4



def createPretrainedResnet50Blocks(weights, progress=True) -> Tuple[Resnet50Blocks, int,int,int, List[int]]:
    """
    return[resnet, embeddim, w, h]
    """
    resnet = Resnet50Blocks()

    if weights is not None:
        resnet.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    embedd_dim = 3840 # added length of layers from 1 to 4
    last_layer_width = 7
    last_layer_height = 7 
    layers_sizes = [256,512,1024,2048]

    return resnet, embedd_dim, last_layer_width, last_layer_height, layers_sizes

class EfficientNetB4Blocks(models.EfficientNet):
    def __init__(self):
        inverted_residual_setting, last_channel = models.efficientnet._efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
        super().__init__(inverted_residual_setting, 0.4, 0.2, 1000, None, last_channel)

    def forward(self, x:Tensor) -> List[Tensor]:
        # x = self.features(x)
        x =  self.features[0].forward(x)
        b1 = self.features[1].forward(x)
        b2 = self.features[2].forward(b1)
        b3 = self.features[3].forward(b2)
        b4 = self.features[4].forward(b3)
        b5 = self.features[5].forward(b4)
        b6 = self.features[6].forward(b5)
        b7 = self.features[7].forward(b6)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)

        # x = self.classifier(x)

        return [b1,b2,b3,b4,b5,b6,b7]


def createPretrainedEfficientNetB4Blocks(weights, progress=True) -> Tuple[EfficientNetB4Blocks, int,int,int, List[int]]:
    """
    return[resnet, embeddim, w, h]
    """
    resnet = EfficientNetB4Blocks()

    if weights is not None:
        resnet.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    embedd_dim = 1104 # added length of layers from 1 to 4
    last_layer_width = 7
    last_layer_height = 7 
    layers_sizes = [24,32,56,112,160,272,448]

    return resnet, embedd_dim, last_layer_width, last_layer_height, layers_sizes

def nancheck( x:Tensor, name) -> Tensor:
        print(name, "Is nan:", x.isnan().any().item(), " min:", x.min().item(), " max:", x.max().item())


class L2Pooling(nn.Module):
    def __init__(self, channels:int, filterSize:int = 3):
        super().__init__()

        #TODO change to hanning and change size 
        hamming = torch.hamming_window(filterSize, periodic=False)
        hammingFilter = hamming[:,None] * hamming[None,:]
        hammingFilter = hammingFilter / torch.sum(hammingFilter)

        # print("hamm", hammingFilter)

        # self.conv = nn.Conv2d(in_channels=channels,
        #                       out_channels=channels,
        #                       groups=channels,
        #                       padding=filterSize // 2,
        #                       kernel_size=filterSize,
        #                       bias=False)
        # self.filter = torch.zeros(self.conv.weight.shape)
        # self.filter[:,:,:,:] = hammingFilter 

        self.register_buffer("filter", hammingFilter[None,None,:,:].repeat((channels,1,1,1)))
        # self.filter = nn.Parameter(hammingFilter.expand(channels, 1, 3, 3), requires_grad=False)
        # self.conv.weight = self.filter
        
        # self.conv.weight = nn.Parameter(self.filter, requires_grad=False)

    def forward(self, x:Tensor) -> Tensor:
        # nancheck(x, "x")
        # dot = torch.mul(x,x)
        dot = x.pow(2)
        # nancheck(dot, "dot")
        # conv = self.conv.forward(dot)
        conv = F.conv2d(dot, self.filter, stride=1, padding=1, groups=x.shape[1])
        # dotXconv = dot * conv
        # nancheck(conv, "conv")
        sqrt = torch.sqrt(conv + 1e-12)
        # nancheck(sqrt, "sqrt")
        return sqrt
    
class L2PoolingPaper(nn.Module):
    def __init__(self, channels:int, filterSize:int = 3):
        super().__init__()
        #TODO change to hanning and change size 
        hamming = torch.hann_window(filterSize+2, periodic=False)[1:-1]
        hammingFilter = hamming[:,None] * hamming[None,:]
        hammingFilter = hammingFilter / torch.sum(hammingFilter)

        self.register_buffer("filter", hammingFilter[None,None,:,:].repeat((channels,1,1,1)))\

    def forward(self, x:Tensor) -> Tensor:
        dot = x.pow(2)
        conv = F.conv2d(dot, self.filter, stride=1, padding=1, groups=x.shape[1])
        sqrt = torch.sqrt(conv + 1e-12)
        return sqrt
    
class PosSineEncoding(nn.Module):
    def __init__(self, size, embedDim, normalize = False):
        super().__init__()

        halfEbmbedDim = embedDim // 2
        
        halfPE = torch.zeros((size,halfEbmbedDim))
        positions = torch.arange(0, size, dtype=torch.float32)
        positions = positions.unsqueeze(1)

        if(normalize):
            positions = positions / positions.max() * 2 * torch.pi

        div = 1 / torch.pow(10000, torch.arange(0,halfEbmbedDim, 2) / halfEbmbedDim)

        halfPE[:,0::2] = torch.sin(positions * div)
        halfPE[:,1::2] = torch.cos(positions * div)

        squareHalfPe = torch.zeros((size,size,halfEbmbedDim))
        squareHalfPe[:,:,:] = halfPE
        pe = torch.cat((squareHalfPe,squareHalfPe.transpose(0,1)), dim=2)

        self.register_buffer("pe", pe.unsqueeze(0))


    def forward(self, x:Tensor) -> Tensor:
        pe = self.pe.flatten(1,2)
        return x + pe 
    
class CnnNormalizer(nn.Module):
    def __init__(self,
                 layers_sizes:List[int],
                 last_layer_width,
                 last_layer_height,
                 l2Pool:L2Pooling|L2PoolingPaper=L2Pooling):
        super().__init__()
        self.layers_sizes = layers_sizes

        self.dropoutLayer = nn.Dropout(p=0.1)
        self.aAvgPool2d = nn.AdaptiveAvgPool2d((last_layer_width,last_layer_height))


        self.l2Pools = nn.ModuleList()
        for size in layers_sizes:
            self.l2Pools.append(l2Pool(size))


    def forward(self, x:List[Tensor] | Tuple[Tensor]) -> List[Tensor]:
        outs = []
        for lx, l2pool in zip(x,self.l2Pools):
            pooled = self.aAvgPool2d(self.dropoutLayer(l2pool.forward(F.normalize(lx,p=2,eps=1e-12, dim=1))))
            outs.append(pooled)

        return outs

class Tres(nn.Module):
    def __init__(self, 
                 mhsa_dropout=0.1, 
                 mhsa_addPoseEveryLayer = False,
                 ffn_dropout=0.1, 
                 fc_trans_dropout = 0,
                 fc_last_dropout = 0,
                 normalizePosEncode = False,
                 normalizeBefore = False,
                 ffn_extraDropout =True,
                 ffn_size=64,
                 cnn_name="resnet50",
                 init_xavier=True,
                 extraNormalizeAfter=False,
                 num_trans_encoders = 2,
                 one_more_linear = False,
                 l2_pool_paper = False):
        super().__init__()

        self.mhsa_addPoseEveryLayer = mhsa_addPoseEveryLayer

        print("using cnn:",cnn_name)
        if(cnn_name == "resnet50"):
            self.cnn, self.embedd_dim, self.ll_width, self.ll_height, self.layers_sizes = createPretrainedResnet50Blocks(models.ResNet50_Weights.IMAGENET1K_V2)
        elif(cnn_name == "effnetb4"):
            self.cnn, self.embedd_dim, self.ll_width, self.ll_height, self.layers_sizes = createPretrainedEfficientNetB4Blocks(models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        elif(cnn_name == "convnext_tiny"):
            self.cnn, self.embedd_dim, self.ll_width, self.ll_height, self.layers_sizes = createPretrainedConvNeXt_TinyBlocks(models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        else:
            raise Exception("Cnn name is not recogized", cnn_name)

        self.aAvgPool2d = nn.AdaptiveAvgPool2d((self.ll_width,self.ll_height))
        self.aAvgPool2dL4 = nn.AdaptiveAvgPool2d((1,1))
        self.aAvgPool2dT = nn.AdaptiveAvgPool1d((1))

        self.flatten = nn.Flatten(2,3)

        #TODO: ilość warstw transformera jako parametr
        self.transformer = nn.Sequential( *[TransformerBlock(embedd_dim=self.embedd_dim, 
                                                             ffn_size=ffn_size, 
                                                             mhsa_dropout=mhsa_dropout, 
                                                             ffn_dropout=ffn_dropout,
                                                             mhsa_addPoseEveryLayer=mhsa_addPoseEveryLayer,
                                                             mhsa_normalizePose=normalizePosEncode,
                                                             normalizeBefore=normalizeBefore,
                                                             ffn_extraDropout=ffn_extraDropout,
                                                             width=self.ll_width,
                                                             init_xavier=init_xavier) for _ in range(num_trans_encoders)])

        self.posEncode = PosSineEncoding(self.ll_height,self.embedd_dim, normalizePosEncode)

        self.extraNormalizeAfter = extraNormalizeAfter
        if(extraNormalizeAfter):
            self.tran_layer_norm = nn.LayerNorm(self.embedd_dim)

        self.cnnNormalizer = CnnNormalizer(self.layers_sizes, self.ll_width, self.ll_height, L2PoolingPaper if l2_pool_paper else L2Pooling)

        # self.l2pool1 = L2Pooling(256)
        # self.l2pool2 = L2Pooling(512)
        # self.l2pool3 = L2Pooling(1024)
        # self.l2pool4 = L2Pooling(2048)

        # self.dropoutLayer = nn.Dropout(p=0.1)

        self.fc_trans = nn.Linear(self.embedd_dim,self.layers_sizes[-1])
        self.fc_trans_dropout = nn.Dropout(fc_trans_dropout)

        self.fc_last = nn.Linear(2*self.layers_sizes[-1], 1)
        self.fc_last_dropout = nn.Dropout(fc_last_dropout)

        self.one_more_linear = one_more_linear
        if(one_more_linear):
            self.last_linear = nn.Linear(1,1)
    
        # if(x.isnan().any()):

    def forward(self, x:Tensor) :

        # nancheck(x, "x tres")
        cnnLayers = self.cnn.forward(x)

        # nancheck(l1, "l1 org")
        l4org = cnnLayers[-1]
        outl4 = cnnLayers[-1]

        # l1 = self.aAvgPool2d(self.dropoutLayer(self.l2pool1.forward(F.normalize(l1,p=2,eps=1e-12, dim=1))))
        # l2 = self.aAvgPool2d(self.dropoutLayer(self.l2pool2.forward(F.normalize(l2,p=2,eps=1e-12, dim=1))))
        # l3 = self.aAvgPool2d(self.dropoutLayer(self.l2pool3.forward(F.normalize(l3,p=2,eps=1e-12, dim=1))))
        # l4 = self.aAvgPool2d(self.dropoutLayer(self.l2pool4.forward(F.normalize(l4,p=2,eps=1e-12, dim=1))))

        normalizedLayers = self.cnnNormalizer.forward(cnnLayers)

        # nancheck(l1, 'l1')

        embedds = torch.cat(normalizedLayers, dim=1)


        embedds = self.flatten.forward(embedds).permute(0,2,1) #[b,49,3840]

        if(self.mhsa_addPoseEveryLayer):
            posEmbedds = embedds
        else:
            posEmbedds = self.posEncode.forward(embedds)


        # nancheck(posEmbedds, 'posEmbedds')

        oTrans:Tensor = self.transformer.forward(posEmbedds)
        if(self.extraNormalizeAfter):
            oTrans = self.tran_layer_norm.forward(oTrans)
        # nancheck(oTrans, 'oTrans')

        l4org = self.aAvgPool2dL4.forward(l4org).flatten(2).squeeze(2)
        oTrans = oTrans.permute(0,2,1)
        oTransOrg = oTrans
        oTrans = self.aAvgPool2dT.forward(oTrans).squeeze(2)

        oTrans = self.fc_trans_dropout.forward(oTrans)
        oTrans = self.fc_trans.forward(oTrans)

        last = torch.cat((l4org,oTrans),dim=1)

        last = self.fc_last_dropout.forward(last)
        out = self.fc_last.forward(last)

        if(self.one_more_linear):
            out = self.last_linear.forward(out)

        return out, outl4, oTransOrg

class CustomLoss(nn.Module):
    def __init__(self, w_mae = 1, w_triplet = 0.05, w_selfConf = 0.5, w_rankingSelfConf = 0.5):
        super().__init__()
        self.w_MAE = w_mae
        self.w_triplet = w_triplet
        self.w_selfConf = w_selfConf
        self.w_rankingSelfConf = w_rankingSelfConf

    def relative_ranking_loss(self, preds: Tensor, targets: Tensor) -> torch.Tensor:
        """
        Implementacja lossu Relative Ranking na podstawie artykułu TReS.
        preds: tensor predykcji (batch_size,)
        targets: tensor ocen subiektywnych (batch_size,)
        """

        # Detach na wszelki wypadek, ale może być pomijany przy trenowaniu
        preds = preds.view(-1)
        targets = targets.view(-1)

        if targets.numel() < 2:
            return torch.tensor(0.0)  # lub inna sensowna wartość


        assert preds.shape == targets.shape, "Rozmiary preds i targets muszą być zgodne"

        # Indeksy ekstremalnych przypadków
        idx_max = torch.argmax(targets)
        idx_min = torch.argmin(targets)

        # Drugi najwyższy i drugi najniższy
        without_max = torch.cat([targets[:idx_max], targets[idx_max+1:]])
        idx_smax = torch.argmax(without_max)
        idx_smax = idx_smax if idx_smax < idx_max else idx_smax + 1

        without_min = torch.cat([targets[:idx_min], targets[idx_min+1:]])
        idx_smin = torch.argmin(without_min)
        idx_smin = idx_smin if idx_smin < idx_min else idx_smin + 1

        # Predykcje
        qmax = preds[idx_max]
        q_smax = preds[idx_smax]
        qmin = preds[idx_min]
        q_smin = preds[idx_smin]

        # Subiektywne
        smax = targets[idx_max]
        s_smax = targets[idx_smax]
        smin = targets[idx_min]
        s_smin = targets[idx_smin]

        # Marginesy
        margin1 = s_smax - smin
        margin2 = smax - s_smin

        # Triplet loss (max(0, d1 - d2 + margin))
        d_max_smax = torch.abs(qmax - q_smax)
        d_max_min = torch.abs(qmax - qmin)
        loss1 = torch.relu(d_max_smax - d_max_min + margin1)

        d_smin_min = torch.abs(q_smin - qmin)
        loss2 = torch.relu(d_smin_min - d_max_min + margin2)

        return loss1 + loss2
    
    def mae_loss(self, preds:Tensor, target:Tensor) -> Tensor:
        return torch.mean(torch.abs(preds - target))


    def forward(self, preds:Tensor, target:Tensor, preds2: Tensor = None, selfLoss:Tensor = None) -> Tensor:

        mae = self.mae_loss(preds, target)
        rtl = self.relative_ranking_loss(preds, target)

        if(preds2 is None or selfLoss is None):
            selfConf = torch.tensor(0.0)
        else:
            rtl2 = self.relative_ranking_loss(preds2, target)
            selfConf = selfLoss + self.w_rankingSelfConf * torch.nn.functional.l1_loss(rtl, rtl2)

        return self.w_MAE * mae + self.w_triplet * rtl + self.w_selfConf * selfConf
    
class CustomLoss2(nn.Module):
    def __init__(self, w_mae = 1, w_srcc = 0.5, w_plcc = 0.5):
        super().__init__()
        self.w_MAE = w_mae
        self.w_plcc = w_plcc
        self.w_srcc = w_srcc

   
    
    def mae_loss(self, preds:Tensor, target:Tensor) -> Tensor:
        return torch.mean(torch.abs(preds - target))


    def forward(self, preds:Tensor, target:Tensor) -> Tensor:

        mae = self.mae_loss(preds, target)
        plcc = -1 * (calc_PLCC(preds, target) - 1)
        srcc = -1 * (calc_SRCC(preds, target) - 1)


        return self.w_MAE * mae  + plcc * self.w_srcc + srcc * self.w_srcc

class CustomLoss3(nn.Module):
    def __init__(self, w_hubert = 1, w_bce = 1):
        super().__init__()
        self.w_MAE = w_hubert
        self.w_bce = w_bce

        self.bce = nn.BCEWithLogitsLoss()
        self.hubert = nn.L1Loss()


    def forward(self, preds:Tensor, target:Tensor) -> Tensor:
        return self.bce.forward(preds, target) * self.w_bce + self.hubert.forward(preds, target) * self.w_MAE
