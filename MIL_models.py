from MIL_utils import *

###################
# SIMCLR Models   #
###################

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d),
                            "resnet50": models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x


###################################
# Classic MIL Architecture Models #
###################################

class MILArchitecture(torch.nn.Module):

    def __init__(self, classes, freeze_bb_weights=False, pretrained=True, mode='embedding', aggregation='mean', backbone='vgg19', include_background=False):
        super(MILArchitecture, self).__init__()

        """Data Generator object for MIL.
            CNN based architecture for MIL classification.
        Args:
          classes: 
          mode:
          aggregation: max, mean, attentionMIL, mcAttentionMIL
          backbone:
          include_background:

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'

        self.classes = classes
        self.n_classes = len(classes)
        self.mode = mode
        self.aggregation = aggregation
        self.backbone = backbone
        self.include_background = include_background
        self.C = []
        self.prototypical = False
        self.pretained=pretrained
        self.freeze_bb_weights = freeze_bb_weights

        if self.include_background:
            self.nClasses = len(classes) + 1
        else:
            self.nClasses = len(classes)
        self.eps = 1e-6

        # Backbone
        self.bb = Encoder(pretrained=self.pretained, backbone=self.backbone, aggregation=True, freeze_bb_weights=self.freeze_bb_weights)

        # Classifiers
        # if self.aggregation == 'mcAttentionMIL':
        #     self.classifiers = torch.nn.ModuleList()
        #     for i in np.arange(0, self.nClasses):
        #         self.classifiers.append(torch.nn.Linear(512, 1))
        # else:
        if self.backbone == 'vgg16':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        if self.backbone == 'vgg16_bn':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        elif self.backbone == 'vgg19':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        elif self.backbone == 'resnet50':
            self.classifier = torch.nn.Linear(1000, self.nClasses)

        # MIL aggregation
        self.milAggregation = MILAggregation(aggregation=aggregation, nClasses=self.nClasses, mode=self.mode, backbone=self.backbone)


    def forward(self, images):
        # Patch-Level feature extraction
        features = self.bb(images)

        # # Modification for feature extraction for Arne
        # return features


        #patch_classification = self.classifier(torch.squeeze(features))
        #global_classification = patch_classification

        if self.mode == 'instance':
            # Classification
            patch_classification = torch.softmax(self.classifier(torch.squeeze(features)), 1)
            #patch_classification = torch.sigmoid(self.classifier(torch.squeeze(features)))
            # MIL aggregation
            global_classification = self.milAggregation(patch_classification)
            #print("holu")

        if self.mode == 'embedding' or self.mode == 'mixed':  # Activation on BCE loss
            #print(features.shape)
            # Embedding aggregation
            if features.shape[0] > 1:
                embedding = self.milAggregation(torch.squeeze(features))
            else:
                embedding = torch.squeeze(self.milAggregation(features))

            global_classification = self.classifier(torch.squeeze(embedding))
            patch_classification = self.classifier(torch.squeeze(features))
        if self.mode == 'embedding_GNN':  # Activation on BCE loss

            self.L = features.shape[1]


            features = torch.squeeze(features) #[Nx512]

            # Adjacency matrix A
            A = torch.ones((len(features), len(features))).cuda()  # pyg_ut.to_dense_adj(E_idx.cuda(), max_num_nodes=x.shape[0]) #

            # Embedding 1  (1 layer GraphSage followed by leaky ReLU activation
            # function (with negative slope equals
            # to 0.01) and batch normalization. Input and
            # output feature dimensions are the same.)

            self.gnn_embd = DenseSAGEConv(in_channels=self.L, out_channels=self.L).cuda() # GraphSAGE layer from https://arxiv.org/abs/1706.02216 (in their code it is always 50)
            Z = self.gnn_embd(features, A) # node embedding
                # After this, they apply Leaky Relu w negative slope=0.01
            Z = F.leaky_relu(Z, negative_slope=0.01)
            loss_emb_1 = self.auxiliary_loss(A, Z)

            # Clustering (1 layer GraphSage followed by leaky ReLU activation
            # function (with negative slope equals to 0.01) and batch
            # normalization. 1 extra layer of MLP with leaky ReLU
            # activation function applied to the output of GraphSage.
            # Input and output feature dimensions are the same.)

            self.gnn_pool = DenseSAGEConv(in_channels=self.L, out_channels=self.C).cuda()
            S = self.gnn_pool(Z, A) # node assignment matrix S
                # After this, they apply Leaky Relu w negative slope=0.01
            S = F.leaky_relu(S, negative_slope=0.01)
            self.mlp = nn.Linear(self.C, self.C, bias=True).cuda()
            S = self.mlp(S)
                # After this, they apply Leaky Relu w negative slope=0.01
            S = F.leaky_relu(S, negative_slope=0.01)
                        #S: Cluster assignment matrix
                        # X*A*S
                        # NUMBER OF CLUSTERS IS A HYPERPARAMETER
            # Coarsened graph
            #self.diff_pool = dense_diff_pool().cuda()
            X, A, l1, e1 = dense_diff_pool(Z, A, S) # The differentiable pooling operator from https://arxiv.org/abs/1806.08804

            # Embedding 2 (Same as embedding 1)
            X = self.gnn_embd(X, A)
                # After this, they apply Leaky Relu w negative slope=0.01
            X = F.leaky_relu(X, negative_slope=0.01)
            loss_emb_2 = self.auxiliary_loss(A, X)

            # Concat
            #X = X.view(1, -1)
            X = torch.squeeze(X)

            # MLP
            input_layers = int(self.L * self.C)
            hidden_layers = int(self.L * self.C / 2)
            output_layer = self.n_classes
            self.lin1 = nn.Linear(input_layers, hidden_layers, bias=True).cuda()
            self.lin2 = nn.Linear(hidden_layers, output_layer, bias=True).cuda()
            X = F.leaky_relu(self.lin1(X), 0.01)
            X = F.leaky_relu(self.lin2(X), 0.01) # logits

            #Y_prob = F.softmax(X.squeeze(), dim=0)
            L_gnn  = l1 + loss_emb_1 + loss_emb_2

            #Adapt to our code
            global_classification = torch.squeeze(X)

        if self.include_background:
            global_classification = global_classification[1:]

        # Adapt for compatibility w TransMIL aggr
        logits = global_classification
        Y_hat = torch.argmax(logits)
        Y_prob = F.softmax(logits, dim=0)

        if self.mode == 'embedding_GNN':
            return Y_prob, Y_hat, logits, L_gnn
        else:
            return Y_prob, Y_hat, logits

        # return global_classification, patch_classification, features


    # GNN Methods
    def auxiliary_loss(self, A, S):
        '''
            A: adjecment matrix {0,1} K x K
            S: nodes R K x D
        '''
        A = A.unsqueeze(0) if A.dim() == 2 else A
        S = S.unsqueeze(0) if S.dim() == 2 else S

        S = torch.softmax(S, dim=-1)

        link_loss = A - torch.matmul(S, S.transpose(1, 2))
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / A.numel()

        return link_loss

class Encoder(torch.nn.Module):

    def __init__(self, freeze_bb_weights=False, pretrained=True, backbone='resnet18', aggregation=False):
        super(Encoder, self).__init__()

        self.aggregation = aggregation
        self.pretrained = pretrained
        self.backbone = backbone
        self.freeze_bb_weights = freeze_bb_weights

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=self.pretrained)
            self.F = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
        if backbone == 'resnet50':
            resnet50 = torchvision.models.resnet50(pretrained=self.pretrained)
            self.F = resnet50

            # self.F = torch.nn.Sequential(resnet50.conv1,
            #                              resnet50.bn1,
            #                              resnet50.relu,
            #                              resnet50.maxpool,
            #                              resnet50.layer1,
            #                              resnet50.layer2,
            #                              resnet50.layer3,
            #                              resnet50.layer4,
            #                              resnet50.avgpool,
            #                              resnet50.fc)

        elif backbone == 'vgg16':
            vgg16 = torchvision.models.vgg16(pretrained=self.pretrained)
            self.F = vgg16.features
        elif backbone == 'vgg16_bn':
            vgg16_bn = torchvision.models.vgg16_bn(pretrained=self.pretrained)
            self.F = vgg16_bn.features
        elif backbone == 'vgg19':
            vgg19 = torchvision.models.vgg19(pretrained=self.pretrained)
            self.F = vgg19.features

        # Freeze backbone weights if required
        if self.freeze_bb_weights:
            for param in self.F.parameters():
                param.requires_grad = False

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x):
        out = self.F(x)

        # register the hook
        #h = out.register_hook(self.activations_hook)

        if self.backbone=="vgg16" or self.backbone=="vgg19" or self.backbone=="vgg16_bn":
            out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)

        return out

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

class MILAggregation(torch.nn.Module):
    def __init__(self, backbone, aggregation='mean', nClasses=1, mode='embedding'):
        super(MILAggregation, self).__init__()

        """Aggregation module for MIL.
        Args:
          aggregation:

        Returns:
          MILAggregation module for CNN MIL Architecture
        Last Updates: Claudio Fernandez Martín (05/07/22)
        """

        self.mode = mode
        self.aggregation = aggregation
        self.nClasses = nClasses
        self.backbone = backbone

        if self.aggregation == 'attention':
            if self.backbone == 'vgg16':
                input_dim = 512
            elif self.backbone == 'resnet50':
                input_dim = 1000

            self.attention_pooling = MILAttention(input_dim=input_dim)

        elif self.aggregation == 'TransMIL':
            self.transmil_pooling = TransMILPooling(n_classes=self.nClasses )


    def forward(self, feats):
        if self.aggregation == 'max':
            embedding = torch.max(feats, dim=0)[0]
            return embedding
        elif self.aggregation == 'attention':
            embedding, attention_weights  = self.attention_pooling(feats)
            return embedding
        elif self.aggregation == 'mean':
            embedding = torch.mean(feats, dim=0)
            return embedding
        elif self.aggregation == 'TransMIL':
            logits, embedding = self.transmil_pooling(feats)
            return embedding
            print("holu")
        # elif self.aggregation == 'transformer':
        #
        #     if feats.shape[0] == 1: # if there is only one instance in the bag
        #         feats = torch.squeeze(feats)
        #         feats = torch.unsqueeze(feats, dim=0)
        #
        #     feats = torch.unsqueeze(feats, dim=0) # expand dimension to batch size=1
        #     transformer_model = TransMIL(n_classes=4).cuda()
        #     embedding = transformer_model(data=feats).squeeze(dim=0)
        #     return embedding
        #
        # elif self.aggregation == 'transformer_v2':
        #
        #     if feats.shape[0] == 1: # if there is only one instance in the bag
        #         feats = torch.squeeze(feats)
        #         feats = torch.unsqueeze(feats, dim=0)
        #
        #     feats = torch.unsqueeze(feats, dim=0) # expand dimension to batch size=1
        #     transformer_model = TransMIL_v2(n_classes=4).cuda()
        #     embedding = transformer_model(feats).squeeze(dim=0)
        #     return embedding

class MILAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MILAttention, self).__init__()

        # Attention MIL embedding from Ilse et al. (2018) for MIL.
        # Class based on Julio Silva's MILAggregation class in PyTorch
        self.L = input_dim
        self.D = 128
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        A_V = self.attention_V(features)  # Attention
        A_U = self.attention_U(features)  # Gate
        w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)
        features = torch.transpose(features, 1, 0)
        embedding = torch.squeeze(torch.mm(features, w))  # MIL Attention
        return embedding, w

####################
# Patch GCN Models #
####################

class PatchGCN(torch.nn.Module):
    def __init__(self, input_dim=2227, num_layers=4, edge_agg='spatial', multires=False, resample=0,
                 fusion=None, num_features=1024, hidden_dim=128, linear_dim=64, use_edges=False, pool=False,
                 dropout=0.25, n_classes=4):
        super(PatchGCN, self).__init__()
        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers - 1
        self.resample = resample
        self.num_features = num_features

        if self.resample > 0:
            self.fc = nn.Sequential(*[nn.Dropout(self.resample), nn.Linear(self.num_features, 256), nn.ReLU(), nn.Dropout(0.25)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(self.num_features, 128), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)


        # Change to test with variable num_gcn_layers
        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * num_layers, D=hidden_dim * num_layers, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim * num_layers, n_classes)

        # self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(0.25)])
        #
        # self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 4, D=hidden_dim * 4, dropout=dropout, n_classes=1)
        # self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 4, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout)])
        #
        # self.classifier = torch.nn.Linear(hidden_dim * 4, n_classes)

    def forward(self, graph, edge_aggr='spatial'):
        #data = kwargs['x_path']
        self.edge_agg = edge_aggr

        if self.edge_agg == 'spatial':
            edge_index = graph['edge_index']
        elif self.edge_agg == 'latent':
            edge_index = graph['edge_latent']

        #batch = data.batch
        edge_attr = None # Aquí se podrían pasar los edge attributes (aka cuanta distancia)

        x = self.fc(graph['x'])
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h).unsqueeze(0)  # logits needs to be a [1 x 4] vector
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return Y_prob, Y_hat, logits


        # Original
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        #
        # return hazards, S, Y_hat, A_path, None

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

##############################
# TransMIL Models 11_07_2023 #
##############################

#########################################################################
######################## ViT AGGREGATION ################################
#########################################################################
## Reference for principal code: https://github.com/huawei-noah/CV-Backbones/blob/ac846b080a694df8e224000582a668b1bdc070a1/tnt_pytorch/tnt.py#L305
class TransMILPooling(nn.Module):
    def __init__(self, n_classes):
        super(TransMILPooling, self).__init__()
        self.n_classes = n_classes
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.final_fc = torch.nn.Sequential(torch.nn.Linear(512, self.n_classes))
        # self.backbone_extractor = AI4SKINClassifier()
        self.final_relu = torch.nn.ReLU()

    def forward(self, h):
        # h = kwargs['data'].float()  # [Bs, n, L]
        # h = self.backbone_extractor(x)
        # h = self._fc1(h)  # [B, n, 512]

        # ---->Pad
        h = h.reshape(1, h.shape[0], -1)
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
        # np.random.choice(np.arange(H), size=add_length, replace=False)

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]  # .squeeze(dim = 0)

        h = self.final_relu(h) # ----> ReLU para tener logits positivos

        # ---->Predict
        logits = self.final_fc(h)  # [BS=1, n_classes]
        return logits, h
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim = 512, dropout = 0.1):  # voy a utilizar una VGG asi que la dimensiÃ³n es 512
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = 8,
            num_landmarks = dim // 2,  # number of landmarks
            pinv_iterations = 6, # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True, # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = dropout
        ).cuda()

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)  # estudiar porque se utilizan estas convoluciones
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x






# ###################
# # TransMIL Models #
# ###################
#
# class TransMILArchitecture(nn.Module):
#     def __init__(self, n_classes, n_channels=3, transmil_bb="resnet50", pretrained_bb=True, freeze_bb_weights=False):
#         super(TransMILArchitecture, self).__init__()
#         self.L = 512  # 512 node fully connected layer
#         self.D = 128  # 128 node attention layer
#         self.K = 1
#         self.sqrtN = 12  # size for squaring
#         self.n_classes = n_classes
#         self.n_channels = n_channels
#         self.transmil_bb = transmil_bb
#         self.pretrained_bb = pretrained_bb
#
#         # Set up feature extractor
#         if transmil_bb == "resnet50":
#             resnet50 = models.resnet50(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in resnet50.parameters():
#                     param.requires_grad = False
#
#             modules = list(resnet50.children())[:-3]
#             self.feature_extractor = nn.Sequential(
#                 *modules,
#                 nn.AdaptiveAvgPool2d(1),
#                 View((-1, 1024)),
#                 nn.Linear(1024, self.L)
#             )
#
#         elif transmil_bb == "vgg16":
#             vgg16 = models.vgg16(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in vgg16.parameters():
#                     param.requires_grad = False
#
#             #self.feature_extractor = vgg16.features
#             self.feature_extractor = nn.Sequential(
#                 vgg16.features,
#                 nn.AdaptiveAvgPool2d(1))
#
#         elif transmil_bb == "vgg19":
#             vgg19 = models.vgg19(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in vgg19.parameters():
#                     param.requires_grad = False
#
#             # self.feature_extractor = vgg16.features
#             self.feature_extractor = nn.Sequential(
#                 vgg19.features,
#                 nn.AdaptiveAvgPool2d(1))
#
#         self.class_token = nn.Parameter(torch.zeros(1, 1, 512))
#
#         self.nystromer1 = Nystromformer(dim=self.L, depth=1)
#         self.nystromer2 = Nystromformer(dim=self.L, depth=1)
#         self.grp_conv1 = nn.Conv2d(512, 512, 3, padding=1)
#         self.grp_conv2 = nn.Conv2d(512, 512, 5, padding=2)
#         self.grp_conv3 = nn.Conv2d(512, 512, 7, padding=3)
#         self.ln = nn.LayerNorm(self.L)
#         if self.n_classes == 2:
#
#             self.fc = nn.Linear(self.L, 1)
#         else:
#             self.fc = nn.Linear(self.L, self.n_classes)
#
#     def forward(self, x):
#
#         #x = x.squeeze(0)
#         if self.n_channels == 1:
#             x = torch.cat((x, x, x), 1)
#
#         try:
#             Hf = self.feature_extractor(x)
#         except ValueError:
#             print("hola que tal")
#
#         Hf = Hf.squeeze(1)  # 512 features (Patch tokens: each patch's features)
#
#         ##################################################
#         # Squaring
#         ####################################################
#
#         N = np.square(self.sqrtN)
#         while Hf.shape[0] != N:
#             if Hf.shape[0] > N:  # some of our WSIs have more than N patches, but most have less
#                 Hf = Hf[:N, :]
#             else:
#                 d = N - Hf.shape[0]
#                 missing = Hf[:d, :]
#                 Hf = torch.cat((Hf, missing), 0)
#         Hf = Hf.squeeze().unsqueeze(0)
#         Hs = torch.cat((self.class_token, Hf), 1)
#
#         ####################################################
#         # MSA
#         ####################################################
#         Hs = self.nystromer1(Hs) + Hs
#         ####################################################
#         # PPEG
#         ####################################################
#         Hc, Hf = torch.split(Hs, [1, Hs.shape[1] - 1], 1) #Hc: Class token; Hf: Patch token
#         Hf = Hf.view(1, -1, self.sqrtN, self.sqrtN)
#
#         Hf1 = self.grp_conv1(Hf)
#         Hf2 = self.grp_conv2(Hf)
#         Hf3 = self.grp_conv3(Hf)
#         Hfs = Hf1 + Hf2 + Hf3
#
#         Hfs = Hfs.view(1, N, -1)
#         Hs = torch.cat((Hc, Hfs), 1)
#
#         ####################################################
#         # MSA 2
#         ####################################################
#         Hs = self.nystromer1(Hs) + Hs
#
#         ####################################################
#         # MLP
#         ####################################################
#         Hc, Hf = torch.split(Hs, [1, Hs.shape[1] - 1], 1)
#         Hc = Hc.squeeze() # features
#
#         ## TODO: Probar a meter lo del Admin aquí, a ver si va mejor
#
#         logits = self.fc(self.ln(Hc)) #logits (reduce dimensionanilty from 1024 to 512????)
#         Y_hat = torch.argmax(logits)
#         Y_prob = F.softmax(logits, dim=0)
#
#         #Y_hat = torch.ge(output, 1 / self.n_classes).float()
#
#         return Y_prob, Y_hat, logits
#
# class Lookahead(Optimizer):
#     r"""PyTorch implementation of the lookahead wrapper.
#     Lookahead Optimizer: https://arxiv.org/abs/1907.08610
#     """
#
#     def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
#         """optimizer: inner optimizer
#         la_steps (int): number of lookahead steps
#         la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
#         pullback_momentum (str): change to inner optimizer momentum on interpolation update
#         """
#         self.optimizer = optimizer
#         self._la_step = 0  # counter for inner optimizer
#         self.la_alpha = la_alpha
#         self._total_la_steps = la_steps
#         pullback_momentum = pullback_momentum.lower()
#         assert pullback_momentum in ["reset", "pullback", "none"]
#         self.pullback_momentum = pullback_momentum
#
#         self.state = defaultdict(dict)
#
#         # Cache the current optimizer parameters
#         for group in optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['cached_params'] = torch.zeros_like(p.data)
#                 param_state['cached_params'].copy_(p.data)
#                 if self.pullback_momentum == "pullback":
#                     param_state['cached_mom'] = torch.zeros_like(p.data)
#
#     def __getstate__(self):
#         return {
#             'state': self.state,
#             'optimizer': self.optimizer,
#             'la_alpha': self.la_alpha,
#             '_la_step': self._la_step,
#             '_total_la_steps': self._total_la_steps,
#             'pullback_momentum': self.pullback_momentum
#         }
#
#     def zero_grad(self):
#         self.optimizer.zero_grad()
#
#     def get_la_step(self):
#         return self._la_step
#
#     def state_dict(self):
#         return self.optimizer.state_dict()
#
#     def load_state_dict(self, state_dict):
#         self.optimizer.load_state_dict(state_dict)
#
#     def _backup_and_load_cache(self):
#         """Useful for performing evaluation on the slow weights (which typically generalize better)
#         """
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 param_state['backup_params'] = torch.zeros_like(p.data)
#                 param_state['backup_params'].copy_(p.data)
#                 p.data.copy_(param_state['cached_params'])
#
#     def _clear_and_load_backup(self):
#         for group in self.optimizer.param_groups:
#             for p in group['params']:
#                 param_state = self.state[p]
#                 p.data.copy_(param_state['backup_params'])
#                 del param_state['backup_params']
#
#     @property
#     def param_groups(self):
#         return self.optimizer.param_groups
#
#     def step(self, closure=None):
#         """Performs a single Lookahead optimization step.
#         Arguments:
#             closure (callable, optional): A closure that reevaluates the model
#                 and returns the loss.
#         """
#         loss = self.optimizer.step(closure)
#         self._la_step += 1
#
#         if self._la_step >= self._total_la_steps:
#             self._la_step = 0
#             # Lookahead and cache the current optimizer parameters
#             for group in self.optimizer.param_groups:
#                 for p in group['params']:
#                     param_state = self.state[p]
#                     p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
#                     param_state['cached_params'].copy_(p.data)
#                     if self.pullback_momentum == "pullback":
#                         internal_momentum = self.optimizer.state[p]["momentum_buffer"]
#                         self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
#                             1.0 - self.la_alpha, param_state["cached_mom"])
#                         param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
#                     elif self.pullback_momentum == "reset":
#                         self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
#
#         return loss
#
# class View(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape
#
#     def forward(self, input):
#         '''
#         Reshapes the input according to the shape saved in the view data structure.
#         '''
#         batch_size = input.size(0)
#         shape = (batch_size, *self.shape)
#         out = input.view(shape)
#         return out
#
# #########################
# # TransMIL_Pablo Models #
# #########################
#
# class TransMILArchitecture_Pablo(nn.Module):
#     def __init__(self, n_classes, n_channels=3, transmil_bb="resnet50", pretrained_bb=True, freeze_bb_weights=False):
#         super(TransMILArchitecture_Pablo, self).__init__()
#         self.pos_layer = PPEG(dim=512).cuda()
#         self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
#         self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
#         self.n_classes = n_classes
#         self.layer1 = TransLayer(dim=512).cuda()
#         self.layer2 = TransLayer(dim=512).cuda()
#         self.norm = nn.LayerNorm(512).cuda()
#         self._fc2 = torch.nn.Sequential(
#             torch.nn.Linear(512, self.n_classes).cuda()
#         )
#         self.n_classes = n_classes
#         self.n_channels = n_channels
#         self.transmil_bb = transmil_bb
#         self.pretrained_bb = pretrained_bb
#         self.freeze_bb_weights = freeze_bb_weights
#         self.L = 512
#
#         #self.backbone_extractor = AI4SKINClassifier(in_channels=3, n_classes=1, n_blocks=5).cuda()
#
#
#         # Set up feature extractor
#         if transmil_bb == "resnet50":
#             resnet50 = models.resnet50(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in resnet50.parameters():
#                     param.requires_grad = False
#
#             modules = list(resnet50.children())[:-3]
#             self.feature_extractor = nn.Sequential(
#                 *modules,
#                 nn.AdaptiveAvgPool2d(1),
#                 View((-1, 1024)),
#                 nn.Linear(1024, self.L)
#             )
#
#         elif transmil_bb == "vgg16":
#             vgg16 = models.vgg16(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in vgg16.parameters():
#                     param.requires_grad = False
#
#             #self.feature_extractor = vgg16.features
#             self.feature_extractor = nn.Sequential(
#                 vgg16.features,
#                 nn.AdaptiveAvgPool2d(1))
#
#         elif transmil_bb == "vgg19":
#             vgg19 = models.vgg19(pretrained=self.pretrained_bb)
#             # freeze weights from feature extractor
#             if freeze_bb_weights:
#                 for param in vgg19.parameters():
#                     param.requires_grad = False
#
#             # self.feature_extractor = vgg16.features
#             self.feature_extractor = nn.Sequential(
#                 vgg19.features,
#                 nn.AdaptiveAvgPool2d(1))
#
#     def forward(self, x):
#         # h = kwargs['data'].float()  # [B, n, 1024]
#
#         #h = self.backbone_extractor(x)
#         h = self.feature_extractor(x).squeeze()
#         # h = self._fc1(h)  # [B, n, 512]
#
#         # Keep 1 dim if 1 image
#         if len(h.shape)==1:
#             h = h.unsqueeze(dim=0)
#
#         # ---->pad
#         h = h.reshape(1, h.shape[0], -1)
#         H = h.shape[1]
#         _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
#         add_length = _H * _W - H
#         h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
#
#         # ---->cls_token
#         B = h.shape[0]
#         cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
#         h = torch.cat((cls_tokens, h), dim=1)
#
#         # ---->Translayer x1
#         h = self.layer1(h)  # [B, N, 512]
#
#         # ---->PPEG
#         h = self.pos_layer(h, _H, _W)  # [B, N, 512]
#
#         # ---->Translayer x2
#         h = self.layer2(h)  # [B, N, 512]
#
#         # ---->cls_token
#         h = self.norm(h)[:, 0]  # .squeeze(dim = 0)
#
#         # ---->predict
#         logits = self._fc2(h)  # [BS=1, n_classes]
#         # Y_hat = torch.argmax(logits, dim=1)
#         # Y_prob = F.softmax(logits, dim=1)
#         # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
#
#         #return logits, h
#
#
#         Y_hat = torch.argmax(logits)
#         Y_prob = F.softmax(logits, dim=1)
#
#         return Y_prob, Y_hat, logits
#
# class TransLayer(nn.Module):
#
#     def __init__(self, norm_layer=nn.LayerNorm, dim=512,
#                  dropout=0.1):  # voy a utilizar una VGG asi que la dimensión es 512
#         super().__init__()
#         self.norm = norm_layer(dim)
#         self.attn = NystromAttention(
#             dim=dim,
#             dim_head=dim // 8,
#             heads=8,
#             num_landmarks=dim // 2,  # number of landmarks
#             pinv_iterations=6,
#             # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
#             residual=True,
#             # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
#             dropout=dropout
#         ).cuda()
#
#     def forward(self, x):
#         x = x + self.attn(self.norm(x))
#         return x
#
# class PPEG(nn.Module):
#     def __init__(self, dim=512):
#         super(PPEG, self).__init__()
#         self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)  # estudiar porque se utilizan estas convoluciones
#         self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
#         self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
#
#     def forward(self, x, H, W):
#         B, _, C = x.shape
#         cls_token, feat_token = x[:, 0], x[:, 1:]
#         cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
#         x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
#         x = x.flatten(2).transpose(1, 2)
#         x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
#         return x
