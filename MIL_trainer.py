from MIL_utils import *
from MIL_models import *

class TransMIL_trainer():
    def __init__(self, dir_out, network, model_save_name, class_weights, mlflow_run_name, aggregation, lr=1*1e-4, alpha_ce=1, id='', early_stopping=False,
                 scheduler=False, virtual_batch_size=1, criterion='auc', optimizer_type="adam", optimizer_weight_decay=0):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.class_weights = class_weights
        self.model_save_name = model_save_name
        #self.mode = mode
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.macro_f1_train = []
        self.macro_f1_val = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []

        self.alpha_ce = alpha_ce
        self.best_criterion_auc = 0
        self.best_criterion_f1 = 0

        self.best_epoch = 0
        self.metrics = {}
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.mlflow_run_name=mlflow_run_name
        self.aggregation=aggregation


        self.params = list(self.network.parameters())
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "lookahead_adam":
            self.base_optim = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)
        elif self.optimizer_type == "lookahead_radam":
            self.base_optim = torch.optim.RAdam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)



    def train(self, train_generator, val_generator, test_generator, epochs, model_save_name, pred_column, pred_mode, loss_function):
        self.model_save_name = model_save_name
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.init_time = time.time()
        self.loss_function = loss_function

        # Move network to gpu
        self.network.cuda()
        # Network in train mode
        self.network.train()

        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch loss
            self.L_epoch = 0
            # Initialize for each epoch the GT and pred lists for training
            self.preds_train = []
            self.refs_train = []

            # Loop over training dataset
            print('[Training]: at bag level...')
            #start_time = time.time()
            for self.i_iteration, (X, Y, X_augm) in enumerate(tqdm(self.train_generator, leave=True, position=0)):
                #end_time = time.time()
                #seconds_elapsed = end_time - start_time
                #print("Iteration lasted in seconds: ", seconds_elapsed)

                if X_augm is None:
                    X_augm = torch.tensor(X).to('cuda')
                else:
                    X_augm = torch.tensor(X_augm).to('cuda')

                Y = torch.tensor(Y).to('cuda')

                # Forward network
                # if self.mode == 'embedding_GNN':
                #     Yprob, Yhat, logits, L_gnn = self.network(X_augm)
                # else:
                Yprob, Yhat, logits  = self.network(X_augm)

                # Store all model predictions
                self.preds_train.append(Yprob.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Categorical Cross Entropy
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')
                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights,
                                                       loss_function=self.loss_function)
                # Backward gradients
                #L = Lce * self.alpha_ce
                # if self.mode == 'embedding_GNN':
                #     L = Lce * self.alpha_ce + L_gnn
                # else:
                L = Lce * self.alpha_ce


                L = L / self.virtual_batch_size
                L.backward()

                # Optimizer
                # lookahead = Lookahead(optimizer=self.optimizer)
                # lookahead.step()
                # lookahead.zero_grad()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update Loss
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            self.on_epoch_end()

            if self.early_stopping: # if criterion does not improve for 20 epochs the training is stopped
                if self.i_epoch + 1 == (self.best_epoch + 30):
                    break

        # End MLFlow run
        mlflow.end_run()
        # On training end evaluate best model with Test Set
        self.on_train_end()

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc=0, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f}".format(
                i_epoch, epochs, iteration, total_iterations, Lce, macro_auc)

        # Print losses
        et = str(datetime.timedelta(seconds=time.time() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def on_train_end(self):
        ("-- Trained finished. Evaluating best val model on Tesºt Set. --")

        # weights_save_model_name_best_auc = self.model_save_name + '_network_weights_best_auc.pth'
        # weights_save_model_name_best_f1 = self.model_save_name + '_network_weights_best_f1.pth'
        weights_save_model_name_best_auc = 'network_weights_best_auc.pth'
        weights_save_model_name_best_f1 = 'network_weights_best_f1.pth'
        weights2eval_best_auc_path = os.path.join(self.dir_results, weights_save_model_name_best_auc)
        weights2eval_best_f1_path = os.path.join(self.dir_results, weights_save_model_name_best_f1)

        #torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

        try:

            eval_bag_level_classification(test_generator=self.test_generator,
                                          network=self.network,
                                          weights2eval_path=weights2eval_best_auc_path,
                                          pred_column=self.pred_column,
                                          pred_mode=self.pred_mode,
                                          aggregation=self.aggregation,
                                          results_save_path=self.dir_results,
                                          best_model_type="auc")

        except ValueError:
            print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))
        try:
            eval_bag_level_classification(test_generator=self.test_generator,
                                              network=self.network,
                                              weights2eval_path=weights2eval_best_f1_path,
                                              pred_column=self.pred_column,
                                              pred_mode=self.pred_mode,
                                              aggregation=self.aggregation,
                                              results_save_path=self.dir_results,
                                              best_model_type="f1")
        except ValueError:
            print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))


    def on_epoch_end(self):

        # Obtain epoch-level metrics
        try:
            macro_auc = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.squeeze(np.array(self.preds_train)),
                                      multi_class='ovr')
            weighted_f1_score = f1_score(np.argmax(np.squeeze(np.array(self.refs_train)), axis=1),
                                         np.argmax(np.squeeze(np.array(self.preds_train)), axis=1), average='weighted')

            self.macro_auc_lc_train.append(macro_auc)
            self.macro_f1_train.append(weighted_f1_score)

        except ValueError:
            macro_auc = 0.
            weighted_f1_score = 0.
            self.macro_auc_lc_train.append(macro_auc)
            self.macro_f1_train.append(weighted_f1_score)
            print("Only one class prediced (bad training).")

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc,
                            end_line='\n')
        # Update learning curves
        self.L_lc.append(self.L_epoch)
        # Obtain results on validation set
        try:
            Lce_val, macro_auc_val, f1_weighted_val = self.test_bag_level_classification(self.val_generator)
        except ValueError:
            print("Only one class prediced (bad training).")
            Lce_val = 0.
            macro_auc_val = 0.
            f1_weighted_val = 0.

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.macro_f1_val.append(f1_weighted_val)


        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4)}
        with open(os.path.join(self.dir_results, str(self.model_save_name) + '_metrics.json'), 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        # Save best AUC and best F1 score models

        if self.best_criterion_auc < self.macro_auc_lc_val[-1]:
            self.best_criterion_auc = self.macro_auc_lc_val[-1]
            if self.criterion == 'auc':
                self.best_epoch = (self.i_epoch + 1)

            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_auc.pth'
                weights_save_model_name =  'network_weights_best_auc.pth'
                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

                # After saving best model, test and report to MLFlow
                try:

                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath, test_ppv, test_npv, test_specificity, test_tpr, test_tnr, test_fpr, test_fnr = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  aggregation=self.aggregation,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="auc",
                                                  return_params=True,
                                                  show_cf=False)
                    mlflow.log_metric("test_BA_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BA_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BA_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BA_precision", test_precision, step=self.i_epoch)
                    mlflow.log_metric("test_BA_ppv", test_ppv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_npv", test_npv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_specificity", test_specificity, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tpr", test_tpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tnr", test_tnr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fpr", test_fpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fnr", test_fnr, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    #print(cf_savepath)
                    #print(os.getcwd())
                    #hola = "/home/clferma/Documents/MIL_global_labels/data/results/MolSub 5x Aggr MeanMax 3CLF CV LR New/PM_LUMINALSvsHER2vsTNBC_AGGR_mean_ML_5x_NN_bb_vgg16_FBB_False_PS_512_DA_non-spatial_SN_False_L_auc_E_100_LR_0002_Order_True_Optim_sgd_N_3_BDG_True_OWD_0_TP_O_0.4-T_1-S_1-I_1-N_1/CVFold_1/test_cfsn_matrix_best_auc.png"
                    mlflow.log_artifact(cf_savepath, "test_BA_cf")
                    #print(f"artifact_uri={mlflow.get_artifact_uri()}")
                    #print(cf_savepath)


                    #mlflow.log_artifact( local_path=cf_savepath)
                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))


        if self.best_criterion_f1 < self.macro_f1_val[-1]:
            self.best_criterion_f1 = self.macro_f1_val[-1]
            if self.criterion == 'f1':
                self.best_epoch = (self.i_epoch + 1)
            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_f1.pth'
                weights_save_model_name = 'network_weights_best_f1.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))
                # After saving best model, test and report to MLFlow

                try:
                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath,test_ppv, test_npv, test_specificity, test_tpr, test_tnr, test_fpr, test_fnr = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  aggregation=self.aggregation,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="f1",
                                                  return_params=True,
                                                  show_cf=False,
                                                  i_epoch = self.i_epoch)

                    mlflow.log_metric("test_BF1_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_precision", test_precision, step=self.i_epoch)
                    mlflow.log_metric("test_BA_ppv", test_ppv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_npv", test_npv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_specificity", test_specificity, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tpr", test_tpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tnr", test_tnr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fpr", test_fpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fnr", test_fnr, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BF1_cf")
                    #print(cf_savepath)

                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))

        print("Logging to MLFlow...")
        # #Log to MLFLow
        mlflow.log_metric("train_loss", float(np.round(self.L_epoch, 4)), step=self.i_epoch)
        mlflow.log_metric("train_auc", float(np.round(self.macro_auc_lc_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("train_f1", float(np.round(self.macro_f1_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("val_loss", float(np.round(Lce_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_auc", float(np.round(macro_auc_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_f1", float(np.round(f1_weighted_val, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_auc", float(np.round(self.best_criterion_auc, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_f1", float(np.round(self.best_criterion_f1, 4)), step=self.i_epoch)

    def test_bag_level_classification(self, generator, binary=False):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0

        with torch.no_grad():
            for self.i_iteration, (X, Y, _) in enumerate(tqdm(generator, leave=True, position=0)):
                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # if self.i_iteration==216:
                #     print("Holu")
                #     print("Ole")

                # Set model to training mode and clear gradients

                # Forward network
                # if self.mode == 'embedding_GNN':
                #     Yprob, Yhat, logits, L_gnn = self.network(X)
                # else:
                Yprob, Yhat, logits = self.network(X)

                Yhat_all.append(Yprob.detach().cpu().numpy())
                # Estimate losses
                # Lce = self.L(Yhat, torch.squeeze(Y))
                #Lce = self.L(torch.squeeze(Yhat), torch.squeeze(Y))
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')

                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights)

                Lce_e += Lce.cpu().detach().numpy() / len(generator)

                Y_all.append(Y.detach().cpu().numpy())

                # Save predictions
                softmax_layer = torch.nn.Softmax(dim=0)

                #Yhat_all.append(softmax_layer(logits).detach().cpu().numpy())
                #Yhat_all.append(Yhat.detach().cpu().numpy())

        # Display losses per iteration
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce.cpu().detach().numpy(),
                            end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if binary:
            Yhat_all = np.max(Yhat_all, 1)
            # Y_all = np.max(Y_all, 1)

        # Compute overall metrics
        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all.squeeze(), axis=1)
        macro_cf = confusion_matrix(y_gt, y_pred)
        f1_weighted = sklearn.metrics.f1_score(y_gt, y_pred, average='weighted')

        macro_auc = roc_auc_score(Y_all, Yhat_all.squeeze(), multi_class='ovr')

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce_e, macro_auc,
                            end_line='\n')
        print(macro_cf)

        return Lce_e, macro_auc, f1_weighted

class Patch_GCN_online_trainer():
    def __init__(self, dir_out, network, model_save_name, class_weights, mlflow_run_name, aggregation,  lr=1*1e-4, alpha_ce=1, id='', early_stopping=False,
                 scheduler=False, virtual_batch_size=1, criterion='auc', optimizer_type="adam", optimizer_weight_decay=0, knn=8, node_feature_extractor="resnet50_3blocks_1024"):
        super(Patch_GCN_online_trainer, self).__init__()
        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.class_weights = class_weights
        self.model_save_name = model_save_name
        self.aggregation = aggregation
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.macro_f1_train = []
        self.macro_f1_val = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []

        self.alpha_ce = alpha_ce
        self.best_criterion_auc = 0
        self.best_criterion_f1 = 0

        self.best_epoch = 0
        self.metrics = {}
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.mlflow_run_name=mlflow_run_name

        #Patch-GCN
        self.knn = knn
        self.patch_gcn_node_feature_extractor = node_feature_extractor


        self.params = list(self.network.parameters())
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "lookahead_adam":
            self.base_optim = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)
        elif self.optimizer_type == "lookahead_radam":
            self.base_optim = torch.optim.RAdam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)



    def train(self, train_generator, val_generator, test_generator, epochs, model_save_name, pred_column, pred_mode, loss_function):
        self.model_save_name = model_save_name
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.init_time = time.time()
        self.loss_function = loss_function

        # Move network to gpu
        self.network.cuda()
        # Network in train mode
        self.network.train()

        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch loss
            self.L_epoch = 0

            # Initialize for each epoch the GT and pred lists for training
            self.preds_train = []
            self.refs_train = []

            # Loop over training dataset
            print('[Training]: at bag level...')
            #start_time = time.time()
            for self.i_iteration, (X, Y, X_augm, img_coords) in enumerate(tqdm(self.train_generator, leave=True, position=0)):

                #end_time = time.time()
                #seconds_elapsed = end_time - start_time
                #print("Iteration lasted in seconds: ", seconds_elapsed)

                if X_augm is None:
                    X_augm = torch.tensor(X).to('cuda')
                else:
                    X_augm = torch.tensor(X_augm).to('cuda')

                Y = torch.tensor(Y).to('cuda')


                # Forward network
                if self.aggregation == "GNN" or self.aggregation=="GNN_basic" or self.aggregation=="GNN_coords":
                    Yprob, Yhat, logits, L_gnn = self.network(X_augm, img_coords)
                elif self.aggregation == "Patch_GCN":
                    # Convert bag to Graph
                    graph_creator = imgs2graph(backbone=self.patch_gcn_node_feature_extractor, pretrained=True, knn=self.knn).cuda()
                    graph_from_bag = graph_creator(X_augm, img_coords).to('cuda')
                    Yprob, Yhat, logits  = self.network(graph=graph_from_bag)
                else:
                    Yprob, Yhat, logits  = self.network(X_augm)


                # Store all model predictions
                self.preds_train.append(Yprob.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Categorical Cross Entropy
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')
                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights,
                                                       loss_function=self.loss_function)

                # Update overall losses
                if self.aggregation == "GNN" or self.aggregation=="GNN_basic" or self.aggregation=="GNN_coords":
                    L = Lce * self.alpha_ce + L_gnn
                else:
                    L = Lce * self.alpha_ce

                #L = Lce * self.alpha_ce

                # Backward gradients
                L = L / self.virtual_batch_size
                L.backward()

                # Optimizer
                # lookahead = Lookahead(optimizer=self.optimizer)
                # lookahead.step()
                # lookahead.zero_grad()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update Loss
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            self.on_epoch_end()

            if self.early_stopping: # if criterion does not improve for 20 epochs the training is stopped
                if self.i_epoch + 1 == (self.best_epoch + 30):
                    break

        # End MLFlow run
        mlflow.end_run()
        # On training end evaluate best model with Test Set
        self.on_train_end()

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc=0, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f}".format(
                i_epoch, epochs, iteration, total_iterations, Lce, macro_auc)

        # Print losses
        et = str(datetime.timedelta(seconds=time.time() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def on_train_end(self):
        ("-- Trained finished. Evaluating best val model on Tesºt Set. --")

        # weights_save_model_name_best_auc = self.model_save_name + '_network_weights_best_auc.pth'
        # weights_save_model_name_best_f1 = self.model_save_name + '_network_weights_best_f1.pth'
        weights_save_model_name_best_auc = 'network_weights_best_auc.pth'
        weights_save_model_name_best_f1 = 'network_weights_best_f1.pth'
        weights2eval_best_auc_path = os.path.join(self.dir_results, weights_save_model_name_best_auc)
        weights2eval_best_f1_path = os.path.join(self.dir_results, weights_save_model_name_best_f1)

        #torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

        # try:

        eval_bag_level_classification(test_generator=self.test_generator,
                                      network=self.network,
                                      weights2eval_path=weights2eval_best_auc_path,
                                      pred_column=self.pred_column,
                                      pred_mode=self.pred_mode,
                                      aggregation=self.aggregation,
                                      results_save_path=self.dir_results,
                                      best_model_type="auc",
                                      node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                      knn=self.knn)

        # except ValueError:
        #     print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))
        # try:
        eval_bag_level_classification(test_generator=self.test_generator,
                                          network=self.network,
                                          weights2eval_path=weights2eval_best_f1_path,
                                          pred_column=self.pred_column,
                                          pred_mode=self.pred_mode,
                                          aggregation=self.aggregation,
                                          results_save_path=self.dir_results,
                                          best_model_type="f1",
                                          node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                          knn=self.knn)

        # except ValueError:
        #     print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))


    def on_epoch_end(self):

        # Obtain epoch-level metrics
        # try:
        macro_auc = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.squeeze(np.array(self.preds_train)), multi_class='ovr')
        weighted_f1_score = f1_score(np.argmax(np.squeeze(np.array(self.refs_train)), axis=1), np.argmax(np.squeeze(np.array(self.preds_train)), axis=1), average='weighted')

        self.macro_auc_lc_train.append(macro_auc)
        self.macro_f1_train.append(weighted_f1_score)

        # except ValueError:
        #     macro_auc = 0.
        #     self.macro_auc_lc_train.append(macro_auc)
        #     print("Only one class prediced (bad training).")

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc,
                            end_line='\n')
        # Update learning curves
        self.L_lc.append(self.L_epoch)
        # Obtain results on validation set
        # try:
        Lce_val, macro_auc_val, f1_weighted_val = self.test_bag_level_classification(self.val_generator)
        # except ValueError:
        #     print("Only one class prediced (bad training).")
        #     Lce_val = 0.
        #     macro_auc_val = 0.
        #     f1_weighted_val = 0.

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.macro_f1_val.append(f1_weighted_val)


        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4)}
        with open(os.path.join(self.dir_results, str(self.model_save_name) + '_metrics.json'), 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        # Save best AUC and best F1 score models

        if self.best_criterion_auc < self.macro_auc_lc_val[-1]:
            self.best_criterion_auc = self.macro_auc_lc_val[-1]
            if self.criterion == 'auc':
                self.best_epoch = (self.i_epoch + 1)

            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_auc.pth'
                weights_save_model_name = 'network_weights_best_auc.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

                # After saving best model, test and report to MLFlow
                try:

                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath, test_ppv, test_npv, test_specificity, test_tpr, test_tnr, test_fpr, test_fnr = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network, aggregation=self.aggregation,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="auc",
                                                  return_params=True,
                                                  show_cf=False,
                                                  node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                                  knn=self.knn)

                    mlflow.log_metric("test_BA_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BA_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BA_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BA_precision", test_precision, step=self.i_epoch)
                    mlflow.log_metric("test_BA_ppv", test_ppv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_npv", test_npv, step=self.i_epoch)
                    mlflow.log_metric("test_BA_specificity", test_specificity, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tpr", test_tpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_tnr", test_tnr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fpr", test_fpr, step=self.i_epoch)
                    mlflow.log_metric("test_BA_fnr", test_fnr, step=self.i_epoch)

                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BA_cf")

                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))


        if self.best_criterion_f1 < self.macro_f1_val[-1]:
            self.best_criterion_f1 = self.macro_f1_val[-1]
            if self.criterion == 'f1':
                self.best_epoch = (self.i_epoch + 1)
            if (self.i_epoch + 1) > 0:
                # weights_save_model_name = self.model_save_name + '_network_weights_best_f1.pth'
                weights_save_model_name = 'network_weights_best_f1.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))
                # After saving best model, test and report to MLFlow
                try:
                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath, test_ppv, test_npv, test_specificity, test_tpr, test_tnr, test_fpr, test_fnr = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network, aggregation=self.aggregation,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="f1",
                                                  return_params=True,
                                                  show_cf=False,
                                                  node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                                  knn=self.knn)

                    mlflow.log_metric("test_BF1_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_precision", test_precision, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_ppv", test_ppv, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_npv", test_npv, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_specificity", test_specificity, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_tpr", test_tpr, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_tnr", test_tnr, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_fpr", test_fpr, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_fnr", test_fnr, step=self.i_epoch)

                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BF1_cf")
                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))

        print("Logging to MLFlow...")
        # #Log to MLFLow
        mlflow.log_metric("train_loss", float(np.round(self.L_epoch, 4)), step=self.i_epoch)
        mlflow.log_metric("train_auc", float(np.round(self.macro_auc_lc_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("train_f1", float(np.round(self.macro_f1_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("val_loss", float(np.round(Lce_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_auc", float(np.round(macro_auc_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_f1", float(np.round(f1_weighted_val, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_auc", float(np.round(self.best_criterion_auc, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_f1", float(np.round(self.best_criterion_f1, 4)), step=self.i_epoch)

    def test_bag_level_classification(self, generator, binary=False):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (X, Y, _, img_coords) in enumerate(tqdm(generator, leave=True, position=0)):
            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()

            # if self.i_iteration==216:
            #     print("Holu")
            #     print("Ole")

            # Set model to training mode and clear gradients
            # Convert bag to Graph
            graph_creator = imgs2graph(backbone=self.patch_gcn_node_feature_extractor, pretrained=True, knn=self.knn).cuda()
            graph_from_bag = graph_creator(X, img_coords).to('cuda')

            # Forward network
            if self.aggregation == "GNN" or self.aggregation=="GNN_basic" or self.aggregation=="GNN_coords":
                Yprob, Yhat, logits, L_gnn = self.network(X, img_coords)
            elif self.aggregation == "Patch_GCN":
                Yprob, Yhat, logits  = self.network(graph=graph_from_bag)
            else:
                Yprob, Yhat, logits = self.network(X)


            #Yprob, Yhat, logits = self.network(X)
            Yhat_all.append(Yprob.detach().cpu().numpy())
            # Estimate losses
            # Lce = self.L(Yhat, torch.squeeze(Y))
            #Lce = self.L(torch.squeeze(Yhat), torch.squeeze(Y))
            Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')

            Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                   y_true=Ycat,
                                                   class_weights=self.class_weights)

            Lce_e += Lce.cpu().detach().numpy() / len(generator)

            Y_all.append(Y.detach().cpu().numpy())

            # Save predictions
            softmax_layer = torch.nn.Softmax(dim=0)

            #Yhat_all.append(softmax_layer(logits).detach().cpu().numpy())
            #Yhat_all.append(Yhat.detach().cpu().numpy())

        # Display losses per iteration
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce.cpu().detach().numpy(),
                            end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if binary:
            Yhat_all = np.max(Yhat_all, 1)
            # Y_all = np.max(Y_all, 1)

        # Compute overall metrics
        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all.squeeze(), axis=1)
        macro_cf = confusion_matrix(y_gt, y_pred)
        f1_weighted = sklearn.metrics.f1_score(y_gt, y_pred, average='weighted')

        macro_auc = roc_auc_score(Y_all, Yhat_all.squeeze(), multi_class='ovr')

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce_e, macro_auc,
                            end_line='\n')
        print(macro_cf)

        return Lce_e, macro_auc, f1_weighted

class Patch_GCN_offline_trainer():
    def __init__(self, dir_out, network, model_save_name, class_weights, mlflow_run_name, aggregation,  lr=1*1e-4, alpha_ce=1, id='', early_stopping=False,
                 scheduler=False, virtual_batch_size=1, criterion='auc', optimizer_type="adam", optimizer_weight_decay=0, knn=8, node_feature_extractor="resnet50_3blocks_1024"):
        super(Patch_GCN_offline_trainer, self).__init__()
        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.class_weights = class_weights
        self.model_save_name = model_save_name
        self.aggregation = aggregation
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.macro_f1_train = []
        self.macro_f1_val = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []

        self.alpha_ce = alpha_ce
        self.best_criterion_auc = 0
        self.best_criterion_f1 = 0

        self.best_epoch = 0
        self.metrics = {}
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.mlflow_run_name=mlflow_run_name

        #Patch-GCN
        self.knn = knn
        self.patch_gcn_node_feature_extractor = node_feature_extractor


        self.params = list(self.network.parameters())
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "lookahead_adam":
            self.base_optim = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)
        elif self.optimizer_type == "lookahead_radam":
            self.base_optim = torch.optim.RAdam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)



    def train(self, train_generator, val_generator, test_generator, epochs, model_save_name, pred_column, pred_mode, loss_function):
        self.model_save_name = model_save_name
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.init_time = time.time()
        self.loss_function = loss_function

        # Move network to gpu
        self.network.cuda()
        # Network in train mode
        self.network.train()

        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch loss
            self.L_epoch = 0

            # Initialize for each epoch the GT and pred lists for training
            self.preds_train = []
            self.refs_train = []

            # Loop over training dataset
            print('[Training]: at bag level...')
            #start_time = time.time()
            for self.i_iteration, (graph_from_bag, Y) in enumerate(tqdm(self.train_generator, leave=True, position=0)):

                # Graph to tensor
                graph_from_bag = graph_from_bag.to('cuda')
                # Label to tensor
                Y = torch.tensor(Y).to('cuda')

                # Forward network
                Yprob, Yhat, logits  = self.network(graph=graph_from_bag)

                # Store all model predictions
                self.preds_train.append(Yprob.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Categorical Cross Entropy
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')
                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights,
                                                       loss_function=self.loss_function)

                # Update overall losses
                if self.aggregation == "GNN" or self.aggregation=="GNN_basic" or self.aggregation=="GNN_coords":
                    L = Lce * self.alpha_ce + L_gnn
                else:
                    L = Lce * self.alpha_ce

                #L = Lce * self.alpha_ce

                # Backward gradients
                L = L / self.virtual_batch_size
                L.backward()

                # Optimizer
                # lookahead = Lookahead(optimizer=self.optimizer)
                # lookahead.step()
                # lookahead.zero_grad()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update Loss
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            self.on_epoch_end()

            if self.early_stopping: # if criterion does not improve for 20 epochs the training is stopped
                if self.i_epoch + 1 == (self.best_epoch + 30):
                    break

        # End MLFlow run
        mlflow.end_run()
        # On training end evaluate best model with Test Set
        self.on_train_end()

    def on_epoch_end(self):

        # Obtain epoch-level metrics
        macro_auc = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.squeeze(np.array(self.preds_train)), multi_class='ovr')
        weighted_f1_score = f1_score(np.argmax(np.squeeze(np.array(self.refs_train)), axis=1), np.argmax(np.squeeze(np.array(self.preds_train)), axis=1), average='weighted')

        self.macro_auc_lc_train.append(macro_auc)
        self.macro_f1_train.append(weighted_f1_score)

        # except ValueError:
        #     macro_auc = 0.
        #     self.macro_auc_lc_train.append(macro_auc)
        #     print("Only one class prediced (bad training).")

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc,
                            end_line='\n')
        # Update learning curves
        self.L_lc.append(self.L_epoch)
        # Obtain results on validation set
        # try:
        Lce_val, macro_auc_val, f1_weighted_val = self.test_bag_level_classification(self.val_generator)
        # except ValueError:
        #     print("Only one class prediced (bad training).")
        #     Lce_val = 0.
        #     macro_auc_val = 0.
        #     f1_weighted_val = 0.

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.macro_f1_val.append(f1_weighted_val)


        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4)}
        #with open(os.path.join(self.dir_results, str(self.model_save_name) + '_metrics.json'), 'w') as fp:
        with open(os.path.join(self.dir_results, 'metrics.json'), 'w') as fp:

            json.dump(metrics, fp)
        print(metrics)

        # Save best AUC and best F1 score models

        if self.best_criterion_auc < self.macro_auc_lc_val[-1]:
            self.best_criterion_auc = self.macro_auc_lc_val[-1]
            if self.criterion == 'auc':
                self.best_epoch = (self.i_epoch + 1)

            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_auc.pth'
                weights_save_model_name = 'network_weights_best_auc.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

                # After saving best model, test and report to MLFlow
                try:

                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath = eval_bag_level_classification_offline_graphs(test_generator=self.test_generator,
                                                  network=self.network, aggregation=self.aggregation,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="auc",
                                                  return_params=True,
                                                  show_cf=False,
                                                  node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                                  knn=self.knn)

                    mlflow.log_metric("test_BA_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BA_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BA_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BA_precision", test_precision, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BA_cf")

                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))

        if self.best_criterion_f1 < self.macro_f1_val[-1]:
            self.best_criterion_f1 = self.macro_f1_val[-1]
            if self.criterion == 'f1':
                self.best_epoch = (self.i_epoch + 1)

            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_f1.pth'
                weights_save_model_name = 'network_weights_best_f1.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))
                # After saving best model, test and report to MLFlow
                try:
                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath = eval_bag_level_classification_offline_graphs(test_generator=self.test_generator,
                                                  network=self.network, aggregation=self.aggregation,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="f1",
                                                  return_params=True,
                                                  show_cf=False,
                                                  node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                                  knn=self.knn)

                    mlflow.log_metric("test_BF1_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_precision", test_precision, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BF1_cf")
                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))

        print("Logging to MLFlow...")
        # #Log to MLFLow
        mlflow.log_metric("train_loss", float(np.round(self.L_epoch, 4)), step=self.i_epoch)
        mlflow.log_metric("train_auc", float(np.round(self.macro_auc_lc_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("train_f1", float(np.round(self.macro_f1_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("val_loss", float(np.round(Lce_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_auc", float(np.round(macro_auc_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_f1", float(np.round(f1_weighted_val, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_auc", float(np.round(self.best_criterion_auc, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_f1", float(np.round(self.best_criterion_f1, 4)), step=self.i_epoch)


    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc=0, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f}".format(
                i_epoch, epochs, iteration, total_iterations, Lce, macro_auc)

        # Print losses
        et = str(datetime.timedelta(seconds=time.time() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def on_train_end(self):
        ("-- Trained finished. Evaluating best val model on Tesºt Set. --")

        # weights_save_model_name_best_auc = self.model_save_name + '_network_weights_best_auc.pth'
        # weights_save_model_name_best_f1 = self.model_save_name + '_network_weights_best_f1.pth'
        weights_save_model_name_best_auc = 'network_weights_best_auc.pth'
        weights_save_model_name_best_f1 = 'network_weights_best_f1.pth'
        weights2eval_best_auc_path = os.path.join(self.dir_results, weights_save_model_name_best_auc)
        weights2eval_best_f1_path = os.path.join(self.dir_results, weights_save_model_name_best_f1)

        #torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

        # try:

        eval_bag_level_classification_offline_graphs(test_generator=self.test_generator,
                                      network=self.network,
                                      weights2eval_path=weights2eval_best_auc_path,
                                      pred_column=self.pred_column,
                                      pred_mode=self.pred_mode,
                                      aggregation=self.aggregation,
                                      results_save_path=self.dir_results,
                                      best_model_type="auc",
                                      node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                      knn=self.knn)

        # except ValueError:
        #     print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))
        # try:
        eval_bag_level_classification_offline_graphs(test_generator=self.test_generator,
                                          network=self.network,
                                          weights2eval_path=weights2eval_best_f1_path,
                                          pred_column=self.pred_column,
                                          pred_mode=self.pred_mode,
                                          aggregation=self.aggregation,
                                          results_save_path=self.dir_results,
                                          best_model_type="f1",
                                          node_feature_extractor=self.patch_gcn_node_feature_extractor,
                                          knn=self.knn)

        # except ValueError:
        #     print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))


    def test_bag_level_classification(self, generator, binary=False):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0
        for self.i_iteration, (graph_from_bag, Y) in enumerate(tqdm(generator, leave=True, position=0)):
            # Graph to tensor
            graph_from_bag = graph_from_bag.to('cuda')
            # Label to tensor
            Y = torch.tensor(Y).to('cuda')

            # X = torch.tensor(X).cuda().float()
            # Y = torch.tensor(Y).cuda().float()

            # if self.i_iteration==216:
            #     print("Holu")
            #     print("Ole")

            # Set model to training mode and clear gradients
            # Convert bag to Graph
            # graph_creator = imgs2graph(backbone=self.patch_gcn_node_feature_extractor, pretrained=True, knn=self.knn).cuda()
            # graph_from_bag = graph_creator(X, img_coords).to('cuda')

            # Forward network
            if self.aggregation == "GNN" or self.aggregation=="GNN_basic" or self.aggregation=="GNN_coords":
                Yprob, Yhat, logits, L_gnn = self.network(X, img_coords)
            elif self.aggregation == "Patch_GCN" or self.aggregation == "Patch_GCN_offline":
                Yprob, Yhat, logits  = self.network(graph=graph_from_bag)
            else:
                Yprob, Yhat, logits = self.network(X)


            #Yprob, Yhat, logits = self.network(X)
            Yhat_all.append(Yprob.detach().cpu().numpy())
            # Estimate losses
            # Lce = self.L(Yhat, torch.squeeze(Y))
            #Lce = self.L(torch.squeeze(Yhat), torch.squeeze(Y))
            Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')

            Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                   y_true=Ycat,
                                                   class_weights=self.class_weights)

            Lce_e += Lce.cpu().detach().numpy() / len(generator)

            Y_all.append(Y.detach().cpu().numpy())

            # Save predictions
            softmax_layer = torch.nn.Softmax(dim=0)

            #Yhat_all.append(softmax_layer(logits).detach().cpu().numpy())
            #Yhat_all.append(Yhat.detach().cpu().numpy())

        # Display losses per iteration
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce.cpu().detach().numpy(),
                            end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if binary:
            Yhat_all = np.max(Yhat_all, 1)
            # Y_all = np.max(Y_all, 1)

        # Compute overall metrics
        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all.squeeze(), axis=1)
        macro_cf = confusion_matrix(y_gt, y_pred)
        f1_weighted = sklearn.metrics.f1_score(y_gt, y_pred, average='weighted')

        macro_auc = roc_auc_score(Y_all, Yhat_all.squeeze(), multi_class='ovr')

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce_e, macro_auc,
                            end_line='\n')
        print(macro_cf)

        return Lce_e, macro_auc, f1_weighted

class TransMIL_trainer_w_features():
    def __init__(self, dir_out, network, model_save_name, class_weights, mlflow_run_name, aggregation, lr=1*1e-4, alpha_ce=1, id='', early_stopping=False,
                 scheduler=False, virtual_batch_size=1, criterion='auc', optimizer_type="adam", optimizer_weight_decay=0):

        self.dir_results = dir_out
        if not os.path.isdir(self.dir_results):
            os.mkdir(self.dir_results)

        # Other
        self.class_weights = class_weights
        self.model_save_name = model_save_name
        #self.mode = mode
        self.best_auc = 0.
        self.init_time = 0
        self.lr = lr
        self.L_epoch = 0
        self.L_lc = []
        self.Lce_lc_val = []
        self.macro_auc_lc_val = []
        self.macro_auc_lc_train = []
        self.macro_f1_train = []
        self.macro_f1_val = []
        self.i_epoch = 0
        self.epochs = 0
        self.i_iteration = 0
        self.iterations = 0
        self.network = network
        self.train_generator = []
        self.preds_train = []
        self.refs_train = []

        self.alpha_ce = alpha_ce
        self.best_criterion_auc = 0
        self.best_criterion_f1 = 0

        self.best_epoch = 0
        self.metrics = {}
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.virtual_batch_size = virtual_batch_size
        self.criterion = criterion
        self.optimizer_type = optimizer_type
        self.optimizer_weight_decay = optimizer_weight_decay
        self.mlflow_run_name=mlflow_run_name
        self.aggregation=aggregation


        self.params = list(self.network.parameters())
        #self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        if self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
        elif self.optimizer_type == "lookahead_adam":
            self.base_optim = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)
        elif self.optimizer_type == "lookahead_radam":
            self.base_optim = torch.optim.RAdam(self.params, lr=self.lr, weight_decay=self.optimizer_weight_decay)
            self.optimizer = Lookahead(self.base_optim)



    def train(self, train_generator, val_generator, test_generator, epochs, model_save_name, pred_column, pred_mode, loss_function):
        self.model_save_name = model_save_name
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.epochs = epochs
        self.iterations = len(train_generator)
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.init_time = time.time()
        self.loss_function = loss_function

        # Move network to gpu
        self.network.cuda()
        # Network in train mode
        self.network.train()

        for i_epoch in range(epochs):
            self.i_epoch = i_epoch
            # init epoch loss
            self.L_epoch = 0
            # Initialize for each epoch the GT and pred lists for training
            self.preds_train = []
            self.refs_train = []

            # Loop over training dataset
            print('[Training]: at bag level...')
            #start_time = time.time()
            for self.i_iteration, (image_features, Y) in enumerate(tqdm(self.train_generator, leave=True, position=0)):
                #end_time = time.time()
                #seconds_elapsed = end_time - start_time
                #print("Iteration lasted in seconds: ", seconds_elapsed)


                image_features = torch.tensor(image_features).to('cuda')
                Y = torch.tensor(Y).to('cuda')

                # Forward network
                # if self.mode == 'embedding_GNN':
                #     Yprob, Yhat, logits, L_gnn = self.network(X_augm)
                # else:
                Yprob, Yhat, logits  = self.network(image_features)

                # Store all model predictions
                self.preds_train.append(Yprob.detach().cpu().numpy())
                self.refs_train.append(Y.detach().cpu().numpy())

                # Categorical Cross Entropy
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')
                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights,
                                                       loss_function=self.loss_function)
                # Backward gradients
                #L = Lce * self.alpha_ce
                # if self.mode == 'embedding_GNN':
                #     L = Lce * self.alpha_ce + L_gnn
                # else:
                L = Lce * self.alpha_ce


                L = L / self.virtual_batch_size
                L.backward()

                # Optimizer
                # lookahead = Lookahead(optimizer=self.optimizer)
                # lookahead.step()
                # lookahead.zero_grad()

                # Update weights and clear gradients
                if ((self.i_epoch + 1) % self.virtual_batch_size) == 0:
                    #torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update Loss
                self.L_epoch += Lce.cpu().detach().numpy() / len(self.train_generator)

            self.on_epoch_end()

            if self.early_stopping: # if criterion does not improve for 20 epochs the training is stopped
                if self.i_epoch + 1 == (self.best_epoch + 30):
                    break

        # End MLFlow run
        mlflow.end_run()
        # On training end evaluate best model with Test Set
        self.on_train_end()

    def display_losses(self, i_epoch, epochs, iteration, total_iterations, Lce, macro_auc=0, end_line=''):

        info = "[INFO] Epoch {}/{}  -- Step {}/{}: Lce={:.4f} ; AUC={:.4f}".format(
                i_epoch, epochs, iteration, total_iterations, Lce, macro_auc)

        # Print losses
        et = str(datetime.timedelta(seconds=time.time() - self.init_time))
        print(info + ',ET=' + et, end=end_line)

    def on_train_end(self):
        ("-- Trained finished. Evaluating best val model on Tesºt Set. --")

        # weights_save_model_name_best_auc = self.model_save_name + '_network_weights_best_auc.pth'
        # weights_save_model_name_best_f1 = self.model_save_name + '_network_weights_best_f1.pth'
        weights_save_model_name_best_auc = 'network_weights_best_auc.pth'
        weights_save_model_name_best_f1 = 'network_weights_best_f1.pth'
        weights2eval_best_auc_path = os.path.join(self.dir_results, weights_save_model_name_best_auc)
        weights2eval_best_f1_path = os.path.join(self.dir_results, weights_save_model_name_best_f1)

        #torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

        try:

            eval_bag_level_classification(test_generator=self.test_generator,
                                          network=self.network,
                                          weights2eval_path=weights2eval_best_auc_path,
                                          pred_column=self.pred_column,
                                          pred_mode=self.pred_mode,
                                          aggregation=self.aggregation,
                                          results_save_path=self.dir_results,
                                          best_model_type="auc")

        except ValueError:
            print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))
        try:
            eval_bag_level_classification(test_generator=self.test_generator,
                                              network=self.network,
                                              weights2eval_path=weights2eval_best_f1_path,
                                              pred_column=self.pred_column,
                                              pred_mode=self.pred_mode,
                                              aggregation=self.aggregation,
                                              results_save_path=self.dir_results,
                                              best_model_type="f1")
        except ValueError:
            print("Only one class prediced (bad training) for " , str(weights2eval_best_auc_path))


    def on_epoch_end(self):

        # Obtain epoch-level metrics
        try:
            macro_auc = roc_auc_score(np.squeeze(np.array(self.refs_train)), np.squeeze(np.array(self.preds_train)),
                                      multi_class='ovr')
            weighted_f1_score = f1_score(np.argmax(np.squeeze(np.array(self.refs_train)), axis=1),
                                         np.argmax(np.squeeze(np.array(self.preds_train)), axis=1), average='weighted')

            self.macro_auc_lc_train.append(macro_auc)
            self.macro_f1_train.append(weighted_f1_score)

        except ValueError:
            macro_auc = 0.
            self.macro_auc_lc_train.append(macro_auc)
            print("Only one class prediced (bad training).")

        # Display losses
        self.display_losses(self.i_epoch + 1, self.epochs, self.iterations, self.iterations, self.L_epoch, macro_auc,
                            end_line='\n')
        # Update learning curves
        self.L_lc.append(self.L_epoch)
        # Obtain results on validation set
        try:
            Lce_val, macro_auc_val, f1_weighted_val = self.test_bag_level_classification(self.val_generator)
        except ValueError:
            print("Only one class prediced (bad training).")
            Lce_val = 0.
            macro_auc_val = 0.
            f1_weighted_val = 0.

        # Save loss value into learning curve
        self.Lce_lc_val.append(Lce_val)
        self.macro_auc_lc_val.append(macro_auc_val)
        self.macro_f1_val.append(f1_weighted_val)


        metrics = {'epoch': self.i_epoch + 1, 'AUCtrain': np.round(self.macro_auc_lc_train[-1], 4),
                   'AUCval': np.round(self.macro_auc_lc_val[-1], 4)}
        with open(os.path.join(self.dir_results, str(self.model_save_name) + '_metrics.json'), 'w') as fp:
            json.dump(metrics, fp)
        print(metrics)

        # Save best AUC and best F1 score models

        if self.best_criterion_auc < self.macro_auc_lc_val[-1]:
            self.best_criterion_auc = self.macro_auc_lc_val[-1]
            if self.criterion == 'auc':
                self.best_epoch = (self.i_epoch + 1)

            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_auc.pth'
                weights_save_model_name =  'network_weights_best_auc.pth'
                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))

                # After saving best model, test and report to MLFlow
                try:

                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  aggregation=self.aggregation,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="auc",
                                                  return_params=True,
                                                  show_cf=False)
                    mlflow.log_metric("test_BA_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BA_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BA_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BA_precision", test_precision, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    #print(cf_savepath)
                    #print(os.getcwd())
                    #hola = "/home/clferma/Documents/MIL_global_labels/data/results/MolSub 5x Aggr MeanMax 3CLF CV LR New/PM_LUMINALSvsHER2vsTNBC_AGGR_mean_ML_5x_NN_bb_vgg16_FBB_False_PS_512_DA_non-spatial_SN_False_L_auc_E_100_LR_0002_Order_True_Optim_sgd_N_3_BDG_True_OWD_0_TP_O_0.4-T_1-S_1-I_1-N_1/CVFold_1/test_cfsn_matrix_best_auc.png"
                    mlflow.log_artifact(cf_savepath, "test_BA_cf")
                    #print(f"artifact_uri={mlflow.get_artifact_uri()}")
                    #print(cf_savepath)


                    #mlflow.log_artifact( local_path=cf_savepath)
                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))


        if self.best_criterion_f1 < self.macro_f1_val[-1]:
            self.best_criterion_f1 = self.macro_f1_val[-1]
            if self.criterion == 'f1':
                self.best_epoch = (self.i_epoch + 1)
            if (self.i_epoch + 1) > 0:
                #weights_save_model_name = self.model_save_name + '_network_weights_best_f1.pth'
                weights_save_model_name = 'network_weights_best_f1.pth'

                torch.save(self.network, os.path.join(self.dir_results, weights_save_model_name))
                # After saving best model, test and report to MLFlow

                try:
                    test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath = eval_bag_level_classification(test_generator=self.test_generator,
                                                  network=self.network,
                                                  weights2eval_path=os.path.join(self.dir_results, weights_save_model_name),
                                                  pred_column=self.pred_column,
                                                  pred_mode=self.pred_mode,
                                                  aggregation=self.aggregation,
                                                  results_save_path=self.dir_results,
                                                  best_model_type="f1",
                                                  return_params=True,
                                                  show_cf=False)

                    mlflow.log_metric("test_BF1_auc", test_roc_auc_score, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_f1", test_f1_score_w, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_recall", test_recall, step=self.i_epoch)
                    mlflow.log_metric("test_BF1_precision", test_precision, step=self.i_epoch)
                    #cf_img = Image.open(cf_savepath)
                    mlflow.log_artifact(cf_savepath, "test_BF1_cf")
                    #print(cf_savepath)

                except ValueError:
                    print("Only one class prediced (bad training) for ", str(weights_save_model_name))

        print("Logging to MLFlow...")
        # #Log to MLFLow
        mlflow.log_metric("train_loss", float(np.round(self.L_epoch, 4)), step=self.i_epoch)
        mlflow.log_metric("train_auc", float(np.round(self.macro_auc_lc_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("train_f1", float(np.round(self.macro_f1_train[-1], 4)), step=self.i_epoch)
        mlflow.log_metric("val_loss", float(np.round(Lce_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_auc", float(np.round(macro_auc_val, 4)), step=self.i_epoch)
        mlflow.log_metric("val_f1", float(np.round(f1_weighted_val, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_auc", float(np.round(self.best_criterion_auc, 4)), step=self.i_epoch)
        mlflow.log_metric("best_val_f1", float(np.round(self.best_criterion_f1, 4)), step=self.i_epoch)

    def test_bag_level_classification(self, generator, binary=False):
        self.network.eval()
        print('[VALIDATION]: at bag level...')

        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        Lce_e = 0

        with torch.no_grad():
            for self.i_iteration, (image_features, Y) in enumerate(tqdm(self.train_generator, leave=True, position=0)):
                #end_time = time.time()
                #seconds_elapsed = end_time - start_time
                #print("Iteration lasted in seconds: ", seconds_elapsed)


                image_features = torch.tensor(image_features).to('cuda')
                Y = torch.tensor(Y).to('cuda')

                # Forward network
                # if self.mode == 'embedding_GNN':
                #     Yprob, Yhat, logits, L_gnn = self.network(X_augm)
                # else:
                Yprob, Yhat, logits  = self.network(image_features)

                Yhat_all.append(Yprob.detach().cpu().numpy())
                # Estimate losses
                # Lce = self.L(Yhat, torch.squeeze(Y))
                #Lce = self.L(torch.squeeze(Yhat), torch.squeeze(Y))
                Ycat = torch.tensor([(Y == 1.).nonzero().item()]).to('cuda')

                Lce = custom_categorical_cross_entropy(y_pred=logits.squeeze(),
                                                       y_true=Ycat,
                                                       class_weights=self.class_weights)

                Lce_e += Lce.cpu().detach().numpy() / len(generator)

                Y_all.append(Y.detach().cpu().numpy())

                # Save predictions
                softmax_layer = torch.nn.Softmax(dim=0)

                #Yhat_all.append(softmax_layer(logits).detach().cpu().numpy())
                #Yhat_all.append(Yhat.detach().cpu().numpy())

        # Display losses per iteration
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce.cpu().detach().numpy(),
                            end_line='\r')
        # Obtain overall metrics
        Yhat_all = np.array(Yhat_all)
        Y_all = np.squeeze(np.array(Y_all))

        if binary:
            Yhat_all = np.max(Yhat_all, 1)
            # Y_all = np.max(Y_all, 1)

        # Compute overall metrics
        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all.squeeze(), axis=1)
        macro_cf = confusion_matrix(y_gt, y_pred)
        f1_weighted = sklearn.metrics.f1_score(y_gt, y_pred, average='weighted')

        macro_auc = roc_auc_score(Y_all, Yhat_all.squeeze(), multi_class='ovr')

        # Display losses per epoch
        self.display_losses(self.i_epoch + 1, self.epochs, self.i_iteration + 1, len(generator),
                            Lce_e, macro_auc,
                            end_line='\n')
        print(macro_cf)

        return Lce_e, macro_auc, f1_weighted
