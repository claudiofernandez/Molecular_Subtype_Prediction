# Imports
from MIL_utils import *
from MIL_data import *
from MIL_models import *
from MIL_trainer import *

def main_gcns_cv(args):

    # Set seeds for reproducibility
    set_seed(args.seed)

    # Set up directories depending on where this program is executed
    if args.where_exec == "slurm_nas":
        args.gnrl_data_dir = os.path.join('/workspace/NASFolder', args.dataset)
        output_directory = os.path.join(args.gnrl_data_dir, "output")
    elif args.where_exec == "slurm_dgx":
        #TODO: change so it is not directly BCNB here
        args.gnrl_data_dir = os.path.join('/workspace/DGXFolder', args.dataset)
        output_directory = os.path.join('/workspace/NASFolder', "output")
    elif args.where_exec == "dgx_gpu":
        args.gnrl_data_dir = os.path.join("../data", args.dataset)
        output_directory = os.path.join("../output")
    elif args.where_exec == "local":
        args.gnrl_data_dir = os.path.join("../data", args.dataset)
        output_directory = os.path.join("../output")

    os.makedirs(output_directory, exist_ok=True) # create output directory

    # Get ground truth data
    if args.dataset == "BCNB":
        dir_data_frame = os.path.join(args.gnrl_data_dir, "ground_truth", "patient-clinical-data.xlsx")
    elif args.dataset == "CLARIFY_DB":
        #TODO: Prepare for CLARIFY DB
        dir_data_frame = os.path.join(args.gnrl_data_dir, "ground_truth", "XXX.xlsx")

    # Set classes depending on training task
    if args.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
        classes = ['Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative']
    elif args.pred_mode == "LUMINALSvsHER2vsTNBC":
        classes = ['Luminal', 'HER2(+)', 'Triple negative']
    elif args.pred_mode == "OTHERvsTNBC":
        classes = ['Other', 'Triple negative']

    # Derive graphs dir depending on K
    for data_folder in os.listdir(os.path.join(args.gnrl_data_dir, "results_graphs_november_23")):
        if args.pred_mode in data_folder:
            mode_graphs_dir = os.path.join(args.gnrl_data_dir, "results_graphs_november_23", data_folder)
            for k_data_folder in os.listdir(mode_graphs_dir):
                if str(args.knn) in k_data_folder:
                    graphs_dir = os.path.join(mode_graphs_dir, k_data_folder)

    # Prepare CV folds
    dir_cv_dataset_splitting_path = os.path.join(args.gnrl_data_dir, "new_CV_folds_BCNB")
    files_folds_splitting_ids = os.listdir(dir_cv_dataset_splitting_path)
    folds_ids = np.unique(np.array([fold.split("_")[1] for fold in files_folds_splitting_ids])) # find the unique folds ids

    # Start CV
    for fold_id in folds_ids:
        train_ids_df, val_ids_df, test_ids_df = get_train_val_test_ids_dfs(fold_id=fold_id,
                                                                           files_folds_splitting_ids=files_folds_splitting_ids,
                                                                           dir_cv_dataset_splitting_path=dir_cv_dataset_splitting_path,
                                                                           dir_data_frame=dir_data_frame,
                                                                           pred_column=args.pred_column)

        print(" Loading training data...")
        dataset_train = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=train_ids_df, classes=classes,
                                                  pred_column=args.pred_column, pred_mode=args.pred_mode,
                                                  graphs_on_ram=args.preload_data)

        data_generator_train = MILDataGenerator_offline_graphs_balanced(dataset_train, batch_size=1,
                                                                        shuffle=args.shuffle,
                                                                        max_instances=np.inf,
                                                                        pred_column=args.pred_column,
                                                                        pred_mode=args.pred_mode,
                                                                        graphs_on_ram=args.preload_data)

        print(" Loading validation data...")
        dataset_val = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=val_ids_df, classes=classes,
                                                pred_column=args.pred_column, pred_mode=args.pred_mode,
                                                graphs_on_ram=args.preload_data)
        data_generator_val = MILDataGenerator_offline_graphs_nonbalanced(dataset_val, batch_size=1,
                                                                         shuffle=args.shuffle,
                                                                         max_instances=np.inf,
                                                                         pred_column=args.pred_column,
                                                                         pred_mode=args.pred_mode,
                                                                         graphs_on_ram=args.preload_data)

        print(" Loading testing data...")
        dataset_test = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=test_ids_df, classes=classes,
                                                 pred_column=args.pred_column, pred_mode=args.pred_mode,
                                                 graphs_on_ram=args.preload_data)
        data_generator_test = MILDataGenerator_offline_graphs_nonbalanced(dataset_test, batch_size=1,
                                                                          shuffle=args.shuffle,
                                                                          max_instances=np.inf,
                                                                          pred_column=args.pred_column,
                                                                          pred_mode=args.pred_mode,
                                                                          graphs_on_ram=args.preload_data)

        ###########
        #  MODEL  #
        ###########

        # Selecting the appropriate training device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(device))
        if torch.cuda.is_available():
            print('GPU Model: {}'.format(torch.cuda.get_device_name(0)))

        # For VGG16 BB
        model_dict = {"dropout": args.drop_out, 'n_classes': len(classes), "num_layers": args.num_gcn_layers,
                      "num_features": 512, "pooling":args.graph_pooling, "include_edge_features": args.include_edge_features,
                      "gnn_layer_type": args.gcn_layer_type}
        network = PatchGCN_MeanMax_LSelec(**model_dict)
        print(network)

        ############
        #  MLFLOW  #
        ############

        mlflow_run_name = "GT_" + str(args.gcn_layer_type) + "_GL_" + str(args.num_gcn_layers) +  "_KNN_" + str(args.knn) \
                   + "_EA_" + str(args.edge_agg) + "_EF_" + str(args.include_edge_features) + "_GP_" + str(args.graph_pooling) \
                   + "_DO_" + str(args.drop_out) + "_LR_" + str(args.lr).replace(".", "") + "_Optim_" + str(args.optimizer_type) \
                   + "_OWD_" + str(args.optimizer_weight_decay)+ "_CVFold_" + str(fold_id)

        # Get MLFlow arguments
        mlflow_experiment_name = args.mlflow_experiment_name
        mlflow_server_url = args.mlflow_server_url

        # Set up MLFlow server location, Otherwise location of ./mlruns
        mlflow.set_tracking_uri(mlflow_server_url)

        # Create MLFlow experiment
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        # Start MLFlow run
        mlflow.start_run(run_name=mlflow_run_name)  # If not provided, run_name will be randomly assigned

        # Log MLFlow Parameters
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        ##################
        # Start Training #
        ##################

        # Prepare output directories
        dir_out_exp = os.path.join(output_directory, args.mlflow_experiment_name)
        dir_out_run = os.path.join(dir_out_exp, mlflow_run_name)
        os.makedirs(dir_out_exp, exist_ok=True)
        os.makedirs(dir_out_run, exist_ok=True)


        trainer = Patch_GCN_offline_trainer(dir_out=dir_out_run, network=network, model_save_name=mlflow_run_name,
                                            lr=args.lr, aggregation=args.aggregation,
                                            alpha_ce=args.alpha_ce, id=id,
                                            early_stopping=args.early_stopping, scheduler=args.scheduler,
                                            virtual_batch_size=args.virtual_batch_size,
                                            criterion=args.criterion,
                                            class_weights=None,
                                            optimizer_type=args.optimizer_type,
                                            optimizer_weight_decay=args.optimizer_weight_decay,
                                            mlflow_run_name=mlflow_run_name,
                                            knn=args.knn)

        trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                      test_generator=data_generator_test, epochs=args.epochs, model_save_name=mlflow_run_name,
                      pred_column=args.pred_column, pred_mode=args.pred_mode, loss_function=args.loss_function)


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    #MLFlow configuration
    parser.add_argument("--mlflow_experiment_name", default="[02_11_2023] GCNs", type=str,  help='Name for experiment in MLFlow')
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # General Configuration
    parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--preload_data', default=True, type=lambda x: (str(x).lower() == 'true'), help="Load data on RAM memory.")
    parser.add_argument('--num_workers', type=int, default=200, help= "number of CPU processes or threads used for data loading and preprocessing during training.")
    parser.add_argument('--seed', type=int, default=42, help="seed that will be used for reproducibility.")

    # Mode Configuration
    parser.add_argument('--pred_column', type=str, default="Molecular subtype", help='Column from GT dataframe that we want to predict')
    parser.add_argument('--training_type', type=str, default="CV", help='CV or full_dataset')
    parser.add_argument('--dataset', type=str, default="BCNB", help='BCNB or CLARIFY_DB')
    parser.add_argument('--shuffle', default=False, type=lambda x: (str(x).lower() == 'true'), help="Load data on RAM memory.")
    parser.add_argument("--aggregation", default='Patch_GCN_offline', type=str)  # max, mean, TransMIL, TransMIL_pablo, Patch_GCN_online, Patch_GCN_offline
    parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str)  # "LUMINALAvsLAUMINALBvsHER2vsTNBC", "LUMINALSvsHER2vsTNBC", "OTHERvsTNBC"
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")
    parser.add_argument("--loss_function", default="cross_entropy", type=str)  # cross_entropy, kll
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")  #  [0.00002, 0.00001, 0.0002, 0.0001]
    parser.add_argument("--criterion", default='auc', type=str, help="Metric to keep the best model: 'auc', 'f1'")
    parser.add_argument("--optimizer_type", default='adam', type=str)  # sgd, adam, lookahead_radam
    parser.add_argument("--optimizer_weight_decay", default=0.00001, type=float)
    parser.add_argument("--alpha_ce", default=1., type=float)
    parser.add_argument("--early_stopping", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--scheduler", default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--virtual_batch_size", default=1, type=int)

    # Graph Configuration
    parser.add_argument('--knn', type=int, default=8, help='# of K nearest neighbours for graph creation.')  # 8, 19, 25
    parser.add_argument('--edge_agg', type=str, default='spatial', help="What edge relationship to use for aggregation.")  # It is possible to use spatial or latent edge aggregation
    parser.add_argument('--include_edge_features',  default=False, type=lambda x: (str(x).lower() == 'true'), help='Include edge_features (euclidean dist) in the graph')

    # GCN Configuration
    parser.add_argument('--gcn_layer_type', type=str, default="GCNConv", help='Type of GCN layers to use.')  # ['GCNConv', 'SAGEConv', 'GATConv', 'GINConv', 'GENConv', 'GraphConv']
    parser.add_argument('--num_gcn_layers', type=int, default=4, help='# of GCN layers to use.')  # [4, 5]
    parser.add_argument('--graph_pooling', type=str, default="mean", help="mean, max, attention")  # TODO: CHECK Graph pooling
    parser.add_argument('--drop_out', default=True, type=lambda x: (str(x).lower() == 'true'), help='Enable dropout (p=0.25)')

    args = parser.parse_args()
    main_gcns_cv(args)