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
        args.gnrl_data_dir = '/workspace/NASFolder'
        output_directory = os.path.join(args.gnrl_data_dir, "output")
    elif args.where_exec == "slurm_dgx":
        #TODO: change so it is not directly BCNB here
        args.gnrl_data_dir = '/workspace/DGXFolder/BCNB'
        output_directory = os.path.join('/workspace/NASFolder', "output")
    elif args.where_exec == "dgx_gpu":
        args.gnrl_data_dir = "../data/BCNB/"
        output_directory = os.path.join("../output")
    elif args.where_exec == "local":
        #TODO: write for executing locally
        args.gnrl_data_dir = "../data/results_graphs_november_23"
        output_directory = os.path.join("../output")

    #############################################################################
    # Set train and validation generators depending on the MIL aggregation type #
    #############################################################################

    if aggregation == "Patch_GCN_offline":  # Datasets and Datagens for offline graphs
        print(" Loading training data...")
        dataset_train = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=train_ids_df, classes=classes,
                                                  pred_column=pred_column, pred_mode=pred_mode,
                                                  magnification_level=magnification_level,
                                                  bag_id='Patient ID', input_shape=input_shape,
                                                  graphs_on_ram=graphs_on_ram)
        if balanced_train_datagen:
            data_generator_train = MILDataGenerator_offline_graphs_balanced(dataset_train, batch_size=1,
                                                                            shuffle=not (ordered),
                                                                            max_instances=max_instances,
                                                                            num_workers=0,
                                                                            pred_column=pred_column,
                                                                            pred_mode=pred_mode,
                                                                            graphs_on_ram=graphs_on_ram)
        else:
            data_generator_val = MILDataGenerator_offline_graphs_nonbalanced(dataset_train, batch_size=1,
                                                                             shuffle=not (ordered),
                                                                             max_instances=max_instances,
                                                                             num_workers=0,
                                                                             pred_column=pred_column,
                                                                             pred_mode=pred_mode,
                                                                             graphs_on_ram=graphs_on_ram)

        # for idx_iter, (G, Y) in enumerate(data_generator_train):
        #     print(idx_iter)
        #     print(G)
        #     print(Y)

        print(" Loading validation data...")
        dataset_val = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=val_ids_df, classes=classes,
                                                pred_column=pred_column, pred_mode=pred_mode,
                                                magnification_level=magnification_level,
                                                bag_id='Patient ID', input_shape=input_shape,
                                                graphs_on_ram=graphs_on_ram)
        data_generator_val = MILDataGenerator_offline_graphs_nonbalanced(dataset_val, batch_size=1,
                                                                         shuffle=not (ordered),
                                                                         max_instances=max_instances, num_workers=0,
                                                                         pred_column=pred_column,
                                                                         pred_mode=pred_mode,
                                                                         graphs_on_ram=graphs_on_ram)

        print(" Loading testing data...")
        dataset_test = MILDataset_offline_graphs(dir_graphs=graphs_dir, data_frame=test_ids_df, classes=classes,
                                                 pred_column=pred_column, pred_mode=pred_mode,
                                                 magnification_level=magnification_level,
                                                 bag_id='Patient ID', input_shape=input_shape,
                                                 graphs_on_ram=graphs_on_ram)
        data_generator_test = MILDataGenerator_offline_graphs_nonbalanced(dataset_test, batch_size=1,
                                                                          shuffle=not (ordered),
                                                                          max_instances=max_instances,
                                                                          num_workers=0,
                                                                          pred_column=pred_column,
                                                                          pred_mode=pred_mode,
                                                                          graphs_on_ram=graphs_on_ram)


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    #MLFlow configuration
    parser.add_argument("--mlflow_experiment_name", default="[02_11_2023] GCNs", type=str,  help='Name for experiment in MLFlow')
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8001", help='URL of MLFlow DB')

    # General Configuration
    parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--preload_data', default=True, type=lambda x: (str(x).lower() == 'true'), help="Load data on RAM memory.")
    parser.add_argument('--num_workers', type=int, default=200, help= "number of CPU processes or threads used for data loading and preprocessing during training.")
    parser.add_argument('--seed', type=int, default=42, help="seed that will be used for reproducibility.")

    # Mode Configuration
    parser.add_argument('--training_type', type=str, default="CV" , help='CV or full_dataset')
    parser.add_argument('--dataset', type=str, default="BCNB", help='BCNB or CLARIFY_DB')
    parser.add_argument("--aggregation", default='Patch_GCN_offline', type=str)  # max, mean, TransMIL, TransMIL_pablo, Patch_GCN_online, Patch_GCN_offline






    # Training configuration
    parser.add_argument("--train_test_mode", default="train", type=str, help="Select train, test, test_allmymodels")
    parser.add_argument("--epochs", default=5, type=int, help="Number of epochs for training")
    parser.add_argument("--learning_rates", default="0.002", type=str, help="Comma-separated list of learning rates. Choose (1 or more): 0.002, 0.0001")
    parser.add_argument("--loss_function", default="cross_entropy", type=str, help="Loss function: cross_entropy, kll")
    parser.add_argument("--optimizers_types", default="sgd", type=str, help="Comma-separated list of optimizers. Choose (1 or more): sgd, adam, lookahead_radam")
    parser.add_argument("--optimizer_weight_decay", default=0, type=float)
    parser.add_argument("--freeze_bb_weights", default=False, type=bool, help="Freeze feature extractor weights. False: retrain True: transfer learning")
    parser.add_argument("--pretrained", default=True, type=bool, help="False: retrain from scratch True: Use pretrained feature extractor on ImageNet.")
    parser.add_argument("--criterion", default='auc', type=str, help="Metric to keep the best model: 'auc', 'f1'")
    parser.add_argument("--mode", default="embedding", type=str)  # embedding,embedding_GNN
    parser.add_argument("--alpha_ce", default=1., type=float)

    # Model configuration
    parser.add_argument("--pred_modes", default="LUMINALSvsHER2vsTNBC", type=str, help="Comma-separated list of prediction modes. Choose (1 or more) between: 'LUMINALAvsLAUMINALBvsHER2vsTNBC', 'LUMINALSvsHER2vsTNBC', 'OTHERvsTNBC'")
    parser.add_argument("--network_backbones", default='vgg16,resnet50', type=str, help="Comma-separated list of backbones. Choose (1 or more) betwwen: 'vgg16', 'resnet50'")
    parser.add_argument("--magnification_level", default="5x", type=str, help="5x, 10x, 20x")
    parser.add_argument("--aggregations", default="mean,max", type=str, help="Comma-separated list of MIL aggregations. Choose (1 or more) between: mean', 'max' 'attention', 'TransMIL'")

    # MIL Parameters
    parser.add_argument("--bag_id", default="Patient ID", type=str, help="Identifier of Bags")
    parser.add_argument("--max_instances", default=100, type=int, help="Max number of instances per bag.")
    parser.add_argument("--pred_column", default="Molecular subtype", type=str, help="Name of dataframe column that you want to predict.")
    parser.add_argument("--regions_filled", default="fullWSIs_TP_0", type=str, help="")
    parser.add_argument("--ordered", default=True, type=bool, help="Ordered dataset.")
    parser.add_argument("--patch_size", default=512, type=int, help="Ordered dataset.")
    parser.add_argument("--data_augmentation", default="non-spatial", type=str, help="Type of image augmentations")
    parser.add_argument("--stain_normalization", default=False, type=bool, help="Normalize staining of the patches.")
    parser.add_argument("--images_on_ram", default=True, type=bool, help="Preload images on RAM memory")
    #TODO: remove include background
    parser.add_argument("--include_background", default=False, type=bool, help="Preload images on RAM memory")
    parser.add_argument("--balanced_train_datagen", default=True, type=bool, help="Balance dataset classes.")
    parser.add_argument("--tissue_percentages_max", default="O_0.4-T_1-S_1-I_1-N_1", type=str, help="tissue_percentages_max")

    # Miscellaneous
    parser.add_argument("--early_stopping", default=True, type=bool)
    parser.add_argument("--scheduler", default=True, type=bool)
    parser.add_argument("--virtual_batch_size", default=1, type=int)

    # # Patch GCN Parameters
    # parser.add_argument('--include_edge_features',  default=False, help='Include edge_features (euclidean dist) in  the graph')
    # parser.add_argument('--num_gcn_layers',  type=int, default=5, help='# of GCN layers to use.')
    # parser.add_argument('--node_knn',  type=int, default=knn, help='# of K nearest neighbours for graph creation.')
    # parser.add_argument('--node_feature_extractor', type=str, default=node_feature_extractor, help="Feature extractor that will be used for creating the nodes of the graph.") # resnet50_3blocks_1024
    # parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.") # It is possible to use spatial or latent edge aggregation
    # parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
    # parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
    #

    args = parser.parse_args()
    main_cv(args)