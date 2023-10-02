# Imports
from MIL_utils import *
from MIL_data import *
from MIL_models import *
from MIL_trainer import *
from MIL_data_utils import *

def train_exp(args_train, dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test):

    #################################################
    # Set Model Architecture for current experiment #
    #################################################

    # Prepare GPU Training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    if torch.cuda.is_available():
        print('GPU Model: {}'.format(torch.cuda.get_device_name(0)))

    # Set network architecture depending on aggregation type

    if args_train.aggregation == "max" or args_train.aggregation == "mean" or args_train.aggregation == "attention" or args_train.aggregation == "TransMIL":
        network = MILArchitecture(classes=args_train.classes, pretrained=args.pretrained, mode=args.mode,
                                  aggregation=args_train.aggregation, freeze_bb_weights=args.freeze_bb_weights,
                                  backbone=args_train.network_backbone, include_background=args_train.include_background)

    elif "Patch_GCN" in args.aggregation:
        if "resnet50" in args.network_backbone:
            model_dict = {"dropout": args.drop_out, 'n_classes': len(args_train.classes), "num_layers": args.num_gcn_layers,
                          "num_features": 1024}
            network = PatchGCN(**model_dict)
        elif "vgg16" in args.network_backbone:
            model_dict = {"dropout": args.drop_out, 'n_classes': len(args_train.classes), "num_layers": args.num_gcn_layers,
                          "num_features": 512}
            network = PatchGCN(**model_dict)

    print(network)

    ##################################################
    # Set Training Parameters for current experiment #
    ##################################################

    if args.train_test_mode == "train":
        # MlFlow Parameters
        mlruns_folder = "../output/mlruns"
        mlflow_experiment_name = args.mlflow_experiment_name
        mlflow_run_name = args_train.run_name

        mlflow.set_tracking_uri(mlruns_folder)
        experiment = mlflow.set_experiment(mlflow_experiment_name)
        mlflow.start_run(run_name=mlflow_run_name)

        # Log Parameters
        # Log Parameters
        for key, value in vars(args_train).items():
            mlflow.log_param(key, value)

        # mlflow.log_param("patch_size", args_train.patch_size)
        # mlflow.log_param("data_augmentation", args_train.data_augmentation)
        # mlflow.log_param("stain_normalization", args_train.stain_normalization)
        # mlflow.log_param("regions_filled", args_train.regions_filled)
        # mlflow.log_param("max_instances", args_train.max_instances)
        # mlflow.log_param("ordered", args_train.ordered)
        # mlflow.log_param("pred_mode", args_train.pred_mode)
        # mlflow.log_param("magnification_level", args_train.magnification_level)
        # mlflow.log_param("pred_mode", args_train.pred_mode)
        #
        #
        # mlflow.log_param("nn_backbone", args_train.network_backbone)
        # mlflow.log_param("pretrained", args.pretrained)
        # mlflow.log_param("epochs", args.epochs)
        # mlflow.log_param("learning_rate", args_train.lr)
        # mlflow.log_param("criterion", args.criterion)
        # mlflow.log_param("aggregation", args_train.aggregation)
        # # mlflow.log_param("norm_c_weights", args.class_weights_norm)
        # mlflow.log_param("balanced_train_datagen", args_train.balanced_train_datagen)
        # mlflow.log_param("tissue_percentages_max", args_train.tissue_percentages_max)
        #
        # mlflow.log_param("optimizer", args_train.optimizer_type)
        # mlflow.log_param("optim_weight_decay", args.optimizer_weight_decay)
        # mlflow.log_param("loss_function", args.loss_function)
        # mlflow.log_param("freeze_bb_weights", args.freeze_bb_weights)

        # Patch GCN
        if "Patch_GCN" in args_train.aggregation:
            mlflow.log_param("num_gcn_layers", args.num_gcn_layers)
            mlflow.log_param("knn", args_train.knn)
            mlflow.log_param("include_edge_features", args_train.include_edge_features)
            mlflow.log_param("edge_agg", args.edge_agg)

        # # Compute Class Weights for Imbalanced Dataset
        # if args.class_weights_norm:
        #     class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
        #                                                                     classes=args_train.dataset_train.classes,
        #                                                                     y=args_train.dataset_train.y)
        # else:
        #
        class_weights = np.ones(len(args_train.classes))

        ##################
        # Start Training #
        ##################
        optimizer_type = args_train.optimizer_type
        if args_train.aggregation == "max" or args_train.aggregation == "mean" or args_train.aggregation == "attention" or args_train.aggregation == "TransMIL" or args_train.aggregation == "TransMIL_Pablo":
            trainer = TransMIL_trainer(dir_out=args_train.dir_out, network=network, model_save_name=args_train.run_name,
                                       aggregation=args_train.aggregation,
                                       lr=args_train.lr,
                                       alpha_ce=args.alpha_ce, id=id,
                                       early_stopping=args.early_stopping, scheduler=args.scheduler,
                                       virtual_batch_size=args.virtual_batch_size,
                                       criterion=args.criterion,
                                       class_weights=class_weights,
                                       optimizer_type=optimizer_type,
                                       optimizer_weight_decay=args.optimizer_weight_decay,
                                       mlflow_run_name=mlflow_run_name)

            trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                          test_generator=data_generator_test, epochs=args.epochs, model_save_name=args_train.run_name,
                          pred_column=args_train.pred_column, pred_mode=args_train.pred_mode, loss_function=args.loss_function)

        elif args_train.aggregation == "Patch_GCN_online":
            trainer = Patch_GCN_online_trainer(dir_out=args_train.dir_out, network=network, model_save_name=args_train.run_name,
                                               lr=args_train.lr, aggregation=args_train.aggregation,
                                               alpha_ce=args.alpha_ce, id=id,
                                               early_stopping=args.early_stopping, scheduler=args.scheduler,
                                               virtual_batch_size=args.virtual_batch_size,
                                               criterion=args.criterion,
                                               class_weights=class_weights,
                                               optimizer_type=optimizer_type,
                                               optimizer_weight_decay=args.optimizer_weight_decay,
                                               mlflow_run_name=mlflow_run_name,
                                               knn=args_train.knn,
                                               node_feature_extractor=args.network_backbone)

            trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                          test_generator=data_generator_test, epochs=args.epochs, model_save_name=args_train.run_name,
                          pred_column=args_train.pred_column, pred_mode=args_train.pred_mode, loss_function=args.loss_function)

        elif args_train.aggregation == "Patch_GCN_offline":
            trainer = Patch_GCN_offline_trainer(dir_out=args_train.dir_out, network=network, model_save_name=args_train.run_name,
                                                lr=args_train.lr, aggregation=args_train.aggregation,
                                                alpha_ce=args.alpha_ce, id=id,
                                                early_stopping=args.early_stopping, scheduler=args.scheduler,
                                                virtual_batch_size=args.virtual_batch_size,
                                                criterion=args.criterion,
                                                class_weights=class_weights,
                                                optimizer_type=optimizer_type,
                                                optimizer_weight_decay=args.optimizer_weight_decay,
                                                mlflow_run_name=mlflow_run_name,
                                                knn=args_train.knn,
                                                node_feature_extractor=args.network_backbone)

            trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                          test_generator=data_generator_test, epochs=args.epochs, model_save_name=args_train.run_name,
                          pred_column=args_train.pred_column, pred_mode=args_train.pred_mode, loss_function=args.loss_function)

            print("Training finished.")

def main_cv(args):
    # Effective training
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    CUDA_LAUNCH_BLOCKING = 1

    # Split the comma-separated string into a list of floats
    args.pred_modes = [str(pred_mode) for pred_mode in args.pred_modes.split(',')]
    args.learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
    args.network_backbones = [str(network_backbone) for network_backbone in args.network_backbones.split(',')]
    args.aggregations = [str(aggregation) for aggregation in args.aggregations.split(',')]
    args.optimizers_types = [str(optimizer_type) for optimizer_type in args.optimizers_types.split(',')]

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
        args.gnrl_data_dir = "../local_dir_where_files_are_stored"
        output_directory = os.path.join("../local_dir_where_output_will_be_stored")

    # Prepare dataloaders for the different datasets
    if args.training_type == "CV":

        # Prepare dataset for each CV fold
        dir_cv_dataset_splitting_path = os.path.join(args.gnrl_data_dir, "new_CV_folds_BCNB")

        files_folds_splitting_ids = os.listdir(dir_cv_dataset_splitting_path)

        folds_ids = np.unique(np.array([fold.split("_")[1] for fold in files_folds_splitting_ids])) # find the unique folds ids

        # Initialize variables

        for fold_id in folds_ids:
            print(f"[CV FOLD {fold_id}]")

            for pred_mode in args.pred_modes:

                # Set classes depending on training task
                if pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                    classes = ['Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative']
                elif pred_mode == "LUMINALSvsHER2vsTNBC":
                    classes = ['Luminal', 'HER2(+)', 'Triple negative']
                elif pred_mode == "OTHERvsTNBC":
                    classes = ['Other', 'Triple negative']

                # Retrieve dataset split for each fold
                dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test = load_CV_fold_dataset(
                    fold_id, args.magnification_level, args.pred_column, pred_mode, classes, args.regions_filled,
                    dir_cv_dataset_splitting_path,
                    args.ordered, args.patch_size, args.max_instances, args.data_augmentation, args.stain_normalization,
                    args.images_on_ram, args.include_background, args.balanced_train_datagen, args.tissue_percentages_max)

                for network_backbone in args.network_backbones:
                    for aggregation in args.aggregations:
                        for learning_rate in args.learning_rates:
                            for optimizer_type in args.optimizers_types:
                                # TODO: fix run_name generation to list comprehension
                                if "Patch_GCN" in aggregation:
                                    run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(args.aggregation) + "_ML_" + str(
                                        args.magnification_level) + "_NN_bb_" + str(
                                        args.network_backbone) + "_FBB_" + str(
                                        args.freeze_bb_weights) + "_PS_" + str(args.patch_size) + "_DA_" + str(
                                        args.data_augmentation) + "_SN_" + str(
                                        args.stain_normalization) + "_L_" + args.criterion + "_E_" + str(
                                        args.epochs) + "_LR_" + str(
                                        args.lr).replace(
                                        ".", "") + "_Order_" + str(args.ordered) + "_Optim_" + str(
                                        args.optimizer_type) + "_N_" + str(
                                        args.max_instances) + "_BDG_" + str(
                                        args.balanced_train_datagen) + "_OWD_" + str(
                                        args.optimizer_weight_decay) + "_TP_" + str(
                                        args.tissue_percentages_max) + "_KNN_" + str(
                                        args.knn) + "_GCNL_" + str(args.num_gcn_layers) + "_EF_" + str(
                                        args.include_edge_features) + "_CVFold_" + str(fold_id)
                                else:
                                    run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(aggregation) + "_ML_" + str(
                                        args.magnification_level) + "_NN_bb_" + str(network_backbone) + "_FBB_" + str(
                                        args.freeze_bb_weights) + "_PS_" + str(args.patch_size) + "_DA_" + str(
                                        args.data_augmentation) + "_SN_" + str(
                                        args.stain_normalization) + "_L_" + args.criterion + "_E_" + str(
                                        args.epochs) + "_LR_" + str(
                                        learning_rate).replace(
                                        ".", "") + "_Order_" + str(args.ordered) + "_Optim_" + str(
                                        optimizer_type) + "_N_" + str(
                                        args.max_instances) + "_BDG_" + str(
                                        args.balanced_train_datagen) + "_OWD_" + str(
                                        args.optimizer_weight_decay) + "_TP_" + str(
                                        args.tissue_percentages_max) + "_CVFold_" + str(
                                        fold_id)

                                # Prepare output directories
                                # TODO: Modify output directories vbased on relative general path and where_exec
                                dir_results = '../data/results/'
                                dir_out_gnrl = os.path.join(dir_results, args.mlflow_experiment_name)
                                dir_out_main = os.path.join(dir_results, args.mlflow_experiment_name,
                                                            run_name.split("_CVFold_")[0])
                                # dir_out = os.path.join(dir_out_main, run_name)
                                # TODO: Check this because it is only saving the first CVFold1
                                dir_out = os.path.join(dir_out_main, "CVFold_" + str(fold_id))
                                if not os.path.isdir(dir_results):
                                    os.mkdir(dir_results)
                                if not os.path.isdir(dir_out_gnrl):
                                    os.mkdir(dir_out_gnrl)
                                if not os.path.isdir(dir_out_main):
                                    os.mkdir(dir_out_main)
                                if not os.path.isdir(dir_out):
                                    os.mkdir(dir_out)

                                # Create argument parser for training
                                parser_train = argparse.ArgumentParser()
                                parser_train.add_argument("--aggregation", default=aggregation,
                                                          type=str)  # max, mean, TransMIL, TransMIL_pablo, Patch_GCN_online, Patch_GCN_offline
                                parser_train.add_argument("--lr", default=learning_rate, type=float)  # Learning rate
                                parser_train.add_argument("--network_backbone", default=network_backbone,
                                                          type=str)  # vgg16, resnet50, vgg16_512
                                parser_train.add_argument("--optimizer_type", default=optimizer_type,
                                                          type=str)  # vgg16, resnet50, vgg16_512
                                parser_train.add_argument("--run_name", default=run_name,
                                                          type=str)
                                parser_train.add_argument("--classes", default=classes,
                                                          type=list)
                                parser_train.add_argument("--include_background", default=args.include_background,
                                                          type=str)
                                parser_train.add_argument("--patch_size", default=args.patch_size,
                                                          type=int)
                                parser_train.add_argument("--data_augmentation", default=args.data_augmentation,
                                                          type=str)
                                parser_train.add_argument("--stain_normalization", default=args.stain_normalization,
                                                          type=bool)
                                parser_train.add_argument("--regions_filled", default=args.regions_filled,
                                                          type=str)
                                parser_train.add_argument("--max_instances", default=args.max_instances,
                                                          type=int)
                                parser_train.add_argument("--ordered", default=args.ordered,
                                                          type=bool)
                                parser_train.add_argument("--pred_mode", default=pred_mode,
                                                          type=str)
                                parser_train.add_argument("--balanced_train_datagen",
                                                          default=args.balanced_train_datagen,
                                                          type=bool)
                                parser_train.add_argument("--tissue_percentages_max",
                                                          default=args.tissue_percentages_max,
                                                          type=str)
                                parser_train.add_argument("--dir_out", default=dir_out,
                                                          type=str)
                                parser_train.add_argument("--pred_column", default=args.pred_column,
                                                          type=str)
                                parser_train.add_argument("--magnification_level", default=args.magnification_level,
                                                          type=str)

                                args_train = parser_train.parse_args()
                                train_exp(args_train, dataset_train, data_generator_train, dataset_val,
                                          data_generator_val,
                                          dataset_test, data_generator_test)

            print("holu")

    elif args.training_type == "full_dataset":

        if args.dataset == "BCNB":

            for pred_mode in args.pred_modes:

                # Retrieve dataset split for the whole BCNB dataset
                #TODO: check if we really do need to retrieve the datasets. Maybe only data generators??
                classes, dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test = load_BCNB_full_dataset(
                   gnrl_data_dir=args.gnrl_data_dir, magnification_level=args.magnification_level, pred_column=args.pred_column,
                    pred_mode=pred_mode, regions_filled=args.regions_filled, ordered=args.ordered, patch_size=args.patch_size,
                    max_instances=args.max_instances, data_augmentation=args.data_augmentation, stain_normalization=args.stain_normalization,
                    images_on_ram=args.images_on_ram, include_background=args.include_background, balanced_train_datagen=args.balanced_train_datagen,
                    tissue_percentages_max=args.tissue_percentages_max, where_exec=args.where_exec)

                for network_backbone in args.network_backbones:
                    for aggregation in args.aggregations:
                        for learning_rate in args.learning_rates:
                            for optimizer_type in args.optimizers_types:
                                #TODO: Fix for GCN
                                if "Patch_GCN" in aggregation:
                                    mlflow_run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(
                                        args.aggregation) + "_ML_" + str(
                                        args.magnification_level) + "_NN_bb_" + str(
                                        args.network_backbone) + "_FBB_" + str(
                                        args.freeze_bb_weights) + "_PS_" + str(args.patch_size) + "_DA_" + str(
                                        args.data_augmentation) + "_SN_" + str(
                                        args.stain_normalization) + "_L_" + args.criterion + "_E_" + str(
                                        args.epochs) + "_LR_" + str(
                                        args.lr).replace(
                                        ".", "") + "_Order_" + str(args.ordered) + "_Optim_" + str(
                                        args.optimizer_type) + "_N_" + str(
                                        args.max_instances) + "_BDG_" + str(
                                        args.balanced_train_datagen) + "_OWD_" + str(
                                        args.optimizer_weight_decay) + "_TP_" + str(
                                        args.tissue_percentages_max) + "_KNN_" + str(
                                        args.knn) + "_GCNL_" + str(args.num_gcn_layers) + "_EF_" + str(
                                        args.include_edge_features) + "_CVFold_" + str(fold_id)
                                else:

                                    mlflow_run_name_ini = "PM_" + str(pred_mode) + "_BB_" + str(
                                        network_backbone) + "_AGGR_" + str(
                                        aggregation) + "_LR_" + str(learning_rate) + "_OPT_" + str(optimizer_type)

                                    key_substitutions = {
                                        'training_type': 'T',
                                        'dataset': 'D',
                                        'epochs': 'E',
                                        'loss_function': 'L',
                                        'optimizer_weight_decay': 'OWD',
                                        'freeze_bb_weights': 'FBB',
                                        'batch_size': 'BS',
                                        'pretrained': 'PT',
                                        'magnification_level': 'MAGN',
                                        'data_augmentation': 'AUGM',
                                        'stain_normalization': 'SN',
                                        'balanced_train_datagen': 'BAL',
                                        'max_instances': 'N'
                                    }
                                    config_dict = vars(args)
                                    mlflow_run_name = "_".join(
                                        [f"{key_substitutions.get(key, key)}_{value}".replace(":", "_")
                                         for key, value in config_dict.items()
                                         if "dir" not in key and "where_exec" not in key
                                         and "train_test_mode" not in key and "mlflow_experiment_name" not in key
                                         and "regions_filled" not in key and "ordered" not in key
                                         and "patch_size" not in key and "images_on_ram" not in key
                                         and "include_background" not in key and "tissue_percentages_max" not in key
                                         and "early_stopping" not in key and "scheduler" not in key
                                         and "virtual_batch_size" not in key and "pred_modes" not in key
                                         and "aggregations" not in key and "network_backbones" not in key
                                         and "optimizers_types" not in key and "learning_rates" not in key
                                         and "pred_column" not in key and "bag_id" not in key and "mode" not in key
                                         and "criterion" not in key and "alpha_ce" not in key])

                                    mlflow_run_name = mlflow_run_name_ini + "_" + mlflow_run_name



                                # Prepare output directories
                                dir_out_exp = os.path.join(output_directory, args.mlflow_experiment_name)
                                dir_out_run = os.path.join(output_directory, args.mlflow_experiment_name, mlflow_run_name)

                                # dir_out = os.path.join(dir_out_main, run_name)
                                os.makedirs(output_directory, exist_ok=True)
                                os.makedirs(dir_out_exp, exist_ok=True)
                                os.makedirs(dir_out_run, exist_ok=True)

                                # Create argument parser for training
                                parser_train = argparse.ArgumentParser()
                                parser_train.add_argument("--aggregation", default=aggregation,
                                                          type=str)  # max, mean, TransMIL, TransMIL_pablo, Patch_GCN_online, Patch_GCN_offline
                                parser_train.add_argument("--lr", default=learning_rate, type=float)  # Learning rate
                                parser_train.add_argument("--network_backbone", default=network_backbone,
                                                          type=str)  # vgg16, resnet50, vgg16_512
                                parser_train.add_argument("--optimizer_type", default=optimizer_type,
                                                          type=str)  # vgg16, resnet50, vgg16_512
                                parser_train.add_argument("--run_name", default=mlflow_run_name,
                                                          type=str)
                                parser_train.add_argument("--classes", default=classes,
                                                          type=list)
                                parser_train.add_argument("--include_background", default=args.include_background,
                                                          type=str)
                                parser_train.add_argument("--patch_size", default=args.patch_size,
                                                          type=int)
                                parser_train.add_argument("--data_augmentation", default=args.data_augmentation,
                                                          type=str)
                                parser_train.add_argument("--stain_normalization", default=args.stain_normalization,
                                                          type=bool)
                                parser_train.add_argument("--regions_filled", default=args.regions_filled,
                                                          type=str)
                                parser_train.add_argument("--max_instances", default=args.max_instances,
                                                          type=int)
                                parser_train.add_argument("--ordered", default=args.ordered,
                                                          type=bool)
                                parser_train.add_argument("--pred_mode", default=pred_mode,
                                                          type=str)
                                parser_train.add_argument("--balanced_train_datagen",
                                                          default=args.balanced_train_datagen,
                                                          type=bool)
                                parser_train.add_argument("--tissue_percentages_max",
                                                          default=args.tissue_percentages_max,
                                                          type=str)
                                parser_train.add_argument("--dir_out", default=dir_out_run,
                                                          type=str)
                                parser_train.add_argument("--pred_column", default=args.pred_column,
                                                          type=str)
                                parser_train.add_argument("--magnification_level", default=args.magnification_level,
                                                          type=str)

                                print("he llegado aqui")

                                args_train = parser_train.parse_args()
                                train_exp(args_train, dataset_train, data_generator_train, dataset_val,
                                          data_generator_val,
                                          dataset_test, data_generator_test)


                print("hola")
        elif args.dataset == "CLARIFY_DB":
            # TODO: add code full CLARIFY dataset
            print("Write code for full CLARIFY_DB")
        else:
            print("Dataset not recognized.")


    print("holu")


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    #MLFlow configuration
    parser.add_argument("--mlflow_experiment_name", default="[02_10_2023] BB MolSub 10x BCNB Final", type=str,  help='Name for experiment in MLFlow')

    # Directories.
    parser.add_argument('--where_exec', type=str, default="dgx_gpu", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--training_type', type=str, default="full_dataset" , help='CV or full_dataset')
    parser.add_argument('--dataset', type=str, default="BCNB", help='BCNB or CLARIFY_DB')

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
    parser.add_argument("--pred_modes", default="LUMINALSvsHER2vsTNBC,OTHERvsTNBC", type=str, help="Comma-separated list of prediction modes. Choose (1 or more) between: 'LUMINALAvsLAUMINALBvsHER2vsTNBC', 'LUMINALSvsHER2vsTNBC', 'OTHERvsTNBC'")
    parser.add_argument("--network_backbones", default='vgg16, resnet50', type=str, help="Comma-separated list of backbones. Choose (1 or more) betwwen: 'vgg16', 'resnet50'")
    parser.add_argument("--magnification_level", default="5x", type=str, help="5x, 10x, 20x")
    parser.add_argument("--aggregations", default="mean, max", type=str, help="Comma-separated list of MIL aggregations. Choose (1 or more) between: mean', 'max' 'attention', 'TransMIL'")

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
