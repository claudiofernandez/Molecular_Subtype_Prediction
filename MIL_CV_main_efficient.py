# Imports
from MIL_utils import *
from MIL_data import *
from MIL_models import *
from MIL_trainer import *

def load_CV_fold_dataset(fold_id, magnification_level, pred_column, pred_mode, classes, regions_filled, dir_cv_dataset_splitting_path, ordered, patch_size, max_instances, data_augmentation, stain_normalization, images_on_ram, include_background, balanced_train_datagen, tissue_percentages_max):

    ##############################################
    # Set directories and parameters for the data #
    ##############################################

    # Directories
    # TODO: change all directories to slurm/local
    dir_data_frame = '../data/BCNB/patient-clinical-data.xlsx'
    dir_results = '../output/results/'
    dir_excels_class_perc = "../data/BCNB/patches_paths_class_perc/"

    # Set input shape of the images depending on the chosen magnification
    if magnification_level == "20x":
        input_shape = (3, 512, 512)
    elif magnification_level == "10x":
        input_shape = (3, 256, 256)
    elif magnification_level == "5x":
        input_shape = (3, 128, 128)

    print("Magnification level: " + str(magnification_level) + " using input shape: " + str(input_shape))


    files_folds_splitting_ids = os.listdir(dir_cv_dataset_splitting_path)

    folds_ids = np.unique(np.array([fold.split("_")[1] for fold in files_folds_splitting_ids])) # find the unique folds ids

    # Initialize variables
    print(f"[CV FOLD {fold_id}]")
    # Retrieve dataset split for each fold
    train_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_train")][0]
    val_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_val")][0]
    test_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_test")][0]

    with open(os.path.join(dir_cv_dataset_splitting_path, train_file), "r") as f:
        train_ids = f.read().splitlines()

    with open(os.path.join(dir_cv_dataset_splitting_path, val_file), "r") as f:
        val_ids = f.read().splitlines()

    with open(os.path.join(dir_cv_dataset_splitting_path, test_file), "r") as f:
        test_ids = f.read().splitlines()

    # Convert IDs from strings to ints
    train_ids = [int(id) for id in train_ids]
    val_ids = [int(id) for id in val_ids]
    test_ids = [int(id) for id in test_ids]

    # Read GT DataFrame
    df = pd.read_excel(dir_data_frame)
    df = df[df[pred_column].notna()]  # Clean the rows including NaN values in the column that we want to predict

    # Select df rows by train, test, val split
    train_ids_df = df[df["Patient ID"].isin(train_ids)][:5]#[:100]#.sample(30)
    val_ids_df = df[df["Patient ID"].isin(val_ids)][:2]#[:50]#.sample(20)
    test_ids_df = df[df["Patient ID"].isin(test_ids)][:2]#[:50]#.sample(20)

    # Read the excel including the images paths and their tissue percentage
    train_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "train_patches_class_perc_0_tp.csv")
    val_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "val_patches_class_perc_0_tp.csv")
    test_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "test_patches_class_perc_0_tp.csv")

    # Update 26/02/2023: As now we have a different distribution for the train, val and test IDs we should combine the class_perc_patches_paths_df to have all of them in one
    combined_class_perc_patches_paths_df = pd.concat(
        [train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df])

    # Choose patches directories fullWSI, filled or not filled
    if regions_filled == "fullWSIs_TP_0":
        dir_images = '../data/patches_' + str(patch_size) + '_fullWSIs_0/'

    #############################################################################
    # Set train and validation generators depending on the MIL aggregation type #
    #############################################################################

    # Create datasets
    print(" Loading tranining data...")
    dataset_train = MILDataset_w_class_perc(dir_images=dir_images, data_frame=train_ids_df, classes=classes,
                                            pred_column=pred_column, pred_mode=pred_mode,
                                            magnification_level=magnification_level,
                                            bag_id='Patient ID', input_shape=input_shape,
                                            data_augmentation=data_augmentation,
                                            stain_normalization=stain_normalization,
                                            images_on_ram=images_on_ram,
                                            include_background=include_background,
                                            class_perc_data_frame=combined_class_perc_patches_paths_df,
                                            tissue_percentages_max=tissue_percentages_max)
    if balanced_train_datagen:
        data_generator_train = MILDataGenerator_balanced_good(dataset_train, batch_size=1,
                                                              shuffle=not (ordered),
                                                              max_instances=max_instances, num_workers=0,
                                                              pred_column=pred_column, pred_mode=pred_mode,
                                                              images_on_ram=images_on_ram)
    else:
        data_generator_train = MILDataGenerator(dataset_train, batch_size=1, shuffle=not (ordered),
                                                max_instances=max_instances, num_workers=0,
                                                pred_column=pred_column,
                                                images_on_ram=images_on_ram)
    print(" Loading validation data...")
    dataset_val = MILDataset_w_class_perc(dir_images=dir_images, data_frame=val_ids_df, classes=classes,
                                          pred_column=pred_column, pred_mode=pred_mode,
                                          magnification_level=magnification_level,
                                          bag_id='Patient ID', input_shape=input_shape,
                                          data_augmentation=False, stain_normalization=stain_normalization,
                                          images_on_ram=images_on_ram, include_background=include_background,
                                          class_perc_data_frame=combined_class_perc_patches_paths_df,
                                          tissue_percentages_max=tissue_percentages_max)

    data_generator_val = MILDataGenerator(dataset_val, batch_size=1, shuffle=False,
                                          max_instances=max_instances, num_workers=0, pred_column=pred_column,
                                          pred_mode=pred_mode,
                                          images_on_ram=images_on_ram)

    print(" Loading testing data...")
    dataset_test = MILDataset_w_class_perc(dir_images=dir_images, data_frame=test_ids_df, classes=classes,
                                           pred_column=pred_column, pred_mode=pred_mode,
                                           magnification_level=magnification_level,
                                           bag_id='Patient ID', input_shape=input_shape,
                                           data_augmentation=False, stain_normalization=stain_normalization,
                                           images_on_ram=images_on_ram, include_background=include_background,
                                           class_perc_data_frame=combined_class_perc_patches_paths_df,
                                           tissue_percentages_max=tissue_percentages_max)

    data_generator_test = MILDataGenerator(dataset_test, batch_size=1, shuffle=False,
                                           max_instances=max_instances,
                                           num_workers=0, pred_column=pred_column, pred_mode=pred_mode,
                                           images_on_ram=images_on_ram)

    # Return datasets
    return dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test

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
        mlflow_experiment_name = args.experiment_name
        mlflow_run_name = args_train.run_name

        mlflow.set_tracking_uri(mlruns_folder)
        experiment = mlflow.set_experiment(mlflow_experiment_name)
        mlflow.start_run(run_name=mlflow_run_name)

        # Log Parameters
        mlflow.log_param("patch_size", args_train.patch_size)
        mlflow.log_param("data_augmentation", args_train.data_augmentation)
        mlflow.log_param("stain_normalization", args_train.stain_normalization)
        mlflow.log_param("regions_filled", args_train.regions_filled)
        mlflow.log_param("max_instances", args_train.max_instances)
        mlflow.log_param("ordered", args_train.ordered)
        mlflow.log_param("pred_mode", args_train.pred_mode)
        mlflow.log_param("magnification_level", args_train.magnification_level)
        mlflow.log_param("pred_mode", args_train.pred_mode)


        mlflow.log_param("nn_backbone", args_train.network_backbone)
        mlflow.log_param("pretrained", args.pretrained)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args_train.lr)
        mlflow.log_param("criterion", args.criterion)
        mlflow.log_param("aggregation", args_train.aggregation)
        # mlflow.log_param("norm_c_weights", args.class_weights_norm)
        mlflow.log_param("balanced_train_datagen", args_train.balanced_train_datagen)
        mlflow.log_param("tissue_percentages_max", args_train.tissue_percentages_max)

        mlflow.log_param("optimizer", args_train.optimizer_type)
        mlflow.log_param("optim_weight_decay", args.optimizer_weight_decay)
        mlflow.log_param("loss_function", args.loss_function)
        mlflow.log_param("freeze_bb_weights", args.freeze_bb_weights)

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

    # Prepare dataset for each CV fold
    # TODO: Add relative path CV folds
    dir_cv_dataset_splitting_path = args.dir_cv_dataset_splitting_path

    files_folds_splitting_ids = os.listdir(dir_cv_dataset_splitting_path)

    folds_ids = np.unique(np.array([fold.split("_")[1] for fold in files_folds_splitting_ids])) # find the unique folds ids

    # Initialize variables

    for fold_id in folds_ids:
        print(f"[CV FOLD {fold_id}]")

        # # Retrieve dataset split for each fold
        # dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test = load_CV_fold_dataset(fold_id, magnification_level, pred_column, pred_mode, classes, regions_filled, dir_cv_dataset_splitting_path,
        #                          ordered, patch_size, max_instances, data_augmentation, stain_normalization,
        #                          images_on_ram, include_background, balanced_train_datagen, tissue_percentages_max)

        # Launch set of experiments
        network_backbones = args.network_backbones
        aggregations = args.aggregations
        learning_rates = args.learning_rates
        optimizers_types = args.optimizers_types

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

            for network_backbone in network_backbones:
                for aggregation in aggregations:
                    for learning_rate in learning_rates:
                        for optimizer_type in optimizers_types:
                            # TODO: fix run_name generation to list comprehension
                            if "Patch_GCN" in aggregation:
                                run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(args.aggregation) + "_ML_" + str(
                                    args.magnification_level) + "_NN_bb_" + str(args.network_backbone) + "_FBB_" + str(
                                    args.freeze_bb_weights) + "_PS_" + str(args.patch_size) + "_DA_" + str(
                                    args.data_augmentation) + "_SN_" + str(
                                    args.stain_normalization) + "_L_" + args.criterion + "_E_" + str(args.epochs) + "_LR_" + str(
                                    args.lr).replace(
                                    ".", "") + "_Order_" + str(args.ordered) + "_Optim_" + str(args.optimizer_type) + "_N_" + str(
                                    args.max_instances) + "_BDG_" + str(args.balanced_train_datagen) + "_OWD_" + str(
                                    args.optimizer_weight_decay) + "_TP_" + str(args.tissue_percentages_max) + "_KNN_" + str(
                                    args.knn) + "_GCNL_" + str(args.num_gcn_layers) + "_EF_" + str(
                                    args.include_edge_features) + "_CVFold_" + str(fold_id)
                            else:
                                run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(aggregation) + "_ML_" + str(
                                    args.magnification_level) + "_NN_bb_" + str(network_backbone) + "_FBB_" + str(
                                    args.freeze_bb_weights) + "_PS_" + str(args.patch_size) + "_DA_" + str(
                                    args.data_augmentation) + "_SN_" + str(
                                    args.stain_normalization) + "_L_" + args.criterion + "_E_" + str(args.epochs) + "_LR_" + str(
                                    learning_rate).replace(
                                    ".", "") + "_Order_" + str(args.ordered) + "_Optim_" + str(optimizer_type) + "_N_" + str(
                                    args.max_instances) + "_BDG_" + str(args.balanced_train_datagen) + "_OWD_" + str(
                                    args.optimizer_weight_decay) + "_TP_" + str(args.tissue_percentages_max) + "_CVFold_" + str(
                                    fold_id)

                            # Prepare output directories
                            #TODO: Modify output directories vbased on relative general path and where_exec
                            dir_results = '../data/results/'
                            dir_out_gnrl = os.path.join(dir_results, args.experiment_name)
                            dir_out_main = os.path.join(dir_results, args.experiment_name, run_name.split("_CVFold_")[0])
                            # dir_out = os.path.join(dir_out_main, run_name)
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
                            parser_train.add_argument("--balanced_train_datagen", default=args.balanced_train_datagen,
                                                type=bool)
                            parser_train.add_argument("--tissue_percentages_max", default=args.tissue_percentages_max,
                                                type=str)
                            parser_train.add_argument("--dir_out", default=dir_out,
                                                type=str)
                            parser_train.add_argument("--pred_column", default=args.pred_column,
                                                type=str)
                            parser_train.add_argument("--magnification_level", default=args.magnification_level,
                                                type=str)


                            args_train = parser_train.parse_args()
                            train_exp(args_train,dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test)

        print("holu")
    print("holu")

##########################
# CREATE ARGUMENT PARSER #
##########################
parser = argparse.ArgumentParser()

# TODO: Put dir_cv_dataset_splitting_path inside CV folds (relative path from a general_data_dir)
parser.add_argument("--dir_cv_dataset_splitting_path", default="../data/BCNB/new_CV_folds_BCNB",
                    type=str)  # Directory where the folds of the new splits are stored

# Directories.
parser.add_argument('--gnrl_data_dir', type=str, default='.')
parser.add_argument('--where_exec', type=str, default="dgx_gpu", help="slurm or local, dgx_gpu")
parser.add_argument('--training_type', type=str, default="CV" , help='CV or Full Dataset')
parser.add_argument('--dataset', type=str, default="BCNB", help='BCNB or CLARIFY_DB')

#MLFlow configuration
parser.add_argument("--experiment_name", default="[29_09_2023] BB MolSub 10x BCNB Final", type=str,  help='Name for experiment in MLFlow')

# Training configuration
parser.add_argument("--train_test_mode", default="train", type=str, help="Select train, test, test_allmymodels")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for training")
parser.add_argument("--learning_rates", default=[0.002], type=list, help="Choose (1 or more): 0.002, 0.0001")
parser.add_argument("--loss_function", default="cross_entropy", type=str, help="Loss function: cross_entropy, kll")
parser.add_argument("--optimizers_types", default=["sgd"], type=list, help="Choose (1 or more): sgd, adam, lookahead_radam")
parser.add_argument("--optimizer_weight_decay", default=0, type=float)
parser.add_argument("--freeze_bb_weights", default=False, type=bool, help="Freeze feature extractor weights. False: retrain True: transfer learning")
parser.add_argument("--pretrained", default=True, type=bool, help="False: retrain from scratch True: Use pretrained feature extractor on ImageNet.")
parser.add_argument("--criterion", default='auc', type=str, help="Metric to keep the best model: 'auc', 'f1'")
parser.add_argument("--mode", default="embedding", type=str)  # embedding,embedding_GNN
parser.add_argument("--alpha_ce", default=1., type=float)

# Model configuration
parser.add_argument("--pred_modes", default=["OTHERvsTNBC"], type=list, help="Choose (1 or more) between: 'LUMINALAvsLAUMINALBvsHER2vsTNBC', 'LUMINALSvsHER2vsTNBC', 'OTHERvsTNBC'")
parser.add_argument("--network_backbones", default=['vgg16'], type=list, help="Choose (1 or more) betwwen: 'vgg16', 'resnet50'")
parser.add_argument("--magnification_level", default="5x", type=str, help="5x, 10x, 20x")
parser.add_argument("--aggregations", default=["mean"], type=list, help="Choose (1 or more) between: mean', 'max' 'attention', 'TransMIL'")

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
