# Imports
from MIL_utils import *
from MIL_data import *
from MIL_models import *
from MIL_trainer import *

def main_cv(args):
    # Effective training
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    CUDA_LAUNCH_BLOCKING = 1

    ##############################################
    # Set directories and parameters for the data #
    ##############################################

    # Directories
    dir_data_frame = '../data/patient-clinical-data.xlsx'
    dir_results = '../data/results/'
    dir_excels_class_perc = "../data/patches_paths_class_perc/"

    # MIL Parameters
    bag_id = 'Patient ID'
    pred_column = "Molecular subtype"
    regions_filled = "fullWSIs_TP_0"
    # pred_mode = "LUMINALAvsLAUMINALBvsHER2vsTNBC"
    # pred_mode = "LUMINALSvsHER2vsTNBC"
    pred_mode = "OTHERvsTNBC"

    ordered = True
    patch_size = 512
    data_augmentation = "non-spatial"
    max_instances = 100
    stain_normalization = False
    images_on_ram = True
    include_background = False
    balanced_train_datagen = True
    tissue_percentages_max = "O_0.4-T_1-S_1-I_1-N_1"
    # magnification_level = "5x"  # [BCNB] 10x, 20x
    magnification_level = args.magnification_level
    #aggregation = "Patch_GCN_offline"  # change if we don't want Patch_GCN aggr
    aggregation = args.aggregation  # change if we don't want Patch_GCN aggr

    # Patch-GCN Parameters
    if "Patch_GCN" in aggregation:
        #knn = 25  # Number of KNN neighbours - 1
        #graph_bb = "vgg16"  # Backbone used for extracting the features from the patches
        #include_edge_features = True  # Include or not edge features in the graphs

        knn = args.node_knn # Number of KNN neighbours - 1
        graph_bb = args.node_feature_extractor  # Backbone used for extracting the features from the patches
        include_edge_features = args.include_edge_features  # Include or not edge features in the graphs

        graphs_on_ram = True  # Preload graphs on memory
        if not include_edge_features:
            graphs_dir = '../data/results_graphs/graphs_k_' + str(knn) + "_bb_" + str(graph_bb) + "_" + str(patch_size)
        else:
            graphs_dir = '../data/results_graphs/graphs_k_' + str(knn) + "_bb_" + str(graph_bb) + "_" + str(
                patch_size) + "_w_edge_features"

    # Set classes depending on training task
    if pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
        classes = ['Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative']
    elif pred_mode == "LUMINALSvsHER2vsTNBC":
        classes = ['Luminal', 'HER2(+)', 'Triple negative']
    elif pred_mode == "OTHERvsTNBC":
        classes = ['Other', 'Triple negative']

    # Set input shape of the images depending on the chosen magnification
    if magnification_level == "20x":
        input_shape = (3, 512, 512)
    elif magnification_level == "10x":
        input_shape = (3, 256, 256)
    elif magnification_level == "5x":
        input_shape = (3, 128, 128)

    print("Magnification level: " + str(magnification_level) + " using input shape: " + str(input_shape))


    dir_cv_dataset_splitting_path = args.dir_cv_dataset_splitting_path

    files_folds_splitting_ids = os.listdir(dir_cv_dataset_splitting_path)

    folds_ids = np.unique(np.array([fold.split("_")[1] for fold in files_folds_splitting_ids])) # find the unique folds ids

    # Initialize variables

    for fold_id in folds_ids:
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
        train_ids_df = df[df["Patient ID"].isin(train_ids)]#[:100]#.sample(30)
        val_ids_df = df[df["Patient ID"].isin(val_ids)]#[:50]#.sample(20)
        test_ids_df = df[df["Patient ID"].isin(test_ids)]#[:50]#.sample(20)

        # Read the excel including the images paths and their tissue percentage
        train_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "train_patches_class_perc_0_tp.csv")
        val_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "val_patches_class_perc_0_tp.csv")
        test_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "test_patches_class_perc_0_tp.csv")

        # Update 26/02/2023: As now we have a different distribution for the train, val and test IDs we should combine the class_perc_patches_paths_df to have all of them in one
        combined_class_perc_patches_paths_df = pd.concat([train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df])

        # Choose patches directories fullWSI, filled or not filled
        if regions_filled == "fullWSIs_TP_0":
            dir_images = '../data/patches_' + str(patch_size) + '_fullWSIs_0/'

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

        elif aggregation == "Patch_GCN_online":  # Datasets and Datagens for online graphs
            print(" Loading training data...")
            dataset_train = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=train_ids_df,
                                                           classes=classes,
                                                           pred_column=pred_column, pred_mode=pred_mode,
                                                           magnification_level=magnification_level,
                                                           bag_id='Patient ID', input_shape=input_shape,
                                                           data_augmentation=data_augmentation,
                                                           stain_normalization=stain_normalization,
                                                           images_on_ram=images_on_ram,
                                                           include_background=include_background,
                                                           class_perc_data_frame=train_class_perc_patches_paths_df,
                                                           tissue_percentages_max=tissue_percentages_max)
            if balanced_train_datagen:
                data_generator_train = MILDataGenerator_balanced_good_coords(dataset_train, batch_size=1,
                                                                             shuffle=not (ordered),
                                                                             max_instances=max_instances, num_workers=0,
                                                                             pred_column=pred_column,
                                                                             pred_mode=pred_mode,
                                                                             images_on_ram=images_on_ram)
            else:
                data_generator_train = MILDataGenerator_coords(dataset_train, batch_size=1, shuffle=not (ordered),
                                                               max_instances=max_instances, num_workers=0,
                                                               pred_column=pred_column,
                                                               images_on_ram=images_on_ram)

            print(" Loading validation data...")
            dataset_val = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=val_ids_df, classes=classes,
                                                         pred_column=pred_column, pred_mode=pred_mode,
                                                         magnification_level=magnification_level,
                                                         bag_id='Patient ID', input_shape=input_shape,
                                                         data_augmentation=False,
                                                         stain_normalization=stain_normalization,
                                                         images_on_ram=images_on_ram,
                                                         include_background=include_background,
                                                         class_perc_data_frame=val_class_perc_patches_paths_df,
                                                         tissue_percentages_max=tissue_percentages_max)
            data_generator_val = MILDataGenerator_coords(dataset_val, batch_size=1, shuffle=False,
                                                         max_instances=max_instances, num_workers=0,
                                                         pred_column=pred_column,
                                                         pred_mode=pred_mode,
                                                         images_on_ram=images_on_ram)

            print(" Loading testing data...")
            dataset_test = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=test_ids_df,
                                                          classes=classes,
                                                          pred_column=pred_column, pred_mode=pred_mode,
                                                          magnification_level=magnification_level,
                                                          bag_id='Patient ID', input_shape=input_shape,
                                                          data_augmentation=False,
                                                          stain_normalization=stain_normalization,
                                                          images_on_ram=images_on_ram,
                                                          include_background=include_background,
                                                          class_perc_data_frame=test_class_perc_patches_paths_df,
                                                          tissue_percentages_max=tissue_percentages_max)
            data_generator_test = MILDataGenerator_coords(dataset_test, batch_size=1, shuffle=False,
                                                          max_instances=max_instances,
                                                          num_workers=0, pred_column=pred_column, pred_mode=pred_mode,
                                                          images_on_ram=images_on_ram)  # Datasets and Datagens for offline graphs

        elif aggregation == "max" or aggregation == "mean" or aggregation == "attention":
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

        if "Patch_GCN" in aggregation:
            run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(args.aggregation) + "_ML_" + str(
                magnification_level) + "_NN_bb_" + str(args.network_backbone) + "_FBB_" + str(
                args.freeze_bb_weights) + "_PS_" + str(patch_size) + "_DA_" + str(data_augmentation) + "_SN_" + str(
                stain_normalization) + "_L_" + args.criterion + "_E_" + str(args.epochs) + "_LR_" + str(
                args.lr).replace(
                ".", "") + "_Order_" + str(ordered) + "_Optim_" + str(args.optimizer_type) + "_N_" + str(
                max_instances) + "_BDG_" + str(balanced_train_datagen) + "_OWD_" + str(
                args.optimizer_weight_decay) + "_TP_" + str(tissue_percentages_max) + "_KNN_" + str(
                knn) + "_GCNL_" + str(args.num_gcn_layers) + "_EF_" + str(include_edge_features) + "_CVFold_" + str(fold_id)
        else:
            run_name = "PM_" + str(pred_mode) + "_AGGR_" + str(args.aggregation) + "_ML_" + str(
                magnification_level) + "_NN_bb_" + str(args.network_backbone) + "_FBB_" + str(
                args.freeze_bb_weights) + "_PS_" + str(patch_size) + "_DA_" + str(data_augmentation) + "_SN_" + str(
                stain_normalization) + "_L_" + args.criterion + "_E_" + str(args.epochs) + "_LR_" + str(
                args.lr).replace(
                ".", "") + "_Order_" + str(ordered) + "_Optim_" + str(args.optimizer_type) + "_N_" + str(
                max_instances) + "_BDG_" + str(balanced_train_datagen) + "_OWD_" + str(
                args.optimizer_weight_decay) + "_TP_" + str(tissue_percentages_max) + "_CVFold_" + str(fold_id)

        # Prepare output directories
        dir_out_gnrl = os.path.join(dir_results, args.experiment_name)
        dir_out_main = os.path.join(dir_results, args.experiment_name, run_name.split("_CVFold_")[0])
        #dir_out = os.path.join(dir_out_main, run_name)
        dir_out = os.path.join(dir_out_main, "CVFold_" + str(fold_id))
        if not os.path.isdir(dir_results):
            os.mkdir(dir_results)
        if not os.path.isdir(dir_out_gnrl):
            os.mkdir(dir_out_gnrl)
        if not os.path.isdir(dir_out_main):
            os.mkdir(dir_out_main)
        if not os.path.isdir(dir_out):
            os.mkdir(dir_out)

        #################################################
        # Set Model Architecture for current experiment #
        #################################################

        # Prepare GPU Training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: {}'.format(device))
        if torch.cuda.is_available():
            print('GPU Model: {}'.format(torch.cuda.get_device_name(0)))

        # Set network architecture depending on aggregation type

        if args.aggregation == "TransMIL":
            network = TransMILArchitecture(n_classes=len(classes), transmil_bb=args.network_backbone,
                                           pretrained_bb=True,
                                           freeze_bb_weights=args.freeze_bb_weights)

        if args.aggregation == "TransMIL_Pablo":
            network = TransMILArchitecture_Pablo(n_classes=len(classes), transmil_bb=args.network_backbone,
                                                 pretrained_bb=True,
                                                 freeze_bb_weights=args.freeze_bb_weights)

        elif args.aggregation == "max" or args.aggregation == "mean" or args.aggregation == "attention":
            network = MILArchitecture(classes=classes, pretrained=args.pretrained, mode=args.mode,
                                      aggregation=args.aggregation, freeze_bb_weights=args.freeze_bb_weights,
                                      backbone=args.network_backbone, include_background=include_background)

        elif "Patch_GCN" in args.aggregation:
            if "resnet50" in args.network_backbone:
                model_dict = {"dropout": args.drop_out, 'n_classes': len(classes), "num_layers": args.num_gcn_layers,
                              "num_features": 1024}
                network = PatchGCN(**model_dict)
            elif "vgg16" in args.network_backbone:
                model_dict = {"dropout": args.drop_out, 'n_classes': len(classes), "num_layers": args.num_gcn_layers,
                              "num_features": 512}
                network = PatchGCN(**model_dict)

        print(network)

        ##################################################
        # Set Training Parameters for current experiment #
        ##################################################

        if args.train_test_mode == "train":
            # MlFlow Parameters
            mlruns_folder = "./mlruns"
            mlflow_experiment_name = args.experiment_name
            mlflow_run_name = run_name

            mlflow.set_tracking_uri(mlruns_folder)
            experiment = mlflow.set_experiment(mlflow_experiment_name)
            mlflow.start_run(run_name=mlflow_run_name)

            # Log Parameters
            mlflow.log_param("patch_size", patch_size)
            mlflow.log_param("data_augmentation", data_augmentation)
            mlflow.log_param("stain_normalization", stain_normalization)
            mlflow.log_param("regions_filled", regions_filled)
            mlflow.log_param("max_instances", max_instances)
            mlflow.log_param("ordered", ordered)
            mlflow.log_param("pred_mode", pred_mode)
            mlflow.log_param("magnification_level", magnification_level)

            mlflow.log_param("nn_backbone", args.network_backbone)
            mlflow.log_param("pretrained", args.pretrained)
            mlflow.log_param("epochs", args.epochs)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("criterion", args.criterion)
            mlflow.log_param("aggregation", args.aggregation)
            mlflow.log_param("norm_c_weights", args.class_weights_norm)
            mlflow.log_param("balanced_train_datagen", balanced_train_datagen)
            mlflow.log_param("tissue_percentages_max", tissue_percentages_max)

            mlflow.log_param("optimizer", args.optimizer_type)
            mlflow.log_param("optim_weight_decay", args.optimizer_weight_decay)
            mlflow.log_param("loss_function", args.loss_function)
            mlflow.log_param("freeze_bb_weights", args.freeze_bb_weights)

            # Patch GCN
            if "Patch_GCN" in aggregation:
                mlflow.log_param("num_gcn_layers", args.num_gcn_layers)
                mlflow.log_param("knn", knn)
                mlflow.log_param("include_edge_features", include_edge_features)
                mlflow.log_param("edge_agg", args.edge_agg)

            # Compute Class Weights for Imbalanced Dataset
            if args.class_weights_norm:
                class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                                                classes=dataset_train.classes,
                                                                                y=dataset_train.y)
            else:
                class_weights = np.ones(len(classes))

            ##################
            # Start Training #
            ##################
            optimizer_type = args.optimizer_type
            if args.aggregation == "max" or args.aggregation == "mean" or args.aggregation == "attention" or args.aggregation == "TransMIL" or args.aggregation == "TransMIL_Pablo":
                trainer = TransMIL_trainer(dir_out=dir_out, network=network, model_save_name=run_name,
                                           aggregation=args.aggregation,
                                           lr=args.lr,
                                           alpha_ce=args.alpha_ce, id=id,
                                           early_stopping=args.early_stopping, scheduler=args.scheduler,
                                           virtual_batch_size=args.virtual_batch_size,
                                           criterion=args.criterion,
                                           class_weights=class_weights,
                                           optimizer_type=optimizer_type,
                                           optimizer_weight_decay=args.optimizer_weight_decay,
                                           mlflow_run_name=mlflow_run_name)

                trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                              test_generator=data_generator_test, epochs=args.epochs, model_save_name=run_name,
                              pred_column=pred_column, pred_mode=pred_mode, loss_function=args.loss_function)

            elif args.aggregation == "Patch_GCN_online":
                trainer = Patch_GCN_online_trainer(dir_out=dir_out, network=network, model_save_name=run_name,
                                                   lr=args.lr, aggregation=args.aggregation,
                                                   alpha_ce=args.alpha_ce, id=id,
                                                   early_stopping=args.early_stopping, scheduler=args.scheduler,
                                                   virtual_batch_size=args.virtual_batch_size,
                                                   criterion=args.criterion,
                                                   class_weights=class_weights,
                                                   optimizer_type=optimizer_type,
                                                   optimizer_weight_decay=args.optimizer_weight_decay,
                                                   mlflow_run_name=mlflow_run_name,
                                                   knn=knn,
                                                   node_feature_extractor=args.network_backbone)

                trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                              test_generator=data_generator_test, epochs=args.epochs, model_save_name=run_name,
                              pred_column=pred_column, pred_mode=pred_mode, loss_function=args.loss_function)

            elif args.aggregation == "Patch_GCN_offline":
                trainer = Patch_GCN_offline_trainer(dir_out=dir_out, network=network, model_save_name=run_name,
                                                    lr=args.lr, aggregation=args.aggregation,
                                                    alpha_ce=args.alpha_ce, id=id,
                                                    early_stopping=args.early_stopping, scheduler=args.scheduler,
                                                    virtual_batch_size=args.virtual_batch_size,
                                                    criterion=args.criterion,
                                                    class_weights=class_weights,
                                                    optimizer_type=optimizer_type,
                                                    optimizer_weight_decay=args.optimizer_weight_decay,
                                                    mlflow_run_name=mlflow_run_name,
                                                    knn=knn,
                                                    node_feature_extractor=args.network_backbone)

                trainer.train(train_generator=data_generator_train, val_generator=data_generator_val,
                              test_generator=data_generator_test, epochs=args.epochs, model_save_name=run_name,
                              pred_column=pred_column, pred_mode=pred_mode, loss_function=args.loss_function)

        print("Hola")

    print("hola")

#########################################
# Common parameters for current experiment #
#########################################
#knn = 5
#node_feature_extractor = 'vgg16'
#experiment_name = "MolSub 512 Aggr Patch-GCN Offline 4CLF CV KNN=" + str(knn) + " BB=" + str(node_feature_extractor)
experiment_name = "[06_07_2023] BB MolSub 2CLF 10x CV"
learning_rates = [0.002, 0.0001, 0.00001]#[0.002, 0.0001, 0.00001, 0.000001] #[0.000001]
network_backbones = ['resnet50','vgg16'] # ['vgg16', 'resnet50']
aggregations = ['max', 'mean', 'attention'] #['max'] # ['mean', 'max']
magnification_level = "10x" # "10x"

#############################
# Experiments for Mean Aggr #
#############################
#########################
# CREATE ARGUMENT PARSER #
#########################

for network_backbone in network_backbones:
    for aggregation in aggregations: 
        for learning_rate in learning_rates:
            parser = argparse.ArgumentParser()
        
            # Model parameters
            parser.add_argument("--dir_cv_dataset_splitting_path", default="../data/new_CV_folds_BCNB", type=str)  # Directory where the folds of the new splits are stored
            parser.add_argument("--train_test_mode", default="train", type=str)  # Select train, test, test_allmymodels
            parser.add_argument("--aggregation", default=aggregation, type=str)  # max, mean, TransMIL, TransMIL_pablo, Patch_GCN_online, Patch_GCN_offline
            parser.add_argument("--epochs", default=100, type=int) # Number of epochs for training
            parser.add_argument("--lr", default=learning_rate, type=float) # Learning rate
            parser.add_argument("--magnification_level", default=magnification_level, type=str)  # # [BCNB] 5x, 10x, 20x
            parser.add_argument("--loss_function", default="cross_entropy", type=str)  # cross_entropy, kll
            parser.add_argument("--optimizer_type", default='sgd', type=str)  # sgd, adam, lookahead_radam
            parser.add_argument("--optimizer_weight_decay", default=0, type=float)
            parser.add_argument("--class_weights_norm", default=False, type=bool)  # Apply c_weights_norm to CrossEntropyLoss
            parser.add_argument("--freeze_bb_weights", default=False, type=bool) # Freeze feature extractor weights. False: retrain True: transfer learning
            parser.add_argument("--pretrained", default=True, type=bool) # Use pretrained feature extractor on ImageNet
            parser.add_argument("--network_backbone", default=network_backbone, type=str) #vgg16, resnet50, vgg16_512
            parser.add_argument("--criterion", default='auc', type=str)  # auc
            parser.add_argument("--mode", default="embedding", type=str)  # embedding,embedding_GNN
            parser.add_argument("--alpha_ce", default=1., type=float)
    
            # # Patch GCN Parameters
            # parser.add_argument('--include_edge_features',  default=False, help='Include edge_features (euclidean dist) in  the graph')
            # parser.add_argument('--num_gcn_layers',  type=int, default=5, help='# of GCN layers to use.')
            # parser.add_argument('--node_knn',  type=int, default=knn, help='# of K nearest neighbours for graph creation.')
            # parser.add_argument('--node_feature_extractor', type=str, default=node_feature_extractor, help="Feature extractor that will be used for creating the nodes of the graph.") # resnet50_3blocks_1024
            # parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.") # It is possible to use spatial or latent edge aggregation
            # parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
            # parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
            #
            # Global label to predict
            parser.add_argument("--experiment_name", default=experiment_name, type=str)
    
            # Miscellaneous
            parser.add_argument("--early_stopping", default=True, type=bool)
            parser.add_argument("--scheduler", default=True, type=bool)
            parser.add_argument("--virtual_batch_size", default=1, type=int)
    
            args = parser.parse_args()
            main_cv(args)

