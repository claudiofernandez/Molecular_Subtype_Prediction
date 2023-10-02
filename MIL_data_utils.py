from MIL_utils import *
from MIL_data import *

#TODO: delete class selection from main, move here
#TODO: delete include_background variable
def load_CV_fold_dataset(fold_id, magnification_level, pred_column, pred_mode, classes, regions_filled, dir_cv_dataset_splitting_path, ordered, patch_size, max_instances, data_augmentation, stain_normalization, images_on_ram, include_background, balanced_train_datagen, tissue_percentages_max):

    ##############################################
    # Set directories and parameters for the data #
    ##############################################

    # Directories
    # TODO: change all directories to slurm/local (change args)
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

def load_BCNB_full_dataset(gnrl_data_dir, magnification_level, pred_column, pred_mode, regions_filled, ordered, patch_size, max_instances, data_augmentation, stain_normalization, images_on_ram, include_background, balanced_train_datagen, tissue_percentages_max):

    # Set classes depending on training task
    if pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
        classes = ['Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative']
    elif pred_mode == "LUMINALSvsHER2vsTNBC":
        classes = ['Luminal', 'HER2(+)', 'Triple negative']
    elif pred_mode == "OTHERvsTNBC":
        classes = ['Other', 'Triple negative']

    ##############################################
    # Set directories and parameters for the data #
    ##############################################

    # TODO: change all directories to slurm/local
    # Directories
    dir_data_frame = os.path.join(gnrl_data_dir, "patient-clinical-data.xlsx")
    dir_excels_class_perc = os.path.join(gnrl_data_dir, "patches_paths_class_perc")
    dir_dataset_splitting_path = os.path.join(gnrl_data_dir, "dataset-splitting")

    # Set input shape of the images depending on the chosen magnification
    if magnification_level == "20x":
        input_shape = (3, 512, 512)
    elif magnification_level == "10x":
        input_shape = (3, 256, 256)
    elif magnification_level == "5x":
        input_shape = (3, 128, 128)

    print("Magnification level: " + str(magnification_level) + " using input shape: " + str(input_shape))

    # Retrieve original dataset split
    train_file = "train_id.txt"
    val_file = "val_id.txt"
    test_file = "test_id.txt"

    with open(os.path.join(dir_dataset_splitting_path, train_file), "r") as f:
        train_ids = f.read().splitlines()

    with open(os.path.join(dir_dataset_splitting_path, val_file), "r") as f:
        val_ids = f.read().splitlines()

    with open(os.path.join(dir_dataset_splitting_path, test_file), "r") as f:
        test_ids = f.read().splitlines()


    # Convert IDs from strings to ints
    train_ids = [int(id) for id in train_ids]
    val_ids = [int(id) for id in val_ids]
    test_ids = [int(id) for id in test_ids]

    # Read GT DataFrame
    df = pd.read_excel(dir_data_frame)
    df = df[df[pred_column].notna()]  # Clean the rows including NaN values in the column that we want to predict

    # Select df rows by train, test, val split
    train_ids_df = df[df["Patient ID"].isin(train_ids)][:10]#[:100]#.sample(30)
    val_ids_df = df[df["Patient ID"].isin(val_ids)][:5]#[:50]#.sample(20)
    test_ids_df = df[df["Patient ID"].isin(test_ids)][:5]#[:50]#.sample(20)

    # Read the excel including the images paths and their tissue percentage
    train_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc, "train_patches_class_perc_0_tp.csv"))
    val_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc, "val_patches_class_perc_0_tp.csv"))
    test_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc, "test_patches_class_perc_0_tp.csv"))

    # Update 26/02/2023: As now we have a different distribution for the train, val and test IDs we should combine the class_perc_patches_paths_df to have all of them in one
    combined_class_perc_patches_paths_df = pd.concat(
        [train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df])

    # Choose patches directories fullWSI, filled or not filled
    if regions_filled == "fullWSIs_TP_0":
        dir_images = os.path.join(gnrl_data_dir, "preprocessing_results_bien/patches_" + str(patch_size) + "_fullWSIs_0")

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
    return classes, dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test

