from MIL_trainer import *
from MIL_models import *
from MIL_data import *

class WSI2Graph_Generator():

    def __init__(self, pred_column, bag_id, feature_extractor_name, feature_extractor_path, model_weights_filename, knn_list, magnification_level, include_edge_features, edges_type,
                 dir_data_frame, dir_dataset_splitting, dir_excels_class_perc, dir_results):
        super(WSI2Graph_Generator, self).__init__()

        # Datasets
        self.pred_column = pred_column
        self.bag_id = bag_id

        # Graph parameters
        self.feature_extractor_name = feature_extractor_name
        self.feature_extractor_path = feature_extractor_path
        self.model_weights_filename = model_weights_filename
        self.knn_list = knn_list
        self.magnification_level =magnification_level # [BCNB] 5x, 10x, 20x
        self.include_edge_features =include_edge_features # Include edge_features based on the euclidean distance between the coordinates of each node
        self.edges_type = edges_type # "latent" based on feature distance or "spatial" based on coordinates distance

        # Directories
        self.dir_data_frame = dir_data_frame
        self.dir_dataset_splitting = dir_dataset_splitting
        self.dir_excels_class_perc = dir_excels_class_perc
        self.dir_results = dir_results
        #TODO: Add graph name and save fur definition based on the parameters
        self.graph_name = "tbd"
        self.dir_results_save_graph = os.path.join(dir_results, self.graph_name)

        # Create dirs for saving resulting graphs
        os.makedirs(self.dir_results, exist_ok=True)
        os.makedirs(self.dir_results_save_graph, exist_ok=True)

        # Obtain train, val and test IDs
        self.train_ids, self.val_ids, self.test_ids = self.get_ids(dir_dataset_splitting=self.dir_dataset_splitting)

        # Obtain GT dfs
        self.train_ids_df, self.val_ids_df, self.test_ids_df = self.get_gt_dfs(train_ids=self.train_ids,
                                                                                               val_ids=self.val_ids,
                                                                                               test_ids=self.test_ids,
                                                                                               dir_data_frame= self.dir_data_frame,
                                                                                               pred_column=self.pred_column)
        # Obtain the train, val, and test dataframes conatining the path paths of the split
        self.train_paths_df, self.val_paths_df, self.test_paths_df = self.get_patches_paths_dfs(dir_excels_class_perc=self.dir_excels_class_perc)

        # Obtain datasets


        # self.dataset_train, self.data_generator_train, self.dataset_val, self.data_generator_val, \
        # self.dataset_test, self.data_generator_test = self.load_MIL_datasets_coordinates()

    def get_ids(self, dir_dataset_splitting):
        # Dataset split
        train_ids, test_ids, val_ids = get_train_test_val_ids(dir_dataset_splitting) # function from MIL_data

        return train_ids, test_ids, val_ids

    def get_gt_dfs(self, train_ids, test_ids, val_ids, dir_data_frame, pred_column):

        # Read GT DataFrame
        df = pd.read_excel(dir_data_frame)
        df = df[df[pred_column].notna()]  # Clean the rows including NaN values in the column that we want to predict

        # Select df rows by train, test, val split
        train_ids_df = df[df["Patient ID"].isin(train_ids)]  # .sample(25)
        val_ids_df = df[df["Patient ID"].isin(val_ids)]  # .sample(15)
        test_ids_df = df[df["Patient ID"].isin(test_ids)]  # .sample(15)

        return train_ids_df, val_ids_df, test_ids_df

    def get_patches_paths_dfs(self, dir_excels_class_perc):

        # Read the excel including the images paths and their tissue percentage
        train_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "train_patches_class_perc_0_tp.csv")
        val_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "val_patches_class_perc_0_tp.csv")
        test_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "test_patches_class_perc_0_tp.csv")

        return train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df

    def load_MIL_datasets_coordinates(self, dir_images, train_ids_df, val_ids_df, test_ids_df):

        dataset_train = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=train_ids_df, classes=classes,
                                                       pred_column=pred_column, pred_mode=pred_mode,
                                                       magnification_level=magnification_level,
                                                       bag_id='Patient ID', input_shape=input_shape,
                                                       data_augmentation=data_augmentation,
                                                       stain_normalization=stain_normalization,
                                                       images_on_ram=images_on_ram,
                                                       include_background=include_background,
                                                       class_perc_data_frame=train_class_perc_patches_paths_df,
                                                       tissue_percentages_max=tissue_percentages_max)

        data_generator_train = MILDataGenerator_coords(dataset_train, batch_size=1, shuffle=not (ordered),
                                                       max_instances=max_instances, num_workers=0,
                                                       pred_column=pred_column,
                                                       images_on_ram=images_on_ram, pred_mode=pred_mode,
                                                       return_patient_id=True)

        # Validation
        dataset_val = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=val_ids_df, classes=classes,
                                                     pred_column=pred_column, pred_mode=pred_mode,
                                                     magnification_level=magnification_level,
                                                     bag_id='Patient ID', input_shape=input_shape,
                                                     data_augmentation=False, stain_normalization=stain_normalization,
                                                     images_on_ram=images_on_ram, include_background=include_background,
                                                     class_perc_data_frame=val_class_perc_patches_paths_df,
                                                     tissue_percentages_max=tissue_percentages_max)

        data_generator_val = MILDataGenerator_coords(dataset_val, batch_size=1, shuffle=False,
                                                     max_instances=max_instances, num_workers=0,
                                                     pred_column=pred_column, pred_mode=pred_mode,
                                                     images_on_ram=images_on_ram, return_patient_id=True)

        # Test
        dataset_test = MILDataset_w_class_perc_coords(dir_images=dir_images, data_frame=test_ids_df, classes=classes,
                                                      pred_column=pred_column, pred_mode=pred_mode,
                                                      magnification_level=magnification_level,
                                                      bag_id='Patient ID', input_shape=input_shape,
                                                      data_augmentation=False, stain_normalization=stain_normalization,
                                                      images_on_ram=images_on_ram,
                                                      include_background=include_background,
                                                      class_perc_data_frame=test_class_perc_patches_paths_df,
                                                      tissue_percentages_max=tissue_percentages_max)

        data_generator_test = MILDataGenerator_coords(dataset_test, batch_size=1, shuffle=False,
                                                      max_instances=max_instances,
                                                      num_workers=0, pred_column=pred_column, pred_mode=pred_mode,
                                                      images_on_ram=images_on_ram, return_patient_id=True)

        return dataset_train

    def get_MIL_datasets_coordinates(self):
        return self.dataset_train, self.data_generator_train, self.dataset_val, self.data_generator_val, self.dataset_test, self.data_generator_test

        print("hola")




if __name__ == '__main__':

    wsi_graph_g = WSI2Graph_Generator(pred_column="Molecular subtype",
                                      bag_id="Patient ID",
                                    feature_extractor_name="PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10x_N_100_AUGM_non-spatial_SN_False_BAL_True",
                                    feature_extractor_path="Z:/Shared_PFC-TFG-TFM/Claudio/MOLSUB_PRED/output/[03_10_2023]_MolSub_SLURM_full",
                                    model_weights_filename="network_weights_best_f1.pth",
                                    knn_list=[5, 8, 19, 25, 50, 75],
                                    magnification_level="10x",
                                    include_edge_features = True,
                                    edges_type = "spatial",
                                    dir_data_frame = "../data/BCNB/patient-clinical-data.xlsx",
                                    dir_dataset_splitting="../data/BCNB/dataset-splitting",
                                    dir_excels_class_perc="../data/BCNB/patches_paths_class_perc/",
                                    dir_results="../data/BCNB/results_graphs_november_23"
                                    )



    print("hola")


