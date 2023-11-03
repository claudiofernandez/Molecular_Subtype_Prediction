from MIL_utils import *

####################
# GCN Data Methods #
####################

class MILDataGenerator_coords_rID(object):

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, return_patient_id=False): #512

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram
        self.return_patient_id = return_patient_id

        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)


    def __iter__(self):
        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # extract patient id
        patient_id = str(df_row['Patient ID'])

        # Get bag-level label GT
        Y = df_row[self.dataset.pred_column]
        Y = np.expand_dims(np.array(Y), 0)

        # Select instances from bag
        ID = list(df_row[[self.dataset.bag_id]].values)[0] #Patient IDk
        images_id = self.dataset.D[str(ID)] #Images corresponding to the Patient ID

        print(len(images_id))
        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)
            print(len(images_id))

        # QUIZÁ SE PUEDE METER AQUI, QUE SI YA HA SAMPLEADO X DE UNA CLASE QUE NO SAMPLEE MAS


        # if self.images_on_ram:
        #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #           self.dataset.X_augm[self._idx, :, :, :])
        #     # for i in images_id:
        #     #     x, x_augm = self.dataset.__getitem__(i)
        #     # # Return requested image from prellocated in memory
        #     # X =
        #     #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #     #           self.dataset.X_augm[self._idx, :, :, :])

        # Load images and include into the batch
        X = []
        X_augm = []
        imgs_coords = []
        for i in images_id:
            x, x_augm, img_coord_x, img_coord_y  = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)
            imgs_coords.append((img_coord_x, img_coord_y))

        # Update bag index iterator
        self._idx += self.batch_size

        if Y.shape == ():
            print("Error with Y shape")

        Y=Y[0]

        if self.pred_column == "Molecular subtype":
            # One-hot encoding
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A' or Y == 'Luminal B':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0.]
                if Y == 'Luminal B':
                    Y = [1., 0.]
                if Y == 'HER2(+)':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]
        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            if not self.return_patient_id:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords
            else:
                return patient_id, np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords

        else:
            if not self.return_patient_id:
                return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords
            else:
                return patient_id, np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords



    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

class MILDataset_offline_graphs(object):

    def __init__(self, dir_graphs, data_frame, pred_column, pred_mode, classes,
                 graphs_on_ram=False, channel_first=True):

        self.dir_graphs = dir_graphs
        self.data_frame = data_frame
        self.classes = classes
        self.graphs_on_ram = graphs_on_ram
        self.channel_first = channel_first
        self.pred_column = pred_column
        self.pred_mode = pred_mode

        # Select graphs from directory
        self.graphs_paths = os.listdir(self.dir_graphs)
        self.graphs_paths.sort(key=lambda x: int(x.split("_")[0])) # sort the graphs in numerical order

        # Filter graphs present in the given dataframe (train, test, val)
        selected_patient_ids = set(self.data_frame["Patient ID"].astype(str)) # Convert the "Patient ID" column to a set for fast membership testing

        # Filter the list of files to only include those that match the IDs in the dataframe
        self.selected_graphs_paths = [f for f in self.graphs_paths if f.split("_")[0] in selected_patient_ids]

        # Retrieve the label of each graph from the dataframe
        self.y = self.data_frame[self.pred_column].values

        # Preallocate Graphs on RAM
        if self.graphs_on_ram:
            self.G = []
            for graph_path in tqdm(self.selected_graphs_paths):
                self.G.append(torch.load(os.path.join(self.dir_graphs, graph_path)))

            print('[INFO]: Graphs loaded')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.selected_graphs_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if self.graphs_on_ram:
            graph = self.G[index]
        else:
            graph_path = self.selected_graphs_paths[index]
            graph = torch.load(os.path.join(self.dir_graphs, graph_path))

        y = self.y[index]
        return graph, y

class MILDataGenerator_offline_graphs_balanced(object):
    def __len__(self):
        N = len(self.dataset.data_frame)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.dataset.data_frame)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, graphs_on_ram, batch_size=1,  shuffle=False, max_instances=100,):
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        self.graphs_on_ram = graphs_on_ram
        self.d_len = len(self.dataset.selected_graphs_paths)

        # Initialize iterator
        self._idx = 0
        self._reset()

        # Initialize last_graph_idxs dictionary to store last selected graph_idx for each class
        self.last_graph_idxs = {}

        # For class balancing
        self.d_classes = self.dataset.y

        # Change computation of class weights depending on training mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.d_classes = self.d_classes
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]

        # Dict with graphs ids (keys) and corresponding labels (values)
        self.graphs_ids = np.arange(self.d_len)
        self.zip_gen = zip(self.graphs_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen) # dict with indexes of graphs paths as keys, and classes as values.

        # Another way of computing class weights
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)

        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes),
                                             y=self.d_classes)
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # List with same length as bag IDs, containing the indexes of the balanced classes
        self.balanced_class_idx = torch.multinomial(input=self.weights_t,
                                                    num_samples=self.d_len,
                                                    replacement=True)
        print(self.weights_t)
        print(self.balanced_class_idx)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Choose balanced class to find bag ID to return its images
        chosen_class_idx = self.balanced_class_idx[self._idx]
        chosen_label = self.ordered_clases[chosen_class_idx]

        # Graphs IDs with the chosen label
        matching_graph_idxs = [graph_idx for graph_idx, label in self.d_dict.items() if label == chosen_label]

        # Choose a new graph index that is different from the last one for this class
        last_graph_idx = self.last_graph_idxs.get(chosen_label)
        while True:
            # Randomly choose a graph index from the matching graph indexes
            graph_idx = random.choice(matching_graph_idxs)
            if graph_idx != last_graph_idx:
                self.last_graph_idxs[chosen_label] = graph_idx
                break

        graph, y = self.dataset.__getitem__(graph_idx)

        # Sanity check
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            assert chosen_label == y, "Chosen label does not match dataset label"
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            if y == "Luminal A" or y == "Luminal B":
                y = "Luminal"
            assert chosen_label == y, "Chosen label does not match dataset label"
        elif self.pred_mode == "OTHERvsTNBC":
            if y == "Luminal A" or y == "Luminal B" or y == "HER2(+)":
                y = "Other"
            assert chosen_label == y, "Chosen label does not match dataset label"

        # One-hot encoding of labels
        Y = chosen_label

        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Other':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]
        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        # Update bag index iterator
        self._idx += self.batch_size

        return graph, Y

        # if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
        #     return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords
        # else:
        #     return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords

class MILDataGenerator_offline_graphs_nonbalanced(object):
    def __len__(self):
        N = len(self.dataset.data_frame)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.dataset.data_frame)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, graphs_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, balanced=True ): #512
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        self.graphs_on_ram = graphs_on_ram
        self.d_len = len(self.dataset.selected_graphs_paths)

        # Initialize iterator
        self._idx = 0
        self._reset()

        # Initialize last_graph_idxs dictionary to store last selected graph_idx for each class
        self.last_graph_idxs = {}
        #
        # # For class balancing
        # self.d_classes = self.dataset.y
        #
        # # Change computation of class weights depending on training mode
        # if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
        #     self.d_classes = self.d_classes
        # elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
        #     self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        # elif self.pred_mode == "OTHERvsTNBC":
        #     self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]
        #
        # # Dict with graphs ids (keys) and corresponding labels (values)
        # self.graphs_ids = np.arange(self.d_len)
        # self.zip_gen = zip(self.graphs_ids, self.d_classes)
        # self.d_dict = dict(self.zip_gen) # dict with indexes of graphs paths as keys, and classes as values.
        #
        # # Another way of computing class weights
        # self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        #
        # self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes),
        #                                      y=self.d_classes)
        # self.weights_t = torch.DoubleTensor(list(self.weights))
        #
        # # List with same length as bag IDs, containing the indexes of the balanced classes
        # self.balanced_class_idx = torch.multinomial(input=self.weights_t,
        #                                             num_samples=self.d_len,
        #                                             replacement=True)
        # print(self.weights_t)
        # print(self.balanced_class_idx)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Retrieve graph and label
        graph, y = self.dataset.__getitem__(self._idx)

        # Pred mode adaptation and sanity check
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            assert y == y, "Chosen label does not match dataset label"
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            if y == "Luminal A" or y == "Luminal B":
                y = "Luminal"
            assert y == y, "Chosen label does not match dataset label"
        elif self.pred_mode == "OTHERvsTNBC":
            if y == "Luminal A" or y == "Luminal B" or y == "HER2(+)":
                y = "Other"
            assert y == y, "Chosen label does not match dataset label"

        # One-hot encoding of labels
        Y = y

        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Other':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]
        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        # Update bag index iterator
        self._idx += self.batch_size

        return graph, Y

        # if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
        #     return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords
        # else:
        #     return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords

class MILDataset_w_class_perc_coords(object):

    def __init__(self, dir_images, data_frame, pred_column, pred_mode, magnification_level, class_perc_data_frame, tissue_percentages_max, classes, bag_id='Patient ID', input_shape=(3, 224, 224),
                 data_augmentation=False, images_on_ram=False, channel_first=True, stain_normalization=False, include_background=False):

        """Dataset object for MIL.
            Dataset object which aims to organize images and labels from a dataset in the form of bags.
        Args:
          dir_images: (h, w, channels)
          data_frame: pandas dataframe with ground truth information.
                      Each bag is one raw, with 'bag_name' as identifier.
          classes: list of classes of interest in data_fame (i.e. ['G3', 'G4', 'G5'])
          input_shape: image input shape (channels first).
          data_augmentation: whether to perform data augmentation (True) or not (False).
          images_on_ram: whether to load images on ram (True) or not (False). Recommended for accelerated training.

        Returns:
          MILDataset object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bag_id = bag_id
        self.data_augmentation = data_augmentation
        self.stain_normalization = stain_normalization
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.magnification_level = magnification_level
        self.include_background = include_background

        if stain_normalization:
            # target_stain_patch_path = os.path.join(self.dir_images, "26", "26_0_0_0.jpg")
            target_stain_patch_path = "../data/color_norm_img_samples/01.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/55_10_12.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/19_10_38.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/664_4608_5120.jpg"
            #target_stain_patch_path = os.path.join(self.dir_images, "71", "71_1_0_0.jpg")
            self.stain_normalization_function = Stain_Normalization(target_image_path = target_stain_patch_path,
                                                                    target_shape=(input_shape[1],input_shape[2]))

        self.tissue_percentages_max = tissue_percentages_max
        class_perc_0_max = float(self.tissue_percentages_max.split("-")[0].split("_")[-1])
        class_perc_1_max = float(self.tissue_percentages_max.split("-")[1].split("_")[-1])
        class_perc_2_max = float(self.tissue_percentages_max.split("-")[2].split("_")[-1])
        class_perc_3_max = float(self.tissue_percentages_max.split("-")[3].split("_")[-1])
        class_perc_4_max = float(self.tissue_percentages_max.split("-")[4].split("_")[-1])


        # Filter and extract patches paths based on tissue percentage
        filtered_rows = class_perc_data_frame.query("class_perc_0 <= " + str(class_perc_0_max) +
                                        "and class_perc_1 <= " + str(class_perc_1_max) +
                                        "and class_perc_2 <= " + str(class_perc_2_max) +
                                        "and class_perc_3 <= " + str(class_perc_3_max) +
                                        "and class_perc_4 <= " + str(class_perc_4_max))

        selected_images_paths = list(filtered_rows["patch_path"])


        # Comment if executing locally
        #TODO: add where_exec param
        #TODO: change all paths relative to where_exec
        selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/",
                                              "../data/BCNB/preprocessing_results_bien/") for path in
                                 selected_images_paths]
        #selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/", "../data/") for path in selected_images_paths]

        self.images_paths = selected_images_paths

        self.data_frame = self.data_frame[
            np.in1d(self.data_frame[self.bag_id], [img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths])]


        # Filter imgs paths by selected IDs
        chosen_ids = np.array(self.data_frame["Patient ID"])         # Array of chosen IDs

        new_selected_images_paths = [path for path in selected_images_paths if int(path.split("/")[-1].split("_")[0]) in chosen_ids]

        self.images_paths = new_selected_images_paths
        print("Total patches for the selected IDs: ", str(len(self.images_paths)))


        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        #for i, item in enumerate([ID.split('_')[0] for ID in self.images]):

        for i, img_path in enumerate(self.images_paths):
            img_path_idx = i # number among the total length of self.images_path
            img_filename = img_path.split('/')[-1] # base filename of the image
            item = img_filename.split("_")[0] # patient ID extracted from filename
            img_coord_x = int(img_filename.split("_")[1])
            img_coord_y = int(img_filename.split("_")[2].split(".")[0])

            if item not in self.D: # key: patient ID
                self.D[item] = [(i, img_coord_x, img_coord_y)]
            else:
                self.D[item].append((i, img_coord_x, img_coord_y))

        # old (23/12/2022)
        # for i, item in enumerate([img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths]):
        #     if item not in self.D:
        #         self.D[item] = [i]
        #     else:
        #         self.D[item].append(i)

        self.y = self.data_frame[self.pred_column].values
        self.indexes = np.arange(len(self.images_paths))


        # Preallocate Images on RAM
        if self.images_on_ram:

            # Pre-allocate images
            self.X = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for images
            self.X_augm = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for augmented images
            self.img_coord_x = np.zeros((len(self.indexes)))
            self.img_coord_y = np.zeros((len(self.indexes)))

            #self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)

            if self.include_background:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)
            else:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes)), dtype=np.float32) # Create empty array for labels

            #if self.dataframe_instances is not False:
                #self.y_instances = -1 * np.ones((len(self.indexes), 4))

            # Load, and normalize images
            print('[INFO]: Loading images on RAM memory ...')
            for i in tqdm(np.arange(len(self.indexes)), leave=True, position=0):
                #print(str(i) + '/' + str(len(self.indexes)), end='\r')

                img_path = self.images_paths[self.indexes[i]]

                #Derive Patient ID
                img_patient_id = int(img_path.split("/")[-1].split("_")[0])

                # Check if Patient ID is in the given DataFrame, if not try with the next one
                if self.data_frame[self.data_frame["Patient ID"] == img_patient_id].shape[0] == 0:
                    print("Patient ID "+ str(img_patient_id) + " not present in the given IDs DataFrame. Trying with next ID.")
                    continue

                # For coordinates
                img_filename = img_path.split('/')[-1]  # base filename of the image
                #item = img_filename.split("_")[0]  # patient ID extracted from filename
                img_coord_x = int(img_filename.split("_")[1])
                img_coord_y = int(img_filename.split("_")[2].split(".")[0])

                # Load image
                x_org = Image.open(img_path)

                # # Select magnification
                # if self.magnification_level == "20x":
                #     x = x_org
                # elif self.magnification_level == "10x":
                #     x = x_org.resize((256, 256))

                x = x_org

                # Apply Macenko Stain Normalization if wanted
                if self.stain_normalization:
                    try:
                        #plt.imshow(x)
                        #plt.show()
                        x = self.stain_normalization_function(x)
                        #plt.imshow(x)
                        #plt.show()
                    except RuntimeError:
                        print("Full black patch found! Check patient's images...")
                        print(img_path)

                x = np.asarray(x, dtype='uint8')

                # Data Augmentation
                if self.data_augmentation:
                    # x_augm = self.image_transformation(x.copy())
                    if self.data_augmentation == "non-spatial":
                        transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Transpose(p=0.5),
                            A.GridDistortion(p=0.5),
                            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)])

                        x_augm = transform(image=x.copy())["image"]
                        # Normalization
                        x_augm = self.image_normalization(x_augm)  # Normalize image

                        # if self.channel_first:
                        #     x_augm = np.transpose(x_augm.copy(), (1, 2, 0))
                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                    if self.data_augmentation == "randstainna":
                        #### calling the randstainna
                        transforms_list = [
                            RandStainNA(yaml_file='../data/color_norm_img_samples/randstainna_params.yaml',
                                        std_hyper=-0.3, probability=1.0,
                                        distribution='normal', is_train=True)
                        ]

                        transform = transforms.Compose(transforms_list)

                        if self.channel_first:
                            img = np.transpose(x.copy(), (1, 2, 0))

                        x_augm = transform(cv2.imread(img_path))
                        x_augm = self.image_normalization(x_augm)

                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                # plt.imshow(x_augm)
                # plt.show()

                # if self.channel_first and self.data_augmentation:
                #     x_augm = np.transpose(x_augm, (2, 0, 1))
                else:
                    # Normalization
                    x = self.image_normalization(x)  # Normalize image
                    x_augm = None

                    #print(type(x), x.shape)
                #x = Image.open(os.path.join(self.dir_images, ID))
                # x = np.asarray(x)
                # # Normalization
                # x = self.image_normalization(x)
                self.X[self.indexes[i], :, :, :] = x
                self.X_augm[self.indexes[i], :, :, :] = x_augm
                self.img_coord_x[self.indexes[i]] = img_coord_x
                self.img_coord_y[self.indexes[i]] = img_coord_y


                #Get the GT Label of the image
                y_label = self.data_frame[self.data_frame["Patient ID"] == img_patient_id][self.pred_column].to_string(index=False)

                if self.pred_column == "Molecular subtype":
                    if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [0., 1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 0., 1.]
                    if self.pred_mode == "LUMINALSvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 1.]
                    if self.pred_mode == "OTHERvsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 1.]

                elif self.pred_column == "ALN status":
                    if y_label == 'N0':
                        y_label = [1., 0., 0.]
                    if y_label == 'N+(1-2)':
                        y_label = [0., 1., 0.]
                    if y_label == 'N+(>2)':
                        y_label = [0., 0., 1.]
                elif self.pred_column == "HER2 Expression":
                    if y_label == '0':
                        y_label = [1., 0., 0., 0.]
                    if y_label == '1+':
                        y_label = [0., 1., 0., 0.]
                    if y_label == '2+':
                        y_label = [0., 0., 1., 0.]
                    if y_label == '3+':
                        y_label = [0., 0., 0., 1.]

                #self.Yglobal[self.indexes[i], 1:] = self.data_frame[classes][self.data_frame[self.bag_id] == int(self.images[i].split('/')[-1].split("_")[0])]

                if self.include_background:
                    self.Yglobal[self.indexes[i], 1:] = y_label
                else:
                    self.Yglobal[self.indexes[i], :] = y_label

            print('[INFO]: Images loaded')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_path = self.images_paths[self.indexes[index[0]]]
        #print(img_path)
        img_coord_x = index[1] #index[0]: patient ID, index[1]: x_coord, index[2]:y_coord
        img_coord_y = index[2]

        # old: 23/12/2022
        # img_path = self.images_paths[self.indexes[index]]

        if self.images_on_ram:
            x = np.squeeze(self.X[self.indexes[index[0]], :, :, :])
            x_augm = np.squeeze(self.X_augm[self.indexes[index[0]], :, :, :])
            img_coord_x = np.squeeze(self.img_coord_x[self.indexes[index[0]]])
            img_coord_y = np.squeeze(self.img_coord_y[self.indexes[index[0]]])

        else:
            # Load image
            x_org = Image.open(img_path)
            # # Select magnification
            # if self.magnification_level == "20x":
            #     x = x_org
            # elif self.magnification_level == "10x":
            #     x = x_org.resize((256,256))

            x = x_org
            # Stain normalization:
            if self.stain_normalization:
                try:
                    #plt.imshow(x)
                    #plt.show()
                    x = self.stain_normalization_function(x)
                    #plt.imshow(x)
                    #plt.show()
                except RuntimeError:
                    print("Full black patch found! Check patient's images...")
                    print(img_path)
                    #print(type(x), x.shape)

            #x = Image.open(os.path.join(self.dir_images, ID))
            x = np.asarray(x, dtype='uint8')
            # Normalization
            x = self.image_normalization(x)

            # Data Augmentation
            if self.data_augmentation:
                #x_augm = self.image_transformation(x.copy())
                if self.data_augmentation == "non-spatial":
                    transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)])
                    if self.channel_first:
                        img = np.transpose(x.copy(), (1, 2, 0))
                    x_augm = transform(image=img.copy())["image"]

                if self.data_augmentation == "randstainna":
                    #### calling the randstainna
                    transforms_list = [
                        RandStainNA(yaml_file='../data/color_norm_img_samples/randstainna_params.yaml', std_hyper=-0.3, probability=1.0,
                                    distribution='normal', is_train=True)
                    ]

                    transform = transforms.Compose(transforms_list)

                    if self.channel_first:
                        img = np.transpose(x.copy(), (1, 2, 0))

                    x_augm = transform(cv2.imread(img_path))
            # plt.imshow(x_augm)
            # plt.show()

            if self.channel_first and self.data_augmentation:
                x_augm = np.transpose(x_augm, (2, 0, 1))
            else:
                x_augm = None

        #if self.stain_normalization:
        #    x = self.stain_normalization_function(x.copy())

        return x, x_augm, img_coord_x, img_coord_y


    def image_normalization(self, x):
        # image resize
        x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        #if random.random() > 0.5:
        #    img = skimage.util.random_noise(img, var=random.random() ** 2)
        #if random.random() > 0.5:
        #    img = img + random.random() - 0.5
        #    img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def image_transformation_noise(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        if random.random() > 0.5:
           img = skimage.util.random_noise(img, var=random.random() ** 2)
        if random.random() > 0.5:
           img = img + random.random() - 0.5
           img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

class MILDataGenerator_balanced_good_coords(object):
    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, balanced=True ): #512
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram

        # Initialize iterator
        self._idx = 0
        self._reset()

        # For class balancing
        self.d_classes = self.dataset.y

        # Change computation of class weights depending on training mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.d_classes = self.d_classes
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]

        self.d_im_ids = list(self.dataset.D.keys())
        self.d_im_ids = [eval(i) for i in self.d_im_ids]
        self.d_im_ids = sorted(self.d_im_ids)
        self.d_len = len(self.d_im_ids)
        self.zip_gen = zip(self.d_im_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen) # dict with img IDs and corresponding label

        # One way of computing class weights
        # self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        # self.weights = 1.0 / self.class_counts
        # self.weights_t = torch.DoubleTensor(list(self.weights))


        # Another way of computing class weights
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)

        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes),
                                             y=self.d_classes)
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # List with same length as bag IDs, containing the indexes of the balanced classes
        self.balanced_class_idx = torch.multinomial(input=self.weights_t,
                                                    num_samples=self.d_len,
                                                    replacement=True)
        print(self.weights_t)
        print(self.balanced_class_idx)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Choose balanced class to find bag ID to return its images
        chosen_class_idx = self.balanced_class_idx[self._idx]
        chosen_label = self.ordered_clases[chosen_class_idx]
        images_ids_w_label = {i for i in self.d_dict if self.d_dict[i] == chosen_label}
        #print(images_ids_w_label)
        random_selected_id_for_balance = random.sample(images_ids_w_label, 1)[0]
        #print(random_selected_id_for_balance)
        images_id = self.dataset.D[str(random_selected_id_for_balance)]
        #print(images_id)

        df_row = self.dataset.data_frame.loc[self.dataset.data_frame["Patient ID"] == random_selected_id_for_balance]

        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)

        # Load images and include into the batch
        X = []
        X_augm = []
        imgs_coords = []
        for i in images_id:
            x, x_augm, img_coord_x, img_coord_y  = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)
            imgs_coords.append((img_coord_x, img_coord_y))

        # Update bag index iterator
        self._idx += self.batch_size

        Y = chosen_label
        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Other':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]

        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords
        else:
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords

class MILDataGenerator_coords(object):

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, ): #512

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram

        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)


    def __iter__(self):
        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # Get bag-level label GT
        Y = df_row[self.dataset.pred_column]
        Y = np.expand_dims(np.array(Y), 0)

        # Select instances from bag

        ID = list(df_row[[self.dataset.bag_id]].values)[0] #Patient IDk

        images_id = self.dataset.D[str(ID)] #Images corresponding to the Patient ID

        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)

        # QUIZÁ SE PUEDE METER AQUI, QUE SI YA HA SAMPLEADO X DE UNA CLASE QUE NO SAMPLEE MAS


        # if self.images_on_ram:
        #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #           self.dataset.X_augm[self._idx, :, :, :])
        #     # for i in images_id:
        #     #     x, x_augm = self.dataset.__getitem__(i)
        #     # # Return requested image from prellocated in memory
        #     # X =
        #     #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #     #           self.dataset.X_augm[self._idx, :, :, :])

        # Load images and include into the batch
        X = []
        X_augm = []
        imgs_coords = []
        for i in images_id:
            x, x_augm, img_coord_x, img_coord_y  = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)
            imgs_coords.append((img_coord_x, img_coord_y))

        # Update bag index iterator
        self._idx += self.batch_size

        if Y.shape == ():
            print("Error with Y shape")

        Y=Y[0]

        if self.pred_column == "Molecular subtype":
            # One-hot encoding
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A' or Y == 'Luminal B':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0.]
                if Y == 'Luminal B':
                    Y = [1., 0.]
                if Y == 'HER2(+)':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]
        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32'), imgs_coords
        else:
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), None, imgs_coords



    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

####################################
# Mean and Max Aggregation Methods #
####################################
class MILDataset_w_class_perc(object):

    def __init__(self, dir_images, data_frame, pred_column, pred_mode, magnification_level, class_perc_data_frame, tissue_percentages_max, classes, gnrl_data_dir, where_exec, bag_id='Patient ID', input_shape=(3, 224, 224),
                 data_augmentation=False, images_on_ram=False, channel_first=True, stain_normalization=False, include_background=False):

        """Dataset object for MIL.
            Dataset object which aims to organize images and labels from a dataset in the form of bags.
        Args:
          dir_images: (h, w, channels)
          data_frame: pandas dataframe with ground truth information.
                      Each bag is one raw, with 'bag_name' as identifier.
          classes: list of classes of interest in data_fame (i.e. ['G3', 'G4', 'G5'])
          input_shape: image input shape (channels first).
          data_augmentation: whether to perform data augmentation (True) or not (False).
          images_on_ram: whether to load images on ram (True) or not (False). Recommended for accelerated training.

        Returns:
          MILDataset object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bag_id = bag_id
        self.data_augmentation = data_augmentation
        self.stain_normalization = stain_normalization
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.magnification_level = magnification_level
        self.include_background = include_background
        self.where_exec = where_exec
        self.gnrl_data_dir = gnrl_data_dir

        if stain_normalization:
            # target_stain_patch_path = os.path.join(self.dir_images, "26", "26_0_0_0.jpg")
            target_stain_patch_path = "../data/color_norm_img_samples/01.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/55_10_12.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/19_10_38.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/664_4608_5120.jpg"
            #target_stain_patch_path = os.path.join(self.dir_images, "71", "71_1_0_0.jpg")
            self.stain_normalization_function = Stain_Normalization(target_image_path = target_stain_patch_path,
                                                                    target_shape=(input_shape[1],input_shape[2]))

        self.tissue_percentages_max = tissue_percentages_max
        class_perc_0_max = float(self.tissue_percentages_max.split("-")[0].split("_")[-1])
        class_perc_1_max = float(self.tissue_percentages_max.split("-")[1].split("_")[-1])
        class_perc_2_max = float(self.tissue_percentages_max.split("-")[2].split("_")[-1])
        class_perc_3_max = float(self.tissue_percentages_max.split("-")[3].split("_")[-1])
        class_perc_4_max = float(self.tissue_percentages_max.split("-")[4].split("_")[-1])


        # Filter and extract patches paths based on tissue percentage
        filtered_rows = class_perc_data_frame.query("class_perc_0 <= " + str(class_perc_0_max) +
                                        "and class_perc_1 <= " + str(class_perc_1_max) +
                                        "and class_perc_2 <= " + str(class_perc_2_max) +
                                        "and class_perc_3 <= " + str(class_perc_3_max) +
                                        "and class_perc_4 <= " + str(class_perc_4_max))

        selected_images_paths = list(filtered_rows["patch_path"])
        #selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/", "../data/") for path in selected_images_paths]
        #selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/", "../data/BCNB/preprocessing_results_bien/") for path in selected_images_paths]
        # TODO: change all paths relative to where_exec
        if self.where_exec == "slurm_nas":
            selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/", "../data/BCNB/preprocessing_results_bien/") for path in selected_images_paths]
        elif self.where_exec == "slurm_dgx":
            selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/",
                                                  os.path.join(self.gnrl_data_dir, "preprocessing_results_bien/")) for path in
                                     selected_images_paths]
        elif self.where_exec == "dgx_gpu":
            selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/",
                                                  "../data/BCNB/preprocessing_results_bien/") for path in
                                     selected_images_paths]
        elif self.where_exec == "local":
            # TODO: write for executing locally
            selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/",
                                                  "../data/BCNB/preprocessing_results_bien/") for path in
                                     selected_images_paths]

        # Now selected_images_paths contains the paths for ALL IDs (unilike before where it only contained the ones for training)
        # Therefore, we need to filter these paths based on the content of the "Patient_ID" column of the self.data_frame (which only contains the IDs for training)
        selected_patient_ids = set(self.data_frame['Patient ID'])
        # filter the selected image paths to keep only those with a matching patient ID
        filtered_image_paths = [path for path in selected_images_paths if
                                int(path.split('/')[-1].split("_")[0]) in selected_patient_ids]

        self.images_paths = filtered_image_paths
        print("Total patches for the selected IDs: ", str(len(filtered_image_paths)))

        self.data_frame = self.data_frame[
            np.in1d(self.data_frame[self.bag_id], [img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths])]

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        #for i, item in enumerate([ID.split('_')[0] for ID in self.images]):
        for i, item in enumerate([img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths]):
            if item not in self.D:
                self.D[item] = [i]
            else:
                self.D[item].append(i)

        self.y = self.data_frame[self.pred_column].values
        self.indexes = np.arange(len(self.images_paths))


        # Preallocate Images on RAM
        if self.images_on_ram:
            # Pre-allocate images
            self.X = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for images
            self.X_augm = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for augmented images
            #self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)

            if self.include_background:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)
            else:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes)), dtype=np.float32) # Create empty array for labels

            #if self.dataframe_instances is not False:
                #self.y_instances = -1 * np.ones((len(self.indexes), 4))

            # Load, and normalize images
            print('[INFO]: Loading images on RAM memory ...')
            for i in tqdm(np.arange(len(self.indexes)), leave=True, position=0):
                #print(str(i) + '/' + str(len(self.indexes)), end='\r')

                img_path = self.images_paths[self.indexes[i]]
                #Derive Patient ID
                img_patient_id = int(img_path.split("/")[-1].split("_")[0])
                # Load image
                x_org = Image.open(img_path)

                # # Select magnification
                # if self.magnification_level == "20x":
                #     x = x_org
                # elif self.magnification_level == "10x":
                #     x = x_org.resize((256, 256))

                x = x_org

                # Apply Macenko Stain Normalization if wanted
                if self.stain_normalization:
                    try:
                        #plt.imshow(x)
                        #plt.show()
                        x = self.stain_normalization_function(x)
                        #plt.imshow(x)
                        #plt.show()
                    except RuntimeError:
                        print("Full black patch found! Check patient's images...")
                        print(img_path)

                x = np.asarray(x, dtype='uint8')

                # Data Augmentation
                if self.data_augmentation:
                    # x_augm = self.image_transformation(x.copy())
                    if self.data_augmentation == "non-spatial":
                        transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Transpose(p=0.5),
                            A.GridDistortion(p=0.5),
                            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)])

                        x_augm = transform(image=x.copy())["image"]
                        # Normalization
                        x_augm = self.image_normalization(x_augm)  # Normalize image

                        # if self.channel_first:
                        #     x_augm = np.transpose(x_augm.copy(), (1, 2, 0))
                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                    if self.data_augmentation == "randstainna":
                        #### calling the randstainna
                        transforms_list = [
                            RandStainNA(yaml_file='../data/color_norm_img_samples/randstainna_params.yaml',
                                        std_hyper=-0.3, probability=1.0,
                                        distribution='normal', is_train=True)
                        ]

                        transform = transforms.Compose(transforms_list)

                        if self.channel_first:
                            img = np.transpose(x.copy(), (1, 2, 0))

                        x_augm = transform(cv2.imread(img_path))
                        x_augm = self.image_normalization(x_augm)

                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                # plt.imshow(x_augm)
                # plt.show()

                # if self.channel_first and self.data_augmentation:
                #     x_augm = np.transpose(x_augm, (2, 0, 1))
                else:
                    # Normalization
                    x = self.image_normalization(x)  # Normalize image
                    x_augm = None

                    #print(type(x), x.shape)
                #x = Image.open(os.path.join(self.dir_images, ID))
                # x = np.asarray(x)
                # # Normalization
                # x = self.image_normalization(x)
                self.X[self.indexes[i], :, :, :] = x
                self.X_augm[self.indexes[i], :, :, :] = x_augm

                #Get the GT Label of the image
                y_label = self.data_frame[self.data_frame["Patient ID"] == img_patient_id][self.pred_column].to_string(index=False)



                if self.pred_column == "Molecular subtype":
                    if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [0., 1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 0., 1.]
                    if self.pred_mode == "LUMINALSvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 1.]
                    if self.pred_mode == "OTHERvsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 1.]

                elif self.pred_column == "ALN status":
                    if y_label == 'N0':
                        y_label = [1., 0., 0.]
                    if y_label == 'N+(1-2)':
                        y_label = [0., 1., 0.]
                    if y_label == 'N+(>2)':
                        y_label = [0., 0., 1.]
                elif self.pred_column == "HER2 Expression":
                    if y_label == '0':
                        y_label = [1., 0., 0., 0.]
                    if y_label == '1+':
                        y_label = [0., 1., 0., 0.]
                    if y_label == '2+':
                        y_label = [0., 0., 1., 0.]
                    if y_label == '3+':
                        Y = [0., 0., 0., 1.]

                #self.Yglobal[self.indexes[i], 1:] = self.data_frame[classes][self.data_frame[self.bag_id] == int(self.images[i].split('/')[-1].split("_")[0])]

                if self.include_background:
                    self.Yglobal[self.indexes[i], 1:] = y_label
                else:
                    self.Yglobal[self.indexes[i], :] = y_label

            print('[INFO]: Images loaded')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_path = self.images_paths[self.indexes[index]]

        if self.images_on_ram:
            x = np.squeeze(self.X[self.indexes[index], :, :, :])
            x_augm = np.squeeze(self.X_augm[self.indexes[index], :, :, :])

        else:
            # Load image
            x_org = Image.open(img_path)
            # # Select magnification
            # if self.magnification_level == "20x":
            #     x = x_org
            # elif self.magnification_level == "10x":
            #     x = x_org.resize((256,256))

            x = x_org
            # Stain normalization:
            if self.stain_normalization:
                try:
                    #plt.imshow(x)
                    #plt.show()
                    x = self.stain_normalization_function(x)
                    #plt.imshow(x)
                    #plt.show()
                except RuntimeError:
                    print("Full black patch found! Check patient's images...")
                    print(img_path)
                    #print(type(x), x.shape)

            #x = Image.open(os.path.join(self.dir_images, ID))
            x = np.asarray(x, dtype='uint8')
            # Normalization
            x = self.image_normalization(x)

            # Data Augmentation
            if self.data_augmentation:
                #x_augm = self.image_transformation(x.copy())
                if self.data_augmentation == "non-spatial":
                    transform = A.Compose([
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.5),
                        A.Transpose(p=0.5),
                        A.GridDistortion(p=0.5),
                        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)])
                    if self.channel_first:
                        img = np.transpose(x.copy(), (1, 2, 0))
                    x_augm = transform(image=img.copy())["image"]

                if self.data_augmentation == "randstainna":
                    #### calling the randstainna
                    transforms_list = [
                        RandStainNA(yaml_file='../data/color_norm_img_samples/randstainna_params.yaml', std_hyper=-0.3, probability=1.0,
                                    distribution='normal', is_train=True)
                    ]

                    transform = transforms.Compose(transforms_list)

                    if self.channel_first:
                        img = np.transpose(x.copy(), (1, 2, 0))

                    x_augm = transform(cv2.imread(img_path))
            # plt.imshow(x_augm)
            # plt.show()

            if self.channel_first and self.data_augmentation:
                x_augm = np.transpose(x_augm, (2, 0, 1))
            else:
                x_augm = None

        #if self.stain_normalization:
        #    x = self.stain_normalization_function(x.copy())

        return x, x_augm


    def image_normalization(self, x):
        # image resize
        #TODO: Either compile opencv with GPU or change
        x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        #if random.random() > 0.5:
        #    img = skimage.util.random_noise(img, var=random.random() ** 2)
        #if random.random() > 0.5:
        #    img = img + random.random() - 0.5
        #    img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def image_transformation_noise(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        if random.random() > 0.5:
           img = skimage.util.random_noise(img, var=random.random() ** 2)
        if random.random() > 0.5:
           img = img + random.random() - 0.5
           img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

class MILDataGenerator_balanced_good(object):
    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, balanced=True ): #512
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram

        # Initialize iterator
        self._idx = 0
        self._reset()

        # For class balancing
        self.d_classes = self.dataset.y

        # Change computation of class weights depending on training mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.d_classes = self.d_classes
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]

        self.d_im_ids = list(self.dataset.D.keys())
        self.d_im_ids = [eval(i) for i in self.d_im_ids]
        self.d_im_ids = sorted(self.d_im_ids)
        self.d_len = len(self.d_im_ids)
        self.zip_gen = zip(self.d_im_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen) # dict with img IDs and corresponding label

        # One way of computing class weights
        # self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        # self.weights = 1.0 / self.class_counts
        # self.weights_t = torch.DoubleTensor(list(self.weights))


        # Another way of computing class weights
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)

        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes),
                                             y=self.d_classes)
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # List with same length as bag IDs, containing the indexes of the balanced classes
        self.balanced_class_idx = torch.multinomial(input=self.weights_t,
                                                    num_samples=self.d_len,
                                                    replacement=True)
        print(self.weights_t)
        print(self.balanced_class_idx)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Choose balanced class to find bag ID to return its images
        chosen_class_idx = self.balanced_class_idx[self._idx]
        chosen_label = self.ordered_clases[chosen_class_idx]
        images_ids_w_label = {i for i in self.d_dict if self.d_dict[i] == chosen_label}
        #print(images_ids_w_label)
        random_selected_id_for_balance = random.sample(images_ids_w_label, 1)[0]
        #print(random_selected_id_for_balance)
        images_id = self.dataset.D[str(random_selected_id_for_balance)]
        #print(images_id)

        df_row = self.dataset.data_frame.loc[self.dataset.data_frame["Patient ID"] == random_selected_id_for_balance]

        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)

        # Load images and include into the batch
        X = []
        X_augm = []
        for i in images_id:
            x, x_augm = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)

        # Update bag index iterator
        self._idx += self.batch_size

        Y = chosen_label
        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Other':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]

        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32')
        else:
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), None

class MILDataGenerator(object):

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, ): #512

        """Data Generator object for MIL.
            Process a MIL dataset object to output batches of instances and its respective labels.
        Args:
          dataset: MIL datasetdataset object.
          batch_size: batch size (number of bags). It will be usually set to 1.
          shuffle: whether to shuffle the bags (True) or not (False).
          max_instances: maximum amount of instances allowed due to computational limitations.

        Returns:
          MILDataGenerator object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram

        self._idx = 0
        self._reset()

    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)


    def __iter__(self):
        return self

    def __next__(self):

        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Get samples of data frame to use in the batch
        df_row = self.dataset.data_frame.iloc[self.indexes[self._idx]]

        # Get bag-level label GT
        Y = df_row[self.dataset.pred_column]
        Y = np.expand_dims(np.array(Y), 0)

        # Select instances from bag

        ID = list(df_row[[self.dataset.bag_id]].values)[0] #Patient IDk

        images_id = self.dataset.D[str(ID)] #Images corresponding to the Patient ID

        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)

        # QUIZÁ SE PUEDE METER AQUI, QUE SI YA HA SAMPLEADO X DE UNA CLASE QUE NO SAMPLEE MAS


        # if self.images_on_ram:
        #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #           self.dataset.X_augm[self._idx, :, :, :])
        #     # for i in images_id:
        #     #     x, x_augm = self.dataset.__getitem__(i)
        #     # # Return requested image from prellocated in memory
        #     # X =
        #     #     print(self.dataset.X[self._idx, :, :, :], self.dataset.Yglobal[self._idx, :],
        #     #           self.dataset.X_augm[self._idx, :, :, :])

        # Load images and include into the batch
        X = []
        X_augm = []
        for i in images_id:
            x, x_augm = self.dataset.__getitem__(i)
            X.append(x)
            X_augm.append(x_augm)

        # Update bag index iterator
        self._idx += self.batch_size

        if Y.shape == ():
            print("Error with Y shape")

        Y=Y[0]

        if self.pred_column == "Molecular subtype":
            # One-hot encoding
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A' or Y == 'Luminal B':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0.]
                if Y == 'Luminal B':
                    Y = [1., 0.]
                if Y == 'HER2(+)':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]
        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':

                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32')
        else:
            return np.array(X).astype('float32'), np.array(Y).astype('float32'), None


    def _reset(self):

        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


class MILDataset_w_class_perc_already_extracted_features(object):

    def __init__(self, selected_images_paths_path, dir_images, already_extracted_features_path, data_frame, pred_column, pred_mode, magnification_level, class_perc_data_frame, tissue_percentages_max, classes, bag_id='Patient ID', input_shape=(3, 224, 224),
                 data_augmentation=False, images_on_ram=False, channel_first=True, stain_normalization=False, include_background=False):

        """Dataset object for MIL.
            Dataset object which aims to organize images and labels from a dataset in the form of bags.
        Args:
          dir_images: (h, w, channels)
          data_frame: pandas dataframe with ground truth information.
                      Each bag is one raw, with 'bag_name' as identifier.
          classes: list of classes of interest in data_fame (i.e. ['G3', 'G4', 'G5'])
          input_shape: image input shape (channels first).
          data_augmentation: whether to perform data augmentation (True) or not (False).
          images_on_ram: whether to load images on ram (True) or not (False). Recommended for accelerated training.

        Returns:
          MILDataset object
        Last Updates: Julio Silva (19/03/21)
        """

        'Internal states initialization'
        self.dir_images = dir_images
        self.data_frame = data_frame
        self.classes = classes
        self.bag_id = bag_id
        self.data_augmentation = data_augmentation
        self.stain_normalization = stain_normalization
        self.input_shape = input_shape
        self.images_on_ram = images_on_ram
        self.channel_first = channel_first
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.magnification_level = magnification_level
        self.include_background = include_background
        self.selected_images_paths_path = selected_images_paths_path
        self.already_extracted_features_path = already_extracted_features_path

        if stain_normalization:
            # target_stain_patch_path = os.path.join(self.dir_images, "26", "26_0_0_0.jpg")
            target_stain_patch_path = "../data/color_norm_img_samples/01.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/55_10_12.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/19_10_38.jpg"
            #target_stain_patch_path = "../data/color_norm_img_samples/664_4608_5120.jpg"
            #target_stain_patch_path = os.path.join(self.dir_images, "71", "71_1_0_0.jpg")
            self.stain_normalization_function = Stain_Normalization(target_image_path = target_stain_patch_path,
                                                                    target_shape=(input_shape[1],input_shape[2]))

        # Recover images
        selected_images_paths = []  # List to store the file paths
        # Open the file in read mode
        with open(selected_images_paths_path, "r") as file:
            # Read each line of the file
            for line in file:
                # Remove any leading or trailing whitespace
                line = line.strip()
                # Append the file path to the list
                selected_images_paths.append(line)

        selected_images_paths = [path.replace("D:/CLAUDIO/BREAST_CANCER_DATASETS/BCNB/preprocessing_results_bien/",
                                           "../data/patches_512_fullWSIs_0/") for path in selected_images_paths]

        # Recover extracted features
        self.selected_images_features = np.load(self.already_extracted_features_path)



        # Now selected_images_paths contains the paths for ALL IDs (unilike before where it only contained the ones for training)
        # Therefore, we need to filter these paths based on the content of the "Patient_ID" column of the self.data_frame (which only contains the IDs for training)
        selected_patient_ids = set(self.data_frame['Patient ID'])
        # filter the selected image paths to keep only those with a matching patient ID
        filtered_image_paths = [path for path in selected_images_paths if
                                int(path.split('/')[-1].split("_")[0]) in selected_patient_ids]

        self.images_paths = filtered_image_paths
        print("Total patches for the selected IDs: ", str(len(self.images_paths)))

        # Now select the corresponding features for the filtered images paths
        self.filtered_image_features = [feature for path, feature in zip(selected_images_paths, self.selected_images_features) if
                                   path in filtered_image_paths]

        self.data_frame = self.data_frame[
            np.in1d(self.data_frame[self.bag_id], [img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths])]

        # Organize bags in the form of dictionary: one key clusters indexes from all instances
        self.D = dict()
        #for i, item in enumerate([ID.split('_')[0] for ID in self.images]):
        for i, item in enumerate([img_path.split('/')[-1].split("_")[0] for img_path in self.images_paths]):
            if item not in self.D:
                self.D[item] = [i]
            else:
                self.D[item].append(i)

        self.y = self.data_frame[self.pred_column].values
        self.indexes = np.arange(len(self.images_paths))


        # Preallocate Images on RAM
        if self.images_on_ram:
            # Pre-allocate images
            self.X = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for images
            self.X_augm = np.zeros((len(self.indexes), input_shape[0], input_shape[1], input_shape[2]), dtype=np.float32) # Create empty array for augmented images
            #self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)

            if self.include_background:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes) + 1), dtype=np.float32)
            else:
                self.Yglobal = np.ones((len(self.indexes), len(self.classes)), dtype=np.float32) # Create empty array for labels

            #if self.dataframe_instances is not False:
                #self.y_instances = -1 * np.ones((len(self.indexes), 4))

            # Load, and normalize images
            print('[INFO]: Loading images on RAM memory ...')
            for i in tqdm(np.arange(len(self.indexes)), leave=True, position=0):
                #print(str(i) + '/' + str(len(self.indexes)), end='\r')

                img_path = self.images_paths[self.indexes[i]]
                #Derive Patient ID
                img_patient_id = int(img_path.split("/")[-1].split("_")[0])
                # Load image
                x_org = Image.open(img_path)

                # # Select magnification
                # if self.magnification_level == "20x":
                #     x = x_org
                # elif self.magnification_level == "10x":
                #     x = x_org.resize((256, 256))

                x = x_org

                # Apply Macenko Stain Normalization if wanted
                if self.stain_normalization:
                    try:
                        #plt.imshow(x)
                        #plt.show()
                        x = self.stain_normalization_function(x)
                        #plt.imshow(x)
                        #plt.show()
                    except RuntimeError:
                        print("Full black patch found! Check patient's images...")
                        print(img_path)

                x = np.asarray(x, dtype='uint8')

                # Data Augmentation
                if self.data_augmentation:
                    # x_augm = self.image_transformation(x.copy())
                    if self.data_augmentation == "non-spatial":
                        transform = A.Compose([
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.Transpose(p=0.5),
                            A.GridDistortion(p=0.5),
                            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5)])

                        x_augm = transform(image=x.copy())["image"]
                        # Normalization
                        x_augm = self.image_normalization(x_augm)  # Normalize image

                        # if self.channel_first:
                        #     x_augm = np.transpose(x_augm.copy(), (1, 2, 0))
                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                    if self.data_augmentation == "randstainna":
                        #### calling the randstainna
                        transforms_list = [
                            RandStainNA(yaml_file='../data/color_norm_img_samples/randstainna_params.yaml',
                                        std_hyper=-0.3, probability=1.0,
                                        distribution='normal', is_train=True)
                        ]

                        transform = transforms.Compose(transforms_list)

                        if self.channel_first:
                            img = np.transpose(x.copy(), (1, 2, 0))

                        x_augm = transform(cv2.imread(img_path))
                        x_augm = self.image_normalization(x_augm)

                        # Normalization
                        x = self.image_normalization(x)  # Normalize image

                # plt.imshow(x_augm)
                # plt.show()

                # if self.channel_first and self.data_augmentation:
                #     x_augm = np.transpose(x_augm, (2, 0, 1))
                else:
                    # Normalization
                    x = self.image_normalization(x)  # Normalize image
                    x_augm = None

                    #print(type(x), x.shape)
                #x = Image.open(os.path.join(self.dir_images, ID))
                # x = np.asarray(x)
                # # Normalization
                # x = self.image_normalization(x)
                self.X[self.indexes[i], :, :, :] = x
                self.X_augm[self.indexes[i], :, :, :] = x_augm

                #Get the GT Label of the image
                y_label = self.data_frame[self.data_frame["Patient ID"] == img_patient_id][self.pred_column].to_string(index=False)



                if self.pred_column == "Molecular subtype":
                    if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [0., 1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 0., 1.]
                    if self.pred_mode == "LUMINALSvsHER2vsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [0., 1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 0., 1.]
                    if self.pred_mode == "OTHERvsTNBC":
                        #One-hot encoding
                        if y_label == 'Luminal A':
                            y_label = [1., 0.]
                        if y_label == 'Luminal B':
                            y_label = [1., 0.]
                        if y_label == 'HER2(+)':
                            y_label = [1., 0.]
                        if y_label == 'Triple negative':
                            y_label = [0., 1.]

                elif self.pred_column == "ALN status":
                    if y_label == 'N0':
                        y_label = [1., 0., 0.]
                    if y_label == 'N+(1-2)':
                        y_label = [0., 1., 0.]
                    if y_label == 'N+(>2)':
                        y_label = [0., 0., 1.]
                elif self.pred_column == "HER2 Expression":
                    if y_label == '0':
                        y_label = [1., 0., 0., 0.]
                    if y_label == '1+':
                        y_label = [0., 1., 0., 0.]
                    if y_label == '2+':
                        y_label = [0., 0., 1., 0.]
                    if y_label == '3+':
                        Y = [0., 0., 0., 1.]

                #self.Yglobal[self.indexes[i], 1:] = self.data_frame[classes][self.data_frame[self.bag_id] == int(self.images[i].split('/')[-1].split("_")[0])]

                if self.include_background:
                    self.Yglobal[self.indexes[i], 1:] = y_label
                else:
                    self.Yglobal[self.indexes[i], :] = y_label

            print('[INFO]: Images loaded')


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indexes)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_path = self.images_paths[self.indexes[index]]
        img_features = self.selected_images_features[self.indexes[index]]

        return img_features


    def image_normalization(self, x):
        # image resize
        x = cv2.resize(x, (self.input_shape[1], self.input_shape[2]))
        # intensity normalization
        x = x / 255.0
        # channel first
        if self.channel_first:
            x = np.transpose(x, (2, 0, 1))
        # numeric type
        x.astype('float32')
        return x

    def image_transformation(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        #if random.random() > 0.5:
        #    img = skimage.util.random_noise(img, var=random.random() ** 2)
        #if random.random() > 0.5:
        #    img = img + random.random() - 0.5
        #    img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def image_transformation_noise(self, img):

        if self.channel_first:
            img = np.transpose(img, (1, 2, 0))

        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        if random.random() > 0.5:
            angle = random.random() * 60 - 30
            img = skimage.transform.rotate(img, angle)
        if random.random() > 0.5:
           img = skimage.util.random_noise(img, var=random.random() ** 2)
        if random.random() > 0.5:
           img = img + random.random() - 0.5
           img = np.clip(img, 0, 1)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))

        return img

class MILDataGenerator_balanced_good_already_extracted_features(object):
    def __len__(self):
        N = len(self.indexes)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, images_on_ram, batch_size=1,  shuffle=False, max_instances=50, num_workers=0, balanced=True ): #512
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset.data_frame))
        self.max_instances = max_instances
        self.images_on_ram = images_on_ram

        # Initialize iterator
        self._idx = 0
        self._reset()

        # For class balancing
        self.d_classes = self.dataset.y

        # Change computation of class weights depending on training mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.d_classes = self.d_classes
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]

        self.d_im_ids = list(self.dataset.D.keys())
        self.d_im_ids = [eval(i) for i in self.d_im_ids]
        self.d_im_ids = sorted(self.d_im_ids)
        self.d_len = len(self.d_im_ids)
        self.zip_gen = zip(self.d_im_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen) # dict with img IDs and corresponding label

        # One way of computing class weights
        # self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        # self.weights = 1.0 / self.class_counts
        # self.weights_t = torch.DoubleTensor(list(self.weights))


        # Another way of computing class weights
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)

        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes),
                                             y=self.d_classes)
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # List with same length as bag IDs, containing the indexes of the balanced classes
        self.balanced_class_idx = torch.multinomial(input=self.weights_t,
                                                    num_samples=self.d_len,
                                                    replacement=True)
        print(self.weights_t)
        print(self.balanced_class_idx)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset.data_frame):
            self._reset()
            raise StopIteration()

        # Choose balanced class to find bag ID to return its images
        chosen_class_idx = self.balanced_class_idx[self._idx]
        chosen_label = self.ordered_clases[chosen_class_idx]
        images_ids_w_label = {i for i in self.d_dict if self.d_dict[i] == chosen_label}
        #print(images_ids_w_label)
        random_selected_id_for_balance = random.sample(images_ids_w_label, 1)[0]
        #print(random_selected_id_for_balance)
        images_id = self.dataset.D[str(random_selected_id_for_balance)]
        #print(images_id)

        df_row = self.dataset.data_frame.loc[self.dataset.data_frame["Patient ID"] == random_selected_id_for_balance]

        # # Memory limitation of patches in one slide
        if len(images_id) > self.max_instances:
            images_id = random.sample(images_id, self.max_instances)

        # Load images and include into the batch
        images_features = []
        for i in images_id:
            image_features = self.dataset.__getitem__(i)
            images_features.append(image_features)

        # Update bag index iterator
        self._idx += self.batch_size

        Y = chosen_label
        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal A':
                    Y = [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    Y = [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 0., 1.]
            if self.pred_mode=="LUMINALSvsHER2vsTNBC":
                # One-hot encoding
                if Y == 'Luminal':
                    Y = [1., 0., 0.]
                if Y == 'HER2(+)':
                    Y = [0., 1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                # One-hot encoding
                if Y == 'Other':
                    Y = [1., 0.]
                if Y == 'Triple negative':
                    Y = [0., 1.]

        elif self.pred_column == "Histological grading":
            # One-hot encoding
            if Y == 1.:
                Y = [1., 0., 0.]
            if Y == 2.:
                Y = [0., 1., 0.]
            if Y == 3.:
                Y = [0., 0., 1.]
        elif self.pred_column == "ALN status":
            if Y == 'N0':
                Y = [1., 0., 0.]
            if Y == 'N+(1-2)':
                Y = [0., 1., 0.]
            if Y == 'N+(>2)':
                Y = [0., 0., 1.]
        elif self.pred_column == "HER2 Expression":
            if Y == '0':
                Y = [1., 0., 0., 0.]
            if Y == '1+':
                Y = [0., 1., 0., 0.]
            if Y == '2+':
                Y = [0., 0., 1., 0.]
            if Y == '3+':
                Y = [0., 0., 0., 1.]
        elif self.pred_column == "Ki67":
            if "%" in Y:
                if "-" in Y:
                    Y = Y.split("-")[0]
                elif "＞" in Y:
                    Y = Y.split("＞")[1]
                elif ">" in Y:
                    Y = Y.split(">")[1]
                elif "＜" in Y:
                    Y = Y.split("＜")[1]
                elif "<" in Y:
                    Y = Y.split("<")[1]

                Y = float(Y.strip('%')) / 100

                if float(Y) < 0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y) >= 0.14: #High Proliferation
                    Y = [0., 1.]

            else:
                if float(Y)<0.14: #Low Proliferation
                    Y = [1., 0.]
                elif float(Y)>=0.14: #High Proliferation
                    Y = [0., 1.]

        if self.dataset.data_augmentation==True or self.dataset.data_augmentation=="non-spatial":
            return np.array(image_features).astype('float32'), np.array(Y).astype('float32'), np.array(X_augm).astype('float32')
        else:
            return np.array(image_features).astype('float32'), np.array(Y).astype('float32')

