import torch.utils.data as torch_data


class DatasetTemplate(torch_data.Dataset):
    def __init__(
        self,
        dataset_cfg=None,
        class_names=None,
        training=True,
        root_path=None,
        logger=None,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = (
            root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        )

        self.data_augmentor = (
            DataAugmentor(
                self.root_path,
                self.dataset_cfg.DATA_AUGMENTOR,
                self.class_names,
                logger=self.logger,
            )
            if self.training
            else None
        )

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def generate_prediction_dicts(
        self, batch_dict, pred_dicts, class_names, output_path=None
    ):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    # def prepare_data(self, data_dict):
    #     """
    #     Args:
    #         data_dict:
    #             points: optional, (N, 3 + C_in)
    #             gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
    #             gt_names: optional, (N), string
    #             ...

    #     Returns:
    #         data_dict:
    #             frame_id: string
    #             points: (N, 3 + C_in)
    #             gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
    #             gt_names: optional, (N), string
    #             use_lead_xyz: bool
    #             voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
    #             voxel_coords: optional (num_voxels, 3)
    #             voxel_num_points: optional (num_voxels)
    #             ...
    #     """
    #     if self.training:
    #         assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
    #         gt_boxes_mask = np.array(
    #             [n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_
    #         )

    #         if 'calib' in data_dict:
    #             calib = data_dict['calib']
    #         if self.augment_randomly:
    #             data_dict = self.data_augmentor.forward_randomly(
    #                 data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
    #             )
    #         else:
    #             data_dict = self.data_augmentor.forward(
    #                 data_dict={**data_dict, 'gt_boxes_mask': gt_boxes_mask}
    #             )

    #         if 'calib' in data_dict:
    #             data_dict['calib'] = calib
    #     if data_dict.get('gt_boxes', None) is not None:
    #         selected = common_utils.keep_arrays_by_name(
    #             data_dict['gt_names'], self.class_names
    #         )
    #         data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
    #         data_dict['gt_names'] = data_dict['gt_names'][selected]
    #         gt_classes = np.array(
    #             [self.class_names.index(n) + 1 for n in data_dict['gt_names']],
    #             dtype=np.int32,
    #         )
    #         gt_boxes = np.concatenate(
    #             (data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)),
    #             axis=1,
    #         )
    #         data_dict['gt_boxes'] = gt_boxes

    #         if data_dict.get('gt_boxes2d', None) is not None:
    #             data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

    #     if data_dict.get('points', None) is not None:
    #         data_dict = self.point_feature_encoder.forward(data_dict)

    #     data_dict = self.data_processor.forward(data_dict=data_dict)

    #     if self.training and len(data_dict['gt_boxes']) == 0:
    #         new_index = np.random.randint(self.__len__())
    #         return self.__getitem__(new_index)

    #     data_dict.pop('gt_names', None)

    #     return data_dict
