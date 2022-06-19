import torch
import torch.nn.functional as F

class DecodeCenter(object):

    def __init__(self, input_size, output_stride, k):

        self.input_size = input_size
        self.output_stride = output_stride
        self.k = k

    def __call__(self, outputs):

        scores = outputs['hm'].sigmoid_()
        scores *= (scores == F.max_pool2d(scores, 3, stride=1, padding=1)).float()
        sizes = outputs['wh'].sigmoid_() * self.input_size / self.output_stride
        offsets = outputs['reg'].tanh_()

        batch_size, num_classes, output_height, output_width = scores.shape

        output_area = output_height * output_width

        scores, idxs = torch.topk(scores.contiguous().view(batch_size, -1), self.k, dim=1)

        #labels = idxs // output_area
        labels = torch.div(idxs, output_area, rounding_mode='floor')

        output_volume = labels * output_area

        spatial_idxs = idxs - output_volume

        col_radii = torch.gather(sizes[:,0].view(batch_size,-1), 1, spatial_idxs) / 2
        row_radii = torch.gather(sizes[:,1].view(batch_size,-1), 1, spatial_idxs) / 2

        col_offsets = torch.gather(offsets[:,0].view(batch_size,-1), 1, spatial_idxs)
        row_offsets = torch.gather(offsets[:,1].view(batch_size,-1), 1, spatial_idxs)

        col_idxs = (idxs - output_volume) % output_width
        #row_idxs = (idxs - output_volume) // output_width
        row_idxs = torch.div(idxs - output_volume, output_width, rounding_mode='floor')

        col_centers = col_idxs.float() + col_offsets
        row_centers = row_idxs.float() + row_offsets

        xmins = (col_centers - col_radii) * self.output_stride
        ymins = (row_centers - row_radii) * self.output_stride
        xmaxs = (col_centers + col_radii) * self.output_stride
        ymaxs = (row_centers + row_radii) * self.output_stride

        return scores, labels, xmins, ymins, xmaxs, ymaxs

class NMS(object):

    def __init__(self, skip=True):
        
        self.skip = skip

    def __call__(self, boxes):
        
        if self.skip:
            
            return boxes, torch.zeros_like(boxes[...,0], dtype=torch.bool)
        
        else:
            # CenterNet does not require NMS. If NMS is required, suppressed bboxes
            # need to be kept. Use a binary mask to indicate if bboxes are suppressed.
            raise NotImplementedError

class RestoreCoords(object):

    def __init__(self, input_size):
        
        self.input_size = input_size

    def __call__(self, boxes_pred, boxes_true, image_widths, image_heights):
        
        image_sizes = torch.max(image_widths, image_heights)
        scales = image_sizes / self.input_size
        col_pads = (image_sizes - image_widths) / 2
        row_pads = (image_sizes - image_heights) / 2

        def restore_coords(boxes):
            boxes[...,0] = boxes[...,0] * scales - col_pads
            boxes[...,1] = boxes[...,1] * scales - row_pads
            boxes[...,2] = boxes[...,2] * scales - col_pads
            boxes[...,3] = boxes[...,3] * scales - row_pads
            return boxes

        return restore_coords(boxes_pred), restore_coords(boxes_true)
