import numpy as np
import torch

class AP(object):

	IOU_THRESH = 0.5

	def __init__(self, dataset, decode, filt, restore, device, use_07_metric=True):

		self.num_classes = len(dataset.CLASSES)
		self.k_true = dataset.max_objects
		self.easy_objects = dataset.easy_objects

		self.decode = decode # a decoder that transforms raw network input into bboxes with scores and labels
		self.k_pred = decode.k

		self.filt = filt # a callable that filters bboxes such as NMS

		self.restore = restore # a callable that transforms resized coordinates to actual coordinates in images

		self.device = device

		self.use_07_metric = use_07_metric

		if self.use_07_metric:
			self.t = torch.arange(start=0, end=1.1, step=0.1, dtype=torch.float, requires_grad=False)

		self.arange_true = torch.arange(self.k_true, dtype=torch.int64, requires_grad=False).to(self.device)
		self.arange_pred = torch.arange(self.k_pred, dtype=torch.int64, requires_grad=False).to(self.device) + 1

		self.reset()

	def accumulate(self, outputs, targets):

		(scores_pred, labels_pred, xmins_pred, ymins_pred, xmaxs_pred, ymaxs_pred) = self.decode(outputs)
		boxes_pred = torch.cat([xmins_pred[...,None], ymins_pred[...,None], xmaxs_pred[...,None], 
			ymaxs_pred[...,None], labels_pred[...,None], scores_pred[...,None]], axis=-1)
		
		boxes_pred, evals_pred = self.filt(boxes_pred)
		xmins_true = targets['xmins'].to(self.device)
		ymins_true = targets['ymins'].to(self.device)
		xmaxs_true = targets['xmaxs'].to(self.device)
		ymaxs_true = targets['ymaxs'].to(self.device)
		labels_true = targets['labels'].to(self.device)
		evals_true = targets['easy_flags'].to(self.device)		
		boxes_true = torch.cat([xmins_true[...,None], ymins_true[...,None], xmaxs_true[...,None],
			ymaxs_true[...,None], labels_true[...,None]], axis=-1)

		image_widths = targets['width'].to(self.device)
		image_heights = targets['height'].to(self.device)

		boxes_pred, boxes_true = self.restore(boxes_pred, boxes_true, image_widths, image_heights)

		ious = self._ious(boxes_pred, boxes_true)

		classes_pred = self._expand_pred(boxes_pred[...,4]).int()
		classes_true = self._expand_true(boxes_true[...,4]).int()

		class_mask = classes_pred == classes_true
		# We set IoUs to 0 where the predicted bounding box and the
        # ground truth bounding box have different class labels. By
        # doing this we ensure each object class has the correct
        # tp and fp statistics
		ious *= class_mask.float()

		evals_pred_mask = self._expand_pred(evals_pred)
		# We set IoUs of discarded detections to 0. By doing this they
        # will be considered as false positives when in fact they should
        # be ignored. We will correct this later.
		ious *= (~evals_pred_mask).float()
		# We mark predicted bounding boxes who do not have an IoU
        # greater than the threshold with any ground truth bounding box
        # of the same class as false positives.
		under_iou_mask = torch.max(ious, 1)[0] < self.IOU_THRESH
        # This tensor has the same as boxes_pred[...,0]. Each element
        # maps to a predicted bounding box. We would like to divide
        # the predicted bounding boxes into 3 categories, true positives,
        # false positives and the ignored. For true positives, we
        # set the corresponding elements to 2; for false positives, we
        # set to 1; for the ignored, we set to 0.  
		detections = torch.zeros_like(boxes_pred[...,0], dtype=torch.uint8)
		detections[under_iou_mask] = 1
		# Discarded detections were marked as false positives. Here we
        # mark them as ignored.
		detections[evals_pred] = 0

		evals_true_mask = self._expand_true(evals_true)
		# We set IoUs to 0 where the ground truth bounding box is marked
        # as "difficult". By doing this we are left with predicted bounding
        # boxes that (1) have an IoU that is greater than the threshold, and
        # (2) are not matched with "difficult" ground truth bounding boxes.
        # These predicted bounding boxes are either true positive or false
        # positives
		ious *= evals_true_mask.float()
		# We have an "idxs" tensor whose shape is batch_size * k_pred. Each
        # element is the index of the ground truth bounding box with which
        # the predicted bounding boxes are matched. For example, if the ith
        # predicted bounding box in image b matches the jth ground truth
        # bounding box in image b, then idxs[b][i] = j. 
		ious, idxs = torch.max(ious, 1)

		over_iou_mask = ious > self.IOU_THRESH
		# We first mark all the remaining predicted bounding boxes as false
        # positives. Next we will find the true positives among them.
		detections[over_iou_mask] = 1
		# We have to sort the predicted bounding boxes in each image
        # based on their confidence score. This may have already been
        # done in the decoding operation

        # After sorting the predicted bounding boxes, we have to determine
        # which ones are true positives. The easiest method is to loop through
        # each predicted bounding box in each image. We check the index
        # of the matched ground truth bounding box. If an index has not been
        # seen before we mark the predicted bounding box as true positive;
        # otherwise the predicted bounding box is a false positive and we do
        # nothing.

        # To check every predicted bounding box in parallel, we first need to
        # expand the "idxs" tensor such that its shape is
        # batch_size * k_pred * k_true. Next we need to transform "idxs" to
        # its one-hot encoding. We do this by checking if idxs[b][i][j] == j.
		idxs_one_hot = self._expand_pred(idxs).permute((0,2,1)) == self.arange_true
		# We set idxs_one_hot to 0 where the predicted bounding box is
        # either ignored or has an IoU less than the threshold
		idxs_one_hot *= self._expand_pred(over_iou_mask).permute((0,2,1))
		# The problem we are facing now is that multiple predicted
        # bounding boxes can match with the same ground truth bounding
        # box. For example, in image b, the i_1st predicted bounding
        # box and the i_2nd predicted bounding box can both match
        # with the jth predicted bounding box. In other words,
        # idxs_one_hot[b][i_1][j] = idxs_one_hot[b][i_2][j] = 1. However,
        # only the i_1st predicted bounding box is a true positive.
        # To solve this problem, we first permute idxs_one_hot so that
        # its shape is batch_size x k_true x k_pred and then we
        # multiply idxs_one_hot_permuted[b][j][i] with i + 1. The reason we
        # multiply idxs_one_hot_permuted[b][j][i] with i + 1 instead of i
        # is because i starts at 0 and 0 x 0 = 0.
		idxs_one_hot = idxs_one_hot.permute((0,2,1)).long() * self.arange_pred
		# We set idxs_one_hot to a sentinel value where idxs_one_hot is 0.
        # The sentinel value must be greater than the maximum value of
        # arange_pred.
		idxs_one_hot[idxs_one_hot == 0] = self.k_pred + 1
		# now we can determine if idxs_one_hot[b][j][i] corresponds to a
        # true positive by checking (1) if idxs_one_hot[b][j][i] ==
        # min(idx_one_hot[b][j]), and (2) if idxs_one_hot[b][j][i] < sentinel
		tp_mask = ((idxs_one_hot == idxs_one_hot.min(2, keepdim=True)[0]) * (idxs_one_hot < self.k_pred + 1))
		tp_mask = tp_mask.max(1)[0]

		detections[tp_mask] = 2
		# We can now find true positives and false positives because they
        # have non zeros values

		evals_mask = detections > 0
        # We append scores of the kept predicted bounding boxes so that
        # we can sort predicted bounding boxes of the entire dataset.
		self.scores.append(scores_pred[evals_mask].cpu())
		# We append labels of the kept predicted bounding boxes so that
        # we can calculate precision and recall for each class
		self.labels.append(labels_pred[evals_mask].cpu())
		# Earlier we marked true postives as 2 and false positives as 1
        # now we subtract by 1 so that true positives are 1 and
        # false positives are 0. In other words, false positives = ~
        # true positives or false positives = 1 - true positives
		self.true_positives.append((detections[evals_mask] - 1).bool().cpu())

	def compute(self):

		with torch.no_grad():

			scores = torch.cat(self.scores)
			labels = torch.cat(self.labels)
			true_positives = torch.cat(self.true_positives)

			scores, idxs = torch.sort(scores, descending=True)
			labels = labels[idxs]
			true_positives = true_positives[idxs]

			mean_ap = 0

			for k in range(self.num_classes):

				class_true_positives = true_positives[labels == k]
				class_false_positives = ~class_true_positives

				class_true_positives = torch.cumsum(class_true_positives, 0)
				class_false_positives = torch.cumsum(class_false_positives, 0)

				precisions = class_true_positives.float() / (class_true_positives + class_false_positives).float()
				recalls = class_true_positives.float() / self.easy_objects[k]

				if self.use_07_metric:

					ap = precisions.unsqueeze(1).expand((-1,11)) * (recalls.unsqueeze(1).expand((-1,11)) >= self.t).float()
					ap = ap.max(0)[0].sum() / 11

					mean_ap += ap / 20

				else:

					mrec = np.concatenate(([0.], recalls.cpu().numpy(), [1.]))
					mpre = np.concatenate(([0.], precisions.cpu().numpy(), [0.]))

					for i in range(mpre.size - 1, 0, -1):

						mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

					i = np.where(mrec[1:] != mrec[:-1])[0]

					mean_ap += np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) / 20

			return mean_ap.item()

	def reset(self):
		self.scores = []
		self.labels = []
		self.true_positives = []


	def _ious(self, boxes_pred, boxes_true):
		ixmins = torch.max(self._expand_pred(boxes_pred[...,0]), self._expand_true(boxes_true[...,0]))
		iymins = torch.max(self._expand_pred(boxes_pred[...,1]), self._expand_true(boxes_true[...,1]))
		ixmaxs = torch.min(self._expand_pred(boxes_pred[...,2]), self._expand_true(boxes_true[...,2]))
		iymaxs = torch.min(self._expand_pred(boxes_pred[...,3]), self._expand_true(boxes_true[...,3]))
		iws = ixmaxs - ixmins + 1
		ihs = iymaxs - iymins + 1
		iws[iws < 0] = 0
		ihs[ihs < 0] = 0
		iareas = iws * ihs
		areas_pred = self._areas(boxes_pred)
		areas_true = self._areas(boxes_true)
		uareas = self._expand_pred(areas_pred) + self._expand_true(areas_true) - iareas
		return iareas / uareas

	def _expand_pred(self, x):
		return x.unsqueeze(1).expand((-1,self.k_true,-1))

	def _expand_true(self, x):
		return x.unsqueeze(2).expand((-1,-1,self.k_pred))

	def _areas(self, boxes):
		ws = boxes[...,2] - boxes[...,0] + 1
		hs = boxes[...,3] - boxes[...,1] + 1
		return ws * hs
