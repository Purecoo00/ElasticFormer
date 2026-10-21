# PANDA evaluation adapted from liwenxi/SparseFormer, Apache-2.0.
from mmdet.datasets.api_wrappers import COCOeval as MM_COCOEval
import numpy as np


class PANDAEval(MM_COCOEval):
    """COCO evaluation metric for PANDA"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the PANDA limit of 500 detections for each reported metric.
        self.params.areaRng = [[0**2, 1e5**2], [0**2, 96**2], [96**2, 288**2], [288**2, 1e5**2]]
        self.params.maxDets = [500, 500, 500]

    def summarize(self):
        """Report AP/AR using PANDA area ranges and detection limits."""

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=500):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
                if iouThr is None
                else "{:0.2f}".format(iouThr)
            )

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1, maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=0.5, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, iouThr=0.5, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=0.5, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        else:
            raise ValueError("PANDA evaluation supports bbox and segm IoU only")
        self.stats = summarize()
