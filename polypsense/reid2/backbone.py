import torch
from PIL import Image

from main import MyModel as RtdetrModel
from main import build_processor, load_config
from polypsense.sfe.model import MyModel as SimclrModel
from polypsense.mve.model import MultiViewEncoder
from torchvision.transforms import v2


class Backbone:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

    def load(self, device="cuda"):
        pass

    def forward(self, image_path, bbox):
        pass


class RtdetrBackbone(Backbone):
    def load(self, architecture="rtdetr_rn18", device="cuda", **kwargs):
        config = load_config(
            [
                f"polypsense/zoo/rtdetr/config/{architecture}.yml",
                "polypsense/zoo/rtdetr/config/dataset/ucd.yml",
                "polypsense/zoo/rtdetr/config/run/test.yml",
            ]
        )

        # detector = MyModel.load_from_checkpoint(checkpoint).eval().to(device)
        detector = RtdetrModel(config)
        detector.load_state_dict(torch.load(self.ckpt_path)["state_dict"])
        detector.eval()
        detector.to(device)

        processor = build_processor(detector.config)

        self.detector = detector
        self.processor = processor

    def forward(self, image_path, bbox):
        im = Image.open(image_path).convert("RGB")
        detections = self._get_detections(im)
        feats = self._get_frame_feats(detections)
        agg_feats = self._aggregate_frame_feats(feats)
        return agg_feats

    def _get_detections(self, image):
        class SaveDecoderLastHiddenStateHook:
            def __call__(self, module, input, output):
                self.last_hidden_state = output

        device = self.detector.device

        target_sizes = torch.tensor(image.size).to(device).unsqueeze(0)  # [b, 2]
        image = self.processor.pre_process_image(image).to(device)  # [c, h, w]

        # create and register hook to save the last hidden state of the decoder

        feats_hook = SaveDecoderLastHiddenStateHook()
        self.detector.rtdetr.decoder.decoder.layers[2].register_forward_hook(feats_hook)

        # forward pass

        with torch.no_grad():
            inp = image.unsqueeze(0)  # [b, c, h, w]
            out = self.detector(inp)

        # post processing

        # return a list of dict with boxes sorted by score (descending) in xyxy
        # format, e.g. [{"boxes": [n, 4], "scores": [n]}]
        detections = self.processor.post_process_object_detection(out, target_sizes)
        # since the forward receive one image, we remove the batch dimension
        detections = detections[0]
        feats = feats_hook.last_hidden_state[0]

        # move back to cpu and convert to numpy
        detections = {
            "boxes": detections["boxes"].cpu(),
            "scores": detections["scores"].cpu(),
            "feats": feats.cpu(),
        }

        return detections

    def _get_frame_feats(self, detection, score_thresh=0.2):
        feats = detection["feats"]
        scores = detection["scores"]
        boxes = detection["boxes"]

        # filter out low score detections
        mask = scores > score_thresh
        feats = feats[mask]

        return feats

    def _aggregate_frame_feats(self, feats):
        if feats.shape[0] == 0:
            return torch.zeros(1, feats.shape[1])
        return feats.mean(dim=0).unsqueeze(0)


class SimclrBackbone(Backbone):
    def load(self, device="cuda", **kwargs):
        model = SimclrModel.load_from_checkpoint(self.ckpt_path)
        model.eval()
        model.to(device)
        self.model = model

    def forward(self, image_path, bbox):
        im_size = 232
        transform = v2.Compose(
            [
                v2.Resize(size=[im_size] * 2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        device = self.model.device
        im = Image.open(image_path).convert("RGB")
        im = im.crop(bbox)

        image = transform(im).to(device).unsqueeze(0)

        with torch.no_grad():
            feats = self.model(image)

        return feats


class MultiViewEncoderBackbone(Backbone):
    def load(self, device="cuda", **kwargs):
        # TODO: remove commented code when legacy MVE checkpoints become unused.
        #
        # We need this workaround code because some MVE checkpoints were saved
        # without SFE hparams. Thus, we need to manually instanciate it and pass
        # the instance to `load_from_checkpoint`.
        #
        # sfe = SimclrModel.load_from_checkpoint("/home/lparolar/Projects/polypsense/wandb_logs/simclr/oex79j1n/checkpoints/epoch=8-step=28980-val_acc_top1=59.83.ckpt", map_location="cpu")
        # sfe_state_dict = torch.load(self.ckpt_path, map_location="cpu")["state_dict"]
        # sfe_state_dict = {k: v for k, v in sfe_state_dict.items() if "sfe." in k}
        # sfe_state_dict = {k.replace("sfe.", ""): v for k, v in sfe_state_dict.items()}
        # model = MultiViewEncoder.load_from_checkpoint(self.ckpt_path, map_location="cpu", sfe=sfe)
        # model.sfe.load_state_dict(sfe_state_dict)

        model = MultiViewEncoder.load_from_checkpoint(self.ckpt_path, map_location="cpu")
        model.eval()
        model.to(device)
        self.model = model

    def forward(self, images, bboxes):
        # images: list of image paths
        # bboxes: list of bounding boxes

        im_size = 232
        transform = v2.Compose(
            [
                v2.Resize(size=[im_size] * 2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        device = self.model.device

        preprocess = lambda image, bbox: Image.open(image).convert("RGB").crop(bbox)

        images = [preprocess(image, bbox) for image, bbox in zip(images, bboxes)]
        images = [transform(im).to(device) for im in images]

        x = torch.stack(images, dim=0).unsqueeze(0)  # x [1, s, c, h, w]

        with torch.no_grad():
            feats = self.model(x)

        return feats


def get_backbone(model_type, ckpt_path, **kwargs):
    return {
        "rtdetr": RtdetrBackbone,
        "simclr": SimclrBackbone,
        "mve": MultiViewEncoderBackbone,
    }[model_type](ckpt_path, **kwargs)
