#!/usr/bin/env python

import argparse
import multiprocessing

import jsk_data


def download_data(*args, **kwargs):
    p = multiprocessing.Process(target=jsk_data.download_data, args=args, kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="quiet", action="store_false")
    args = parser.parse_args()
    args.quiet

    PKG = "tracking_ros"

    # segment anything
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_b.pth",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        md5="01ec64d29a2fca3f0661936605ae66f8",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_l.pth",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        md5="0b3195507c641ddb6910d2bb5adee89c",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_h.pth",
        url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        md5="4b8939a88964f0f4ff5f5b2642c598a6",
    )

    # segment anything hq
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_b_hq.pth",
        url="https://drive.google.com/uc?id=11yExZLOve38kRZPfRx_MRxfIAKmfMY47",
        md5="c6b8953247bcfdc8bb8ef91e36a6cacc",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_l_hq.pth",
        url="https://drive.google.com/uc?id=1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G",
        md5="08947267966e4264fb39523eccc33f86",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/sam_vit_h_hq.pth",
        url="https://drive.google.com/uc?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8",
        md5="3560f6b6a5a6edacd814a1325c39640a",
    )

    # mobile sam
    download_data(
        pkg_name=PKG,
        path="trained_data/sam/mobile_sam.pth",
        url="https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt",
        md5="f3c0d8cda613564d499310dab6c812cd",
    )

    # cutie
    download_data(
        pkg_name=PKG,
        path="trained_data/cutie/cutie-base-mega.pth",
        url="https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth",
        md5="a6071de6136982e396851903ab4c083a",
    )

    # DEVA
    download_data(
        pkg_name=PKG,
        path="trained_data/deva/DEVA-propagation.pth",
        url="https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth",
        md5="a614cc9737a5b4c22ecbdc93e7842ecb",
    )

    # grounding dino
    download_data(
        pkg_name=PKG,
        path="trained_data/groundingdino/groundingdino_swint_ogc.pth",
        url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        md5="075ebfa7242d913f38cb051fe1a128a2",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/groundingdino/GroundingDINO_SwinT_OGC.py",
        url="https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py",
        md5="bdb07fc17b611d622633d133d2cf873a",
    )

    # VLPart
    # r50
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_voc.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_voc.pth",
        md5="70ce256b6f05810b5334f41f2402eb09",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_coco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_coco.pth",
        md5="1684ce2c837abfba9e884a30524df5bc",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis.pth",
        md5="42da3fccb40901041c0ee2fdc5683fd3",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_partimagenet.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_partimagenet.pth",
        md5="4def166e3d1fd04a54d05992df32500c",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_pascalpart.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_pascalpart.pth",
        md5="e5671ffdb282eb903173520a7e3c035a",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_paco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_paco.pth",
        md5="f8d5032b8e8caef5423dc8ce69f6e04c",
    )

    # r50 with joint training
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis_paco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco.pth",
        md5="c2c80112b3d6ac6109053b6e5986b698",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis_paco_pascalpart.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart.pth",
        md5="854769b7901728fc9d1ce84189bb2e7b",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis_paco_pascalpart_partimagenet.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet.pth",
        md5="6783141824dd228f559063f52ec7aadd",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis_paco_pascalpart_partimagenet_in.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_in.pth",
        md5="0943cefccedad36598bc1b32ae52c0e2",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/r50_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/r50_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        md5="2e8999eb3a8fabe9b5191bdd0a094f75",
    )

    # swinbase-cascade
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_voc.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_voc.pth",
        md5="5e0a17bdfd0517ea7de14de6bcbea121",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_coco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_coco.pth",
        md5="f18c6e2aa099118afb07820ff970c5fe",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis.pth",
        md5="a909f64862104121817ba89784efd1c9",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_partimagenet.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_partimagenet.pth",
        md5="ab757d04bd1c48ba8e4995f55d0fee54",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_pascalpart.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_pascalpart.pth",
        md5="0119fa74970c32824c5806ea86658126",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_paco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_paco.pth",
        md5="157dd2d35400a1aa16c5dde8d17355b0",
    )

    # swinbase-cascade with joint training
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis_paco.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco.pth",
        md5="7969a7dfd9a5be8a7e88c7c174d54db2",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis_paco_pascalpart.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart.pth",
        md5="5f5d1dd844eee2a3ccacc74f176b2bc1",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth",
        md5="b0a1960530b4150ccb9c8d84174fa985",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth",
        md5="013f21eb55b8ec0e5f3d5841e2186ab5",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/vlpart/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        url="https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth",
        md5="c2b4865ad03db1cf00d5da921a72c5de",
    )

    # MaskDINO
    # instance segmentation and object detection
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/instance/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
        md5="0542c05ce5eef21d7073e421cec3bf16",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/instance/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
        md5="443a397e48f2901f115d514695c27097",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_mask52.1ap_box58.3ap.pth",
        md5="af112f2b59607b95bbbc7b285efb561d",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/instance/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",
        md5="05307c4356d20a258fdd96a8dade61b1",
    )

    # semantic segmentation
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_ade20k_48.7miou.pth",
        md5="2b89bb1950174eaff27c8acfa6203efe",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/semantic/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_100q_celoss_hid1024_3s_semantic_cityscapes_79.8miou.pth",
        md5="8debe34409d4a9ca1add8ba72b311397",
    )

    # panoptic segmentation
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/panoptic/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_panoptic_pq53.0.pth",
        md5="1cbe19c16f4aacd64431e01c2b076ce8",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/MaskDINO/panoptic/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth",
        url="https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_panoptic_58.3pq.pth",
        md5="7f5c513abf9d42aba6e04e428bc8baac",
    )

    # OneFormer
    # ADE20K
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth",
        md5="0621b4c08b7ccd34004e71c14d2765d7",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/896x896_250_16_swin_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_swin_l_oneformer_ade20k_160k.pth",
        md5="13f323987e60357aacb31024bdceb751",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/1280x1280_250_16_swin_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_swin_l_oneformer_ade20k_160k.pth",
        md5="26d9ff04b29bff9cad14bebcbe32bd75",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/250_16_convnext_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_l_oneformer_ade20k_160k.pth",
        md5="efe9472bcff45b69079ed4148cc00e1c",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth",
        md5="36d47ad7d9d12a9a271bb13d716f8ee6",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/896x896_250_16_dinat_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/896x896_250_16_dinat_l_oneformer_ade20k_160k.pth",
        md5="7e1d8e53f307bc423c4e0518bab625e6",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/1280x1280_250_16_dinat_l_oneformer_ade20k_160k.pth",
        md5="288aab3cc897c39827824509e46e51e1",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/coco_pretrain_1280x1280_150_16_dinat_l_oneformer_ade20k_160k.pth",
        md5="994d0bf7c7909bd78c60e1bc25b90302",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/ade20k/250_16_convnext_xl_oneformer_ade20k_160k.pth",
        url="https://shi-labs.com/projects/oneformer/ade20k/250_16_convnext_xl_oneformer_ade20k_160k.pth",
        md5="d1f55acd992504786654550261130487",
    )

    # Cityscapes
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/250_16_swin_l_oneformer_cityscapes_90k.pth",
        md5="356b2ae2d52e32ccf39b25c47c044503",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/250_16_convnext_l_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_l_oneformer_cityscapes_90k.pth",
        md5="54aff774f23954399708859d63323b28",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/mapillary_pretrain_250_16_convnext_l_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_l_oneformer_cityscapes_90k.pth",
        md5="791165a275fe71fe52e2d51c358e1907",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/250_16_dinat_l_oneformer_cityscapes_90k.pth",
        md5="b4df7721bac0b4971c88e95079e44714",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/250_16_convnext_xl_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/250_16_convnext_xl_oneformer_cityscapes_90k.pth",
        md5="c800e676674857da2a8e59c03aa5a27d",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/cityscapes/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth",
        url="https://shi-labs.com/projects/oneformer/cityscapes/mapillary_pretrain_250_16_convnext_xl_oneformer_cityscapes_90k.pth",
        md5="a7bca52cff480aedace71a1dccb8f794",
    )

    # COCO
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/coco/150_16_swin_l_oneformer_coco_100ep.pth",
        url="https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth",
        md5="db975d7a4b42ba29f63e64921ec96240",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/coco/150_16_dinat_l_oneformer_coco_100ep.pth",
        url="https://shi-labs.com/projects/oneformer/coco/150_16_dinat_l_oneformer_coco_100ep.pth",
        md5="91610efcf17b14776ae6dc10ee4b0642",
    )

    # Mapillary Vistas
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/mapillary/250_16_swin_l_oneformer_mapillary_300k.pth",
        url="https://shi-labs.com/projects/oneformer/mapillary/250_16_swin_l_oneformer_mapillary_300k.pth",
        md5="add55dbe4f0fce3a26a565d54123ecd7",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/mapillary/250_16_convnext_l_oneformer_mapillary_300k.pth",
        url="https://shi-labs.com/projects/oneformer/mapillary/250_16_convnext_l_oneformer_mapillary_300k.pth",
        md5="617f1627a76d069757af3d507e23a265",
    )
    download_data(
        pkg_name=PKG,
        path="trained_data/OneFormer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth",
        url="https://shi-labs.com/projects/oneformer/mapillary/250_16_dinat_l_oneformer_mapillary_300k.pth",
        md5="691a6e7e4243cc58b6fa497f2107b735",
    )


if __name__ == "__main__":
    main()
