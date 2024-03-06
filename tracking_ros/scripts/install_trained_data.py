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


if __name__ == "__main__":
    main()
