#!/usr/bin/env python

import argparse
import multiprocessing

import jsk_data

def download_data(*args, **kwargs):
    p = multiprocessing.Process(
            target=jsk_data.download_data,
            args=args,
            kwargs=kwargs)
    p.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest='quiet', action='store_false')
    args = parser.parse_args()
    args.quiet

    PKG = 'tracking_ros'

    # segment anything
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_b.pth',
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        md5='01ec64d29a2fca3f0661936605ae66f8',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_l.pth',
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
        md5='0b3195507c641ddb6910d2bb5adee89c',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_h.pth',
        url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        md5='4b8939a88964f0f4ff5f5b2642c598a6',
    )

    # segment anything hq
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_b_hq.pth',
        url='https://drive.google.com/uc?id=11yExZLOve38kRZPfRx_MRxfIAKmfMY47',
        md5='c6b8953247bcfdc8bb8ef91e36a6cacc',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_l_hq.pth',
        url='https://drive.google.com/uc?id=1Uk17tDKX1YAKas5knI4y9ZJCo0lRVL0G',
        md5='08947267966e4264fb39523eccc33f86',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/sam_vit_h_hq.pth',
        url='https://drive.google.com/uc?id=1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8',
        md5='3560f6b6a5a6edacd814a1325c39640a',
    )

    # mobile sam
    download_data(
        pkg_name=PKG,
        path='trained_data/sam/mobile_sam.pth',
        url='https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/mobile_sam.pt',
        md5='f3c0d8cda613564d499310dab6c812cd',
    )

    # cutie
    download_data(
        pkg_name=PKG,
        path='trained_data/cutie/cutie-base-mega.pth',
        url='https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
        md5='a6071de6136982e396851903ab4c083a',
    )

    # DEVA
    download_data(
        pkg_name=PKG,
        path='trained_data/deva/DEVA-propagation.pth',
        url='https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth',
        md5='a614cc9737a5b4c22ecbdc93e7842ecb',
    )

    # grounding dino
    download_data(
        pkg_name=PKG,
        path='trained_data/groundingdino/groundingdino_swint_ogc.pth',
        url='https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
        md5='075ebfa7242d913f38cb051fe1a128a2',
    )
    download_data(
        pkg_name=PKG,
        path='trained_data/groundingdino/GroundingDINO_SwinT_OGC.py',
        url='https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/GroundingDINO_SwinT_OGC.py',
        md5='bdb07fc17b611d622633d133d2cf873a',
    )

if __name__ == '__main__':
    main()
