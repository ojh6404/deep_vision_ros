#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

_TRACKING_ROS_ROOT_INSIDE_CONTAINER = "/home/user/tracking_ws/src/tracking_ros"


def add_prefix(file_path: Path, prefix: str) -> Path:
    parent = file_path.parent
    return parent / (prefix + file_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-mount", type=str, help="mount source launch file or directory"
    )
    parser.add_argument("-name", type=str, help="launch file name")
    parser.add_argument(
        "-host", type=str, default="pr1040", help="host name or ip-address"
    )
    parser.add_argument(
        "launch_args",
        nargs=argparse.REMAINDER,
        help="launch args in ros style e.g. foo:=var",
    )
    args = parser.parse_args()

    mount_path_str: Optional[str] = args.mount
    assert mount_path_str is not None
    mount_path = Path(mount_path_str)

    launch_file_name: Optional[str] = args.name
    assert launch_file_name is not None

    for launch_arg in args.launch_args:
        assert bool(re.match(r".*:=.*", launch_arg))
    launch_args = " ".join(args.launch_args)

    with TemporaryDirectory() as td:
        tmp_launch_path = Path(td) / "launch"

        if mount_path.is_dir():
            shutil.copytree(mount_path, tmp_launch_path)
        else:
            shutil.copyfile(mount_path, tmp_launch_path)

        docker_run_command = """
            docker run \
                -v {tmp_launch_path}:{tracking_ros_root}/launch \
                --rm --net=host -it \
                --gpus 1 tracking_ros:latest \
                /bin/bash -i -c \
                "source ~/.bashrc; \
                roscd tracking_ros; \
                rossetip; rossetmaster {host}; \
                roslaunch tracking_ros {launch_file_name} {launch_args}"
                """.format(
            tmp_launch_path=tmp_launch_path,
            tracking_ros_root=_TRACKING_ROS_ROOT_INSIDE_CONTAINER,
            host=args.host,
            launch_file_name=launch_file_name,
            launch_args=launch_args,
        )
        subprocess.call(docker_run_command, shell=True)
