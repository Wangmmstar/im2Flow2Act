# Copyright 2023 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import pybullet as pb


def compute_collision_mesh(source_path: str, target_path: str, stdout_path: str):
  pb.vhacd(source_path, target_path, stdout_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_path', type=str)
  parser.add_argument('--target_path', type=str)
  parser.add_argument('--stdout_path', type=str)
  args = parser.parse_args()

  compute_collision_mesh(stdout_path=args.stdout_path,
                         source_path=args.source_path,
                         target_path=args.target_path)
