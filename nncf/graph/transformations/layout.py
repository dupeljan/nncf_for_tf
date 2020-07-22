"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


class TransformationLayout:
    def __init__(self):
        self._transformations = []

    @property
    def transformations(self):
        return self._transformations

    def register(self, transformation):
        def find_transformation(transformation):
            for idx, t in enumerate(self.transformations):
                if t.type == transformation.type and \
                        t.target_point == transformation.target_point:
                    return idx
            return None

        idx = find_transformation(transformation)
        if idx is None:
            self.transformations.append(transformation)
        else:
            self.transformations[idx] = self.transformations[idx] + transformation
